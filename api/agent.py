import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, TypedDict

from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableSerializable,
)
from pydantic import SecretStr
from transformers.pipelines import pipeline
from whoosh.index import FileIndex
from whoosh.qparser import FuzzyTermPlugin, MultifieldParser, PrefixPlugin, QueryParser

from .document_indexer import DirectoryRAGIndexer
from .models import ChatMessage, Visitor

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
MODEL_NAME = os.environ["ANTHROPIC_MODEL_NAME"]
HUGGINGFACE_HUB_TOKEN = os.environ["HUGGINGFACE_HUB_TOKEN"]
# Maximum length of a chat message to use for the title prompt
MAX_TITLE_MESSAGE_LENGTH = int(os.getenv("MAX_TITLE_MESSAGE_LENGTH", "100"))
DOC_DIRECTORY = os.environ["DOCUMENTS_DIRECTORY"]
DOC_INDEX_PATH = os.environ["DOCUMENT_INDEX_PATH"]
DOC_SCORE_THRESHOLD = float(os.getenv("DOC_SCORE_THRESHOLD", "0.6"))
rag_indexer = DirectoryRAGIndexer(DOC_DIRECTORY, doc_index_path=DOC_INDEX_PATH)

qp = MultifieldParser(["content", "description"], schema=rag_indexer.whoosh_schema())
qp.add_plugin(FuzzyTermPlugin())
qp.add_plugin(PrefixPlugin())
document_index: FileIndex | None = None

print("Loading Prompt-Guard-86M pipeline...")
prompt_guard_classifier = pipeline(
    "text-classification",
    model="meta-llama/Llama-Prompt-Guard-2-86M",
    top_k=None,
    token=HUGGINGFACE_HUB_TOKEN,
)


def get_document_index() -> FileIndex:
    """
    Returns the Whoosh index for the documents.
    This is used to search for relevant documents based on user queries.
    """
    global document_index
    if document_index is None:
        document_index = rag_indexer.get_whoosh_index()
    return document_index


class ChatAgentMessage(TypedDict):
    role: Literal["visitor", "assistant", "system"]
    content: str
    timestamp: datetime
    token_count: int


class AIChat(TypedDict):
    input: LanguageModelInput
    max_tokens: int


class TokenUsage(TypedDict):
    input_tokens: int
    output_tokens: int


class ChatException(Exception):
    """Base class for chat-related exceptions."""

    pass


class ChatMessageTooLongException(ChatException):
    """Raised when a chat message exceeds the maximum allowed length."""

    def __init__(self, message: str, max_length: int):
        message_length = len(message)
        trunc_message = (
            message[:max_length] + "..." if message_length > max_length else message
        )
        super().__init__(
            f"Message of length {message_length} exceeds {max_length} characters. "
            + f"Truncated message: {trunc_message}"
        )
        self.message = message
        self.max_length = max_length


class ChatResponse(TypedDict):
    response: str
    relevant_documents: list[str]
    token_usage: TokenUsage


@dataclass
class ChatAgentData:
    messages: list[ChatMessage]
    max_tokens_to_sample: int = 1024

    def get_ai_messages(self) -> list[BaseMessage]:
        lc_messages: list[BaseMessage] = []
        for msg in self.messages:
            msg_content = msg.content
            if msg.role == "system":
                lc_messages.append(SystemMessage(content=msg_content))
            if msg.role == "visitor":
                lc_messages.append(HumanMessage(content=msg_content))
            elif msg.role == "assistant":
                lc_messages.append(AIMessage(content=msg_content))
        return lc_messages

    def get_tuple_messages(self) -> list[tuple[Literal["system", "human", "ai"], str]]:
        lc_messages: list[tuple[Literal["system", "human", "ai"], str]] = []
        for msg in self.messages:
            msg_content = msg.content
            if msg.role == "system":
                lc_messages.append(("system", msg_content))
            if msg.role == "visitor":
                lc_messages.append(("human", msg_content))
            elif msg.role == "assistant":
                lc_messages.append(("ai", msg_content))
        return lc_messages

    def as_ai_chat(self) -> AIChat:
        """Convert the chat to an AIChat format."""
        return AIChat(
            input=self.get_ai_messages(),
            max_tokens=self.max_tokens_to_sample,
        )


class ChatAgent:
    chat_history_data: ChatAgentData
    llm: ChatAnthropic
    visitor: Visitor

    def __init__(self, chat_data: ChatAgentData, visitor: Visitor) -> None:
        self.chat_history_data = chat_data
        self.llm = ChatAnthropic(
            api_key=SecretStr(ANTHROPIC_API_KEY),
            model_name=MODEL_NAME,
            timeout=60,
            stop=None,
            max_tokens_to_sample=chat_data.max_tokens_to_sample,
            temperature=0,
        )
        self.visitor = visitor

    def get_system_message_template(self) -> str:
        system_message_template = """
You are an AI chat assistant designed to answer questions about Troy Fulton's
ePortfolio. You will answer the user's most recent prompt based ONLY on the
EXACT content of documents that contain information about Troy. Another agent
has curated a query to retrieve documents about Troy based on the user's input.
You will use these documents as context when responding to user prompts.

Respond in a professionally positive tone, and do not continue the conversation
on your own. If you have no documents to support your response, it does not mean
that those documents don't exist. It just means that the query did not result in
the right documents.

* Value brevity above all else. Always limit responses to 4 concise sentences or
    3 short bullet points at most. Only elaborate beyond that if the user
    specifically asks for a longer response.

* Documents are provided to you in order of relevance and priority.

* If you find evidence in the document(s) to answer the prompt, provide a
    short, contextually accurate, and relevant answer. Use in-line citations by
    including the file name in parentheses, like this: (File: filename.ext).

* If you do not find evidence in the document(s) to answer the prompt, give an
    evidence-based response like "Based on the documents resulting from the
    query, I do not have any information about that."

* End each response with an emoji that matches the tone of your response.

Conversations will be wrapped in an XML tag for clarity using the <chat> tag,
but you will only respond with the content of your singular response to the most
recent human prompt. Do not include any XML/HTML in your response.

Here is an example conversation to help you understand how to respond. It
contains hypothetical messages and documents, but you should not use this as
context for your responses:
<chat>
"""

        example_conversation = [
            ("human", "What experience does Troy have with Python?"),
            (
                "ai",
                "Searching for documents using this query: "
                + "Python AND (programming OR coding)",
            ),
            (
                "system",
                "[File: school_project.pptx, Description: Troy's CSCE Project] "
                + "This is Troy's presentation from his CSCE 1200 class where he "
                + "designed a Python program to analyze data.\n"
                + "[File: puppies.png, Description: A photo of Troy's dogs] "
                + "This is a photo of Troy's dogs.",
            ),
            (
                "ai",
                "According to Troy's CSCE Project (File: school_project.pptx), "
                + "he has experience with Python programming from his CSCE 1200 class, "
                + "where he used Python to analyze data. 🐍",
            ),
            ("human", "What other programming languages does Troy know?"),
            (
                "ai",
                "Searching for documents using this query: "
                "language AND (programming OR coding OR software OR computer)",
            ),
            (
                "system",
                "[File: school_project.pptx, Description: Troy's CSCE Project] "
                + "This is Troy's presentation from his CSCE 1200 class where he "
                + "designed a Python program to analyze data.\n"
                + "[File: engr_project.pptx, Description: Troy's Engineering Project] "
                + "This is Troy's presentation from his Engineering class where he "
                + "designed a Java program to simulate a circuit.",
            ),
        ]
        system_message_template += ChatPromptTemplate.from_messages(
            example_conversation
        ).format()
        system_message_template += (
            "\n</chat>\n\nHere is how you should respond, "
            + "given the hypothetical context above:\n"
        )
        system_message_template += (
            "According to Troy's Engineering Project (File: engr_project2.pptx), "
            + "he also has experience with Java programming from his Engineering "
            + "class, where he used Java to simulate a circuit. ⚙️\n\n"
        )
        if (
            self.visitor.name != ""
            or self.visitor.interests != ""
            or self.visitor.company != ""
        ):
            system_message_template += f"""Here is some user data:
Name: "{self.visitor.name}"
Interests: "{self.visitor.interests}"
Organization: "{self.visitor.company}"

"""

        system_message_template += "Below is a recent history of the chat.\n\n"
        system_message_template += "<chat>\n"
        system_message_template = system_message_template.strip()
        return system_message_template

    def format_documents(self, documents: list[Document]) -> str:
        file_descriptions = list()
        for doc in documents:
            file_name = doc.metadata.get("file", "Unknown File")
            description = doc.metadata.get("description", "")
            content = doc.page_content.strip()
            file_descriptions.append(
                f"[File: {file_name}, Description: {description}] {content}"
            )
        return "\n\n".join(file_descriptions)

    def generate_chat_history(
        self,
    ) -> list[tuple[Literal["system", "human", "ai"], str]]:
        """
        Generate the chat history in LangChain format.
        This includes the system message, visitor messages, and assistant
        messages.

        If the chat history does not start with a system message, it will
        prepend a system message based on the provided template.
        """
        chat_history = self.chat_history_data.get_tuple_messages()
        if len(chat_history) == 0 or chat_history[0][0] != "system":
            system_message_template = self.get_system_message_template()
            chat_history.insert(0, ("system", system_message_template))
        return chat_history

    def get_doc_retrieval_prompt_template(self) -> str:
        return """
You are an expert at interpreting user prompts, determining keywords to search
on, and rewriting them into valid Whoosh keyword search queries for retrieving
documents about Troy Fulton from his ePortfolio. Your job is to follow these
strict instructions:

    1. If the user prompt is NOT asking about Troy, reply with exactly: N/A

    2. If the user prompt IS asking about Troy, rewrite the prompt into a
        concise Whoosh keyword search query using only keywords and quoted
        multi-word phrases.
       - ALWAYS enclose every multi-word phrase in double quotes, even if it
         appears in parentheses or as part of a logical expression.
       - NEVER leave a multi-word phrase unquoted.
       - Use logical operators (AND, OR) as appropriate, and group terms with
         parentheses if needed.
       - Do NOT include any punctuation, special characters, or quotes except
         for quoting multi-word phrases.
       - Unless the user is specifically asking about Troy's name or about Troy
         in general, do NOT include "Troy" or "Fulton" in the query.
       - Expand broad queries into up to 10 relevant keywords or quoted phrases,
         including synonyms if helpful.
       - Include only keywords likely to be in the documents. That is, do NOT
         include stop words or question words like "what" or "how" or "and."

    3. Reply ONLY with the rewritten query, with no explanation or extra text.

    Examples:

    User: Does Troy have any experience with Python or Java?
    Assistant: Python OR Java

    User: Does Troy have pets?
    Assistant: pet OR animal OR cat OR dog OR fish OR bird OR hamster OR rabbit

    User: Who are you?
    Assistant: N/A

    User: Who is Troy?
    Assistant: Troy OR Fulton

    User: What does Troy do for fun?
    Assistant: fun OR hobbies OR interests OR leisure OR activity

    User: What is Troy's favorite color?
    Assistant: color AND (favorite OR preferred OR like OR love)

    User: What did Troy do in high school?
    Assistant: "high school" AND (experience OR activities OR clubs OR sports OR events)

    User: Tell me about Troy's software engineering projects.
    Assistant: "software engineering" OR project OR application OR code
            OR development OR "software project"

    User: what about his personal life?
    Assistant: "personal life" OR family OR friends OR relationships OR hobbies

    REMEMBER: If you use a phrase with more than one word, it MUST be in double
    quotes. If you do not follow this, the query will be invalid and will cause
    a critical error.

    REMEMBER: Only include meaningful keywords and phrases and do NOT include
    question words or stop words like "who" or "what". Doing so will result in a
    nuclear meltdown.

    The user message is: {user_input}
    """

    def get_doc_retrieval_query(self, user_input: str) -> tuple[str, TokenUsage]:
        """
        Generate a prompt for document retrieval based on the user input.
        This prompt is used to rewrite the user message into a query for
        the document store.
        """
        doc_retrieval_prompt = self.get_doc_retrieval_prompt_template()
        num_input_tokens = self.llm.get_num_tokens(
            doc_retrieval_prompt.format(user_input=user_input)
        )
        chain: RunnableSerializable[dict[str, str], str] = (
            PromptTemplate.from_template(doc_retrieval_prompt)
            | self.llm
            | StrOutputParser()
        )
        # Invoke the chain with the user input to get the rewritten query
        rewritten_query = chain.invoke({"user_input": user_input})
        num_output_tokens = self.llm.get_num_tokens(rewritten_query)
        return rewritten_query.strip(), {
            "input_tokens": num_input_tokens,
            "output_tokens": num_output_tokens,
        }

    def get_documents(self, document_search_prompt: str) -> list[Document]:
        q = qp.parse(document_search_prompt)
        documents = []
        doc_index = get_document_index()
        with doc_index.searcher() as searcher:
            results = searcher.search(q)
            if not results:
                return []
            for result in results:
                document = Document(
                    page_content=result["content"],
                    metadata={
                        "file": result["filepath"],
                        "description": result["description"],
                        "priority": result["priority"],
                        "score": result.score,
                    },
                )
                documents.append(document)

        return sorted(documents, key=lambda doc: doc.metadata.get("priority", 100))

    def get_resume(self) -> Document:
        """
        Retrieve the resume document from the index.
        Returns the resume document if found, otherwise raises an error.
        """
        doc_index = get_document_index()
        with doc_index.searcher() as searcher:
            query = QueryParser("filepath", schema=doc_index.schema).parse("resume.pdf")
            results = searcher.search(query)
            if results:
                result = results[0]
                return Document(
                    page_content=result["content"],
                    metadata={
                        "file": result["filepath"],
                        "description": result["description"],
                        "priority": result["priority"],
                        "score": result.score,
                    },
                )
        raise ValueError("Resume document not found.")

    def chat(self, visitor_message: str, document_query: str | None) -> ChatResponse:
        """
        Process the chat messages and return the assistant's reply.
        This method expects the chat_data to contain messages in LangChain format.
        It uses the LLM to generate a response based on the provided messages.

        NOTE: This method assumes that visitor_message is already validated as
        safe from malicious content, such as prompt injection.
        """
        chat_history = self.generate_chat_history()
        relevant_docs = [self.get_resume()]
        if document_query is not None:
            print(f"Retrieving relevant documents... using query: {document_query}")
            # Retrieve relevant documents using the query
            retrieved_docs = self.get_documents(document_query)
            # Check if the first document is the resume (by filename)
            if (
                len(retrieved_docs) >= 1
                and retrieved_docs[0].metadata.get("file", "") == "resume.pdf"
            ):
                # If resume is already included, limit to top 3 documents
                relevant_docs = retrieved_docs[:3]
            else:
                # Otherwise, ensure resume is included at the start, then top 2 others
                relevant_docs += retrieved_docs[:2]
        context_block = (
            self.format_documents(relevant_docs)
            if relevant_docs
            else "No documents found."
        )
        chat_history.append(("system", context_block))
        chat_prompt = ChatPromptTemplate(chat_history, input_variables=["user_input"])
        llm_prompt = chat_prompt.format()
        llm_prompt += "\n</chat>\n\nRemember to respond only with a single message."
        print("Invoking the LLM...")
        rag_response = self.llm.invoke(
            llm_prompt,
            max_tokens=self.chat_history_data.max_tokens_to_sample,
        )
        # Ensure the response is a string (handles both message objects and
        # plain strings)
        response_string = rag_response.content
        if not isinstance(response_string, str):
            raise TypeError(
                f"Expected response content to be a string, got {type(response_string)}"
            )
        return {
            "response": response_string,
            "relevant_documents": [doc.metadata["file"] for doc in relevant_docs],
            "token_usage": {
                "input_tokens": self.llm.get_num_tokens(llm_prompt),
                "output_tokens": self.llm.get_num_tokens(response_string),
            },
        }

    def name_chat(self, first_message: str) -> str:
        """
        Returns a name for the chat based on the first visitor message.
        """
        first_message_content = first_message[:MAX_TITLE_MESSAGE_LENGTH]
        if len(first_message) > MAX_TITLE_MESSAGE_LENGTH:
            first_message_content += "..."
        prompt = (
            "Suggest a short (at most three words), descriptive, and slightly "
            + f"silly title for this conversation: '{first_message_content}'"
        )
        response = self.llm.invoke([HumanMessage(content=prompt)])
        chat_name = str(response.content).strip()
        return chat_name

    def detect_malicious_prompt(self, prompt: str) -> tuple[bool, Any]:
        """
        Uses Prompt-Guard-86M to detect if a prompt is malicious.
        Returns (is_malicious: bool, raw_output: dict).
        """
        try:
            # Load the pipeline (you may want to cache this in production)
            print("Classifying prompt for malicious content...")
            result = prompt_guard_classifier(prompt)
            # result is a list of dicts, one per label
            if (
                not result
                or not isinstance(result, list)
                or not isinstance(result[0], list)
            ):
                return False, {"error": "Invalid response from classifier"}
            first_result = result[0]
            malicious_score = 0.0
            for label_result in first_result:
                if not isinstance(label_result, dict):
                    continue
                if label_result.get("label") == "LABEL_1":
                    malicious_score = label_result.get("score", 0.0)
                    break
            return malicious_score >= 0.9, first_result
        except Exception:
            # If the model fails to load or classify, assume it's not malicious
            print("Failed to classify prompt as malicious or benign.")
            return False, {"error": "Failed to classify prompt"}


# Example usage:
# agent = ChatAgent()
# reply = await agent.chat([{"role": "visitor", "content": "Hello!"}])
