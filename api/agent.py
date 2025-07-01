import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, TypedDict

from langchain.prompts import ChatPromptTemplate, FewShotPromptTemplate, PromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import (
    RunnableSerializable,
)
from pydantic import SecretStr
from transformers.pipelines import pipeline

from .examples import examples_json
from .models import ChatMessage, Visitor

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
MODEL_NAME = os.environ["ANTHROPIC_MODEL_NAME"]
HUGGINGFACE_HUB_TOKEN = os.environ["HUGGINGFACE_HUB_TOKEN"]
# Maximum length of a chat message to use for the title prompt
MAX_TITLE_MESSAGE_LENGTH = int(os.getenv("MAX_TITLE_MESSAGE_LENGTH", "100"))


class ChatAgentMessage(TypedDict):
    role: Literal["visitor", "assistant", "system"]
    content: str
    timestamp: datetime
    token_count: int


class AIChat(TypedDict):
    input: LanguageModelInput
    max_tokens: int


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


@dataclass
class ChatAgentData:
    messages: list[ChatMessage]
    max_tokens_to_sample: int = 1024

    def sanitize_visitor_message(self, msg_content: str) -> str:
        """Sanitize message to ensure it is safe for processing."""
        return (
            json.dumps({"message": msg_content}).replace("{", "{{").replace("}", "}}")
        )

    def get_ai_messages(self) -> list[BaseMessage]:
        lc_messages: list[BaseMessage] = []
        for msg in self.messages:
            msg_content = msg.content
            if msg.role == "system":
                lc_messages.append(SystemMessage(content=msg_content))
            if msg.role == "visitor":
                safe_msg_content = self.sanitize_visitor_message(msg_content)
                lc_messages.append(HumanMessage(content=safe_msg_content))
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
                safe_msg_content = self.sanitize_visitor_message(msg_content)
                lc_messages.append(("human", safe_msg_content))
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
    chat_start: bool
    visitor: Visitor
    retriever: BaseRetriever

    def __init__(
        self, chat_data: ChatAgentData, visitor: Visitor, retriever: BaseRetriever
    ) -> None:
        self.chat_history_data = chat_data
        self.llm = ChatAnthropic(
            api_key=SecretStr(ANTHROPIC_API_KEY),
            model_name=MODEL_NAME,
            timeout=60,
            stop=None,
            max_tokens_to_sample=chat_data.max_tokens_to_sample,
            temperature=0,
        )
        self.chat_start = len(chat_data.messages) == 0
        self.visitor = visitor
        self.retriever = retriever

    def get_examples(self) -> str:
        example_prompt = PromptTemplate.from_template(
            """User: {{{{{{{{ "message": "{user_message}" }}}}}}}}
Response: {response}"""
        )

        prompt = FewShotPromptTemplate(
            examples=examples_json,
            example_prompt=example_prompt,
            prefix="Here are some examples of how to respond to questions:",
            suffix="Here is the current conversation history:\n\n",
        )
        return prompt.invoke({}).to_string()

    def get_system_message_template(self) -> str:
        system_message_template = """
You are an assistant designed to answer questions about Troy Fulton's
ePortfolio. The user has asked a question, and based on the question, the
following documents from Troy's ePortfolio are relevant:

{context_block}

If no documents are available, it doesn't mean there are no documents in Troy's
ePortfolio. It means that the user has not prompted you with anything relevant
to the documents in Troy's ePortfolio.

You will chat with a user asking questions about Troy Fulton. Respond to
questions in a professionally positive tone, and limit your answers to those
supported by any documents above. When considering the question:

* If you have the document(s) to answer the question, provide a
    concise, accurate, and relevant answer based ONLY on the ePortfolio
    content, and cite your source in your answer.

* If you do not have the document(s) to answer the question, give an
    evidence-based response like "I couldn't find that information in the
    documents available to me." Do not provide false information or
    information that is not directly supported by the ePortfolio.

* Value brevity, and limit responses to about 1-3 concise sentences. Only elaborate
    beyond 3 sentences if the user specifically asks for:
    * More context, clarification, or information than what was provided in the
        1-3 sentences.
    * An exhaustive list of items
    * A specific length of a response, not to exceed 2 paragraphs

* If someone asks for an Easter Egg or a secret, simply respond ONLY with a
    link to a gif of Steven Colbert doing a slow clap and nothing else at
    all.

User messages will be formatted as a JSON objects like this:

`{{"message": "User message here"}}`

You will respond in plain text without any additional formatting.
        """
        if (
            self.visitor.name != ""
            or self.visitor.interests != ""
            or self.visitor.company != ""
        ):
            system_message_template += f"""

Here are the user's responses to some get-to-know-you questions
to help you assist them better:
1. What's your name? "{self.visitor.name}"
2. What, about Troy, are you interested in? "{self.visitor.interests}"
3. What organization do you represent, if any? "{self.visitor.company}"
            """
        system_message_template += "\n\n" + self.get_examples()
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
        chat_history.append(("human", "{user_input}"))
        return chat_history

    def chat(self, visitor_message: str) -> tuple[str, list[str], int]:
        """
        Process the chat messages and return the assistant's reply.
        This method expects the chat_data to contain messages in LangChain format.
        It uses the LLM to generate a response based on the provided messages.
        """
        chat_history = self.generate_chat_history()
        chat_prompt = ChatPromptTemplate.from_messages(chat_history)
        # Retrieve the most relevant document for the visitor_message
        print("Retrieving relevant documents...")
        relevant_docs = self.retriever.invoke(visitor_message)
        relevant_docs = relevant_docs[:3]  # Limit to top 3 documents
        context_block = (
            self.format_documents(relevant_docs)
            if relevant_docs
            else "No documents found."
        )
        print("Defining RAG chain and invoking...")
        rag_chain: RunnableSerializable[dict[str, str], str] = (
            chat_prompt | self.llm | StrOutputParser()
        )
        # Convert messages to LangChain format
        rag_response = rag_chain.invoke(
            {"user_input": visitor_message, "context_block": context_block},
            max_output_tokens=self.chat_history_data.max_tokens_to_sample,
        )
        response_token_count = self.llm.get_num_tokens(rag_response)
        return (
            rag_response,
            [doc.metadata["file"] for doc in relevant_docs],
            response_token_count,
        )

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
            print("Loading Prompt-Guard-86M pipeline...")
            classifier = pipeline(
                "text-classification",
                model="meta-llama/Llama-Prompt-Guard-2-86M",
                top_k=None,
                token=HUGGINGFACE_HUB_TOKEN,
            )
            print("Classifying prompt for malicious content...")
            result = classifier(prompt)
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
