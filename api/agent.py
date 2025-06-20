import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import SecretStr

from .models import Visitor

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
MODEL_NAME = os.environ["ANTHROPIC_MODEL_NAME"]
# Maximum length of a visitor message
MAX_MESSAGE_LENGTH = int(os.getenv("MAX_MESSAGE_LENGTH", "1000"))
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
    messages: list[ChatAgentMessage]
    max_tokens_to_sample: int = 1024

    def __post_init__(self) -> None:
        last_visitor_message: ChatAgentMessage | None = None
        for msg in reversed(self.messages):
            if msg["role"] == "visitor":
                last_visitor_message = msg
                break
        if last_visitor_message is None:
            raise ValueError("No visitor messages found in chat data.")
        latest_message = last_visitor_message["content"]
        if len(latest_message) > MAX_MESSAGE_LENGTH:
            raise ChatMessageTooLongException(latest_message, MAX_MESSAGE_LENGTH)

    def sanitize_visitor_message(self, msg_content: str) -> str:
        """Sanitize message to ensure it is safe for processing."""
        if len(msg_content) > MAX_MESSAGE_LENGTH:
            raise ChatMessageTooLongException(msg_content, MAX_MESSAGE_LENGTH)
        return json.dumps({"message": msg_content})

    def get_ai_messages(self) -> list[BaseMessage]:
        """Return only AI messages from the chat."""
        lc_messages: list[BaseMessage] = []
        for msg in self.messages:
            msg_content = msg["content"]
            if msg["role"] == "system":
                lc_messages.append(SystemMessage(content=msg_content))
            if msg["role"] == "visitor":
                safe_msg_content = self.sanitize_visitor_message(msg_content)
                lc_messages.append(HumanMessage(content=safe_msg_content))
            elif msg["role"] == "assistant":
                lc_messages.append(AIMessage(content=msg_content))
        return lc_messages

    def as_ai_chat(self) -> AIChat:
        """Convert the chat to an AIChat format."""
        return AIChat(
            input=self.get_ai_messages(),
            max_tokens=self.max_tokens_to_sample,
        )


def get_system_message_content(visitor: Visitor) -> str:
    system_message_template = """
    You are an portfolio search agent for Troy Fulton.

    You will be given ePortfolio content about Troy Fulton, and you will be
    prompted with questions about Troy that may or may not be answered in the
    ePortfolio content.

    You will chat with a user asking questions about Troy Fulton. Respond to
    questions in a professionally positive tone, as if you were Troy's personal
    assistant. When considering the question, always rely purely on the
    ePortfolio content:

    * If you have the materials to answer the question, provide a
        concise, accurate, and relevant answer based on the ePortfolio content.
        For example, if you are asked "Does Troy Fulton have a dog?", and you
        have at least one document that mentions Troy's dog, your response could
        start with: "Yes, Troy Fulton has a dog named..." Always cite your
        sources with direct links or quotes copied exactly, verbatim from the
        supporting reference material.

    * If you do not have the materials to answer the question, give an
        evidence-based response like "I don't have that information." Do not
        hallucinate answers or provide false information. For example, if
        someone asks "Does Troy Fulton have experience with deploying to AWS?",
        and you do not have any documents that mention AWS or related
        information, your response should be worded in a way that informs the
        user what you looked for in the ePortfolio content, and that you could
        not find any information about that topic. For example, you could say
        something that starts like this: "I don't know if Troy Fulton has
        experience with deploying to AWS because I don't have any information
        about that in the ePortfolio content. But I do know that he has
        experience with deploying to Azure..."

    Value brevity, and limit them to about 1-3 concise sentences. Only elaborate
    beyond 3 sentences if the user asks for:
    * More context, clarification, or information than what was provided in the
        1-3 sentences.
    * An exhaustive list
    * A specific length of a response, not to exceed 2 paragraphs

    If someone asks for an Easter Egg or a secret, simply respond ONLY with a
    link to a gif of Steven Colbert doing a slow clap and nothing else at all.
    """

    input_format_spec = """

    Messages will be formatted as a JSON objects like this:

    {"message": "User message here"}
    """
    if visitor.name != "" or visitor.interests != "" or visitor.company != "":
        system_message_template += f"""

        Here are the user's responses to some get-to-know-you questions
        to help you assist them better:
        1. What's your name? "{visitor.name}"
        2. What, about Troy, are you interested in? "{visitor.interests}"
        3. What organization do you represent, if any? "{visitor.company}"
        """
    system_message_template += input_format_spec
    return system_message_template.strip()


class ChatAgent:
    chat_data: ChatAgentData
    llm: ChatAnthropic
    chat_start: bool
    visitor: Visitor

    def __init__(self, chat_data: ChatAgentData, visitor: Visitor):
        self.chat_data = chat_data
        self.llm = ChatAnthropic(
            api_key=SecretStr(ANTHROPIC_API_KEY),
            model_name=MODEL_NAME,
            timeout=60,
            stop=None,
            max_tokens_to_sample=chat_data.max_tokens_to_sample,
        )
        self.chat_start = len(chat_data.messages) == 1
        self.visitor = visitor

    def chat(self) -> str:
        """
        messages: list of dicts, e.g. [{"role": "user", "content": "Hello!"}]
        Returns: assistant's reply as string
        """
        # Convert messages to LangChain format
        response = self.llm.invoke(**self.chat_data.as_ai_chat())
        return str(response.content)

    def name_chat(self) -> str:
        """
        Returns a name for the chat based on the first visitor message.
        """
        if not self.chat_data.messages:
            return "Unnamed Chat"

        visitor_messages = [
            msg for msg in self.chat_data.messages if msg["role"] == "visitor"
        ]
        if not visitor_messages:
            return "Unnamed Chat"
        first_message = visitor_messages[0]
        first_message_content = (
            first_message["content"][:MAX_TITLE_MESSAGE_LENGTH] + "..."
        )
        prompt = (
            "Suggest a short (at most three words), descriptive, and slightly "
            + f"silly title for this conversation: '{first_message_content}'"
        )
        response = self.llm.invoke([HumanMessage(content=prompt)])
        chat_name = str(response.content).strip()
        return chat_name


# Example usage:
# agent = ChatAgent()
# reply = await agent.chat([{"role": "visitor", "content": "Hello!"}])
