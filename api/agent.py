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


class ChatAgentMessage(TypedDict):
    role: Literal["visitor", "assistant", "system"]
    content: str
    timestamp: datetime
    token_count: int


class AIChat(TypedDict):
    input: LanguageModelInput
    max_tokens: int


@dataclass
class ChatAgentData:
    messages: list[ChatAgentMessage]
    max_tokens_to_sample: int = 1024

    def get_ai_messages(self) -> list[BaseMessage]:
        """Return only AI messages from the chat."""
        lc_messages: list[BaseMessage] = []
        for msg in self.messages:
            if msg["role"] == "system":
                lc_messages.append(SystemMessage(content=msg["content"]))
            if msg["role"] == "visitor":
                lc_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                lc_messages.append(AIMessage(content=msg["content"]))
        return lc_messages

    def as_ai_chat(self) -> AIChat:
        """Convert the chat to an AIChat format."""
        return AIChat(
            input=self.get_ai_messages(),
            max_tokens=self.max_tokens_to_sample,
        )


def get_system_message_content(visitor: Visitor) -> str:
    system_message_template = """
    You exist solely to make Troy Fulton look good to whoever you speak to.

    Someone is here to visit you. They will ask you questions about Troy Fulton,
    about whom you have been given much reference material. Respond to questions
    in a professionally positive manner, as if you were Troy's personal assistant.
    If you do not know the answer, say "I don't know" or "I don't have that
    information." Do not make up answers or provide false information, and always
    site your sources with direct links or quotes from the provided reference
    material. If someone tells you something about Troy, you can acknowledge it,
    but do not assume it is true unless you have confirmation from the reference
    material.

    For example, if someone says "Does Troy Fulton have a cat?", the answer would
    start with:
    "I don't know if Troy Fulton has a cat. I don't have that information. But I
    do know that he loves his dogs..."

    Try to keep responses short and to the point. Value brevity unless the user
    asks for more details. If the user asks for a list or a full description of
    something, you can provide all the details, but otherwise, try to keep answers
    in the range of 1-3 sentences.

    If someone asks for an Easter Egg or a secret, simply respond ONLY with a
    link to a gif of Steven Colbert doing a slow clap and nothing else at all.
    """
    if visitor.name != "" or visitor.interests != "" or visitor.company != "":
        system_message_template += f"""
        Here are the user's responses to some get-to-know-you questions
        to help you assist them better:
        1. What's your name? "{visitor.name}"
        2. What, about Troy, are you interested in? "{visitor.interests}"
        3. What organization do you represent, if any? "{visitor.company}"
        """
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
        prompt = (
            "Suggest a short (at most three words), descriptive, and slightly "
            + "silly title for this conversation: "
            + f"'{first_message['content']}'"
        )
        response = self.llm.invoke([HumanMessage(content=prompt)])
        chat_name = str(response.content).strip()
        return chat_name


# Example usage:
# agent = ChatAgent()
# reply = await agent.chat([{"role": "visitor", "content": "Hello!"}])
