import json
import os
from datetime import datetime

from django.db.models import Sum
from rest_framework import status
from rest_framework.permissions import AllowAny
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from .agent import (
    ChatAgent,
    ChatAgentData,
    ChatMessageTooLongException,
)
from .document_indexer import DirectoryRAGIndexer
from .models import ChatMessage, Conversation, Visitor

MAX_MESSAGE_LENGTH = int(os.getenv("MAX_MESSAGE_LENGTH", "1000"))
MAX_MESSAGE_HISTORY = int(os.getenv("MAX_MESSAGE_HISTORY", "4"))
DOC_DIRECTORY = os.environ["DOCUMENTS_DIRECTORY"]
DOC_INDEX_PATH = os.environ["DOCUMENT_INDEX_PATH"]
DOC_SCORE_THRESHOLD = float(os.getenv("DOC_SCORE_THRESHOLD", "0.5"))
SESSION_HOURLY_TOKEN_LIMIT = int(os.getenv("SESSION_HOURLY_TOKEN_LIMIT", "100000"))
rag_indexer = DirectoryRAGIndexer(DOC_DIRECTORY, doc_index_path=DOC_INDEX_PATH)


class WelcomeView(APIView):
    permission_classes = [AllowAny]

    def get(self, request: Request) -> Response:
        session_id = request.session.session_key
        if not session_id:
            request.session.create()
            session_id = request.session.session_key

        visitor_exists = Visitor.objects.filter(
            session_id=session_id, name__isnull=False
        ).exists()
        return Response({"visitor_exists": visitor_exists}, status=status.HTTP_200_OK)

    def post(self, request: Request) -> Response:
        name = request.data.get("name")
        if name is not None:
            name = str(name).strip()
        interests = request.data.get("interests")
        if interests is not None:
            interests = str(interests).strip()
        company = request.data.get("company")
        if company is not None:
            company = str(company).strip()

        session_id = request.session.session_key
        if not session_id:
            request.session.create()
            session_id = request.session.session_key

        visitor, _ = Visitor.objects.get_or_create(session_id=session_id)
        visitor.name = name
        visitor.interests = interests
        visitor.company = company
        visitor.save()

        return Response(
            {"message": "Visitor information saved."}, status=status.HTTP_200_OK
        )


def get_visitor(request: Request) -> Visitor:
    session_id = request.session.session_key
    return Visitor.objects.get(session_id=session_id)


def get_conversation(visitor: Visitor, end_previous: bool = False) -> Conversation:
    # Get the latest active conversation or create a new one if none exist.
    conversation = (
        Conversation.objects.filter(visitor=visitor, ended_at__isnull=True)
        .order_by("-created_at")
        .first()
    )
    if not conversation:
        conversation = Conversation.objects.create(visitor=visitor)
    elif end_previous:
        conversation.ended_at = datetime.now()
        conversation.save()
        conversation = Conversation.objects.create(visitor=visitor)
    return conversation


class ConversationListView(APIView):
    permission_classes = [AllowAny]

    def get(self, request: Request) -> Response:
        try:
            visitor = get_visitor(request)
        except Visitor.DoesNotExist:
            return Response({"redirect": "/welcome"}, status=status.HTTP_302_FOUND)

        conversations = Conversation.objects.filter(visitor=visitor).order_by(
            "-started_at"
        )
        conversation_data = [
            {
                "id": conv.id,
                "title": conv.title or f"Conversation {conv.id}",
            }
            for conv in conversations
        ]
        return Response({"conversations": conversation_data}, status=status.HTTP_200_OK)

    def post(self, request: Request) -> Response:
        """
        Starts a new conversation for the current visitor
        """
        try:
            visitor = get_visitor(request)
        except Visitor.DoesNotExist:
            return Response({"redirect": "/welcome"}, status=status.HTTP_302_FOUND)

        conversation = Conversation.objects.create(visitor=visitor)
        return Response(
            {
                "conversation_id": conversation.id,
            },
            status=status.HTTP_200_OK,
        )


def get_visitor_remaining_tokens(visitor: Visitor) -> int:
    """
    Returns the remaining tokens for the visitor based on the hourly limit.
    """
    total_tokens = (
        ChatMessage.objects.filter(conversation__visitor=visitor).aggregate(
            total_tokens=Sum("token_count")
        )["total_tokens"]
        or 0
    )
    return max(0, SESSION_HOURLY_TOKEN_LIMIT - total_tokens)


class ChatAPIView(APIView):
    permission_classes = [AllowAny]

    def find_conversation(
        self, visitor: Visitor, conversation_id: int | None
    ) -> Conversation:
        if conversation_id:
            conversation = Conversation.objects.filter(
                id=conversation_id, visitor=visitor
            ).first()
            if not conversation:
                raise Conversation.DoesNotExist
        else:
            conversation = get_conversation(visitor)
        return conversation

    def get(self, request: Request) -> Response:
        """
        Returns the chat history for the current visitor's active conversation.
        If no active conversation exists, returns a welcome message.

        If a conversation ID is provided as a query parameter, it loads that
        conversation instead.
        """
        try:
            visitor = get_visitor(request)
        except Visitor.DoesNotExist:
            return Response({"redirect": "/welcome"}, status=status.HTTP_302_FOUND)

        conversation_id = request.query_params.get("conversation_id")
        if conversation_id is None:
            return Response(
                {"error": "conversation_id query parameter is required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        conversation = Conversation.objects.filter(
            id=conversation_id, visitor=visitor
        ).first()
        if not conversation:
            return Response(
                {"error": "Conversation not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        messages = ChatMessage.objects.filter(conversation=conversation).order_by(
            "timestamp"
        )
        if not messages.exists():
            message_data = []
        else:
            message_data = [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp,
                    "sources": (
                        json.loads(msg.referenced_documents)
                        if msg.referenced_documents
                        else []
                    ),
                }
                for msg in messages
            ]
        return Response({"messages": message_data}, status=status.HTTP_200_OK)

    def get_post_parameters(
        self, request: Request
    ) -> tuple[Visitor, Conversation, str]:
        """
        Helper method to extract visitor and conversation from the request.
        Raises an exception if the visitor or conversation is not found.
        """
        visitor = get_visitor(request)

        conversation_id = request.data["conversation_id"]
        user_message = request.data["message"]
        if len(user_message) > MAX_MESSAGE_LENGTH:
            raise ChatMessageTooLongException(user_message, MAX_MESSAGE_LENGTH)

        conversation = self.find_conversation(visitor, conversation_id)
        return visitor, conversation, user_message

    def validate_remaining_tokens(
        self, message: str, visitor: Visitor, user_message_tokens: int
    ) -> None:
        remaining_tokens = get_visitor_remaining_tokens(visitor)
        if user_message_tokens > remaining_tokens:
            raise ChatMessageTooLongException(message, remaining_tokens)

    def post(self, request: Request) -> Response:
        """
        Accepts a POST request with a 'message' key.
        Returns a dummy chatgpt-like response (echoes or returns a canned response).
        """
        try:
            visitor, conversation, user_message = self.get_post_parameters(request)
        except Visitor.DoesNotExist:
            return Response({"redirect": "/welcome"}, status=status.HTTP_302_FOUND)
        except Conversation.DoesNotExist:
            return Response(
                {"error": "Conversation not found."}, status=status.HTTP_404_NOT_FOUND
            )
        except KeyError as key_error:
            return Response(
                {"error": str(key_error)},
                status=status.HTTP_400_BAD_REQUEST,
            )
        except ChatMessageTooLongException as e:
            return Response(
                {
                    "error": str(e),
                    "message": "Your message is too long. Please shorten it.",
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        user_message_timestamp = datetime.now()
        # Pull the whole conversation history
        messages = ChatMessage.objects.filter(conversation=conversation).order_by(
            "-timestamp"
        )[:MAX_MESSAGE_HISTORY]
        existing_conversation_length = messages.count()
        is_first_message = existing_conversation_length == 0

        retriever = rag_indexer.get_vectorstore().as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": DOC_SCORE_THRESHOLD},
        )
        agent = ChatAgent(
            ChatAgentData(messages=list(messages), max_tokens_to_sample=1024),
            visitor,
            retriever,
        )

        user_message_tokens = agent.llm.get_num_tokens(user_message)
        self.validate_remaining_tokens(user_message, visitor, user_message_tokens)

        ChatMessage.objects.create(
            conversation=conversation,
            content=user_message,
            role="visitor",
            token_count=agent.llm.get_num_tokens(user_message),
            timestamp=user_message_timestamp,
        )

        is_malicious, safety_data = agent.detect_malicious_prompt(user_message)
        if is_malicious:
            return Response(
                {
                    "error": "Your message contains potentially harmful content.",
                    "safety_data": str(safety_data),
                },
                status=status.HTTP_400_BAD_REQUEST,
            )
        try:
            response, relevant_docs, response_num_tokens = agent.chat(user_message)
        except Exception as e:
            return Response(
                {"error": "Failed to generate response.", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        ChatMessage.objects.create(
            conversation=conversation,
            content=response["response"],
            referenced_documents=json.dumps(relevant_docs),
            role="assistant",
            token_count=response_num_tokens,
        )

        if is_first_message:
            # If this is the first message, we can set a title for the conversation
            conversation.title = agent.name_chat()
            conversation.save()

        return Response(
            {"response": response, "sources": relevant_docs}, status=status.HTTP_200_OK
        )


class VisitorUsageAPIView(APIView):
    permission_classes = [AllowAny]

    def get(self, request: Request) -> Response:
        """
        Returns the current visitor's usage statistics.
        """
        try:
            visitor = get_visitor(request)
        except Visitor.DoesNotExist:
            return Response({"redirect": "/welcome"}, status=status.HTTP_302_FOUND)

        return Response(
            {
                "hourly_limit": SESSION_HOURLY_TOKEN_LIMIT,
                "remaining_tokens": get_visitor_remaining_tokens(visitor),
            },
            status=status.HTTP_200_OK,
        )
