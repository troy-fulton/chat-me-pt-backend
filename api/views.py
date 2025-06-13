import os
from datetime import datetime

from rest_framework import status
from rest_framework.permissions import AllowAny
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import ChatMessage, Conversation, Visitor

WELCOME_MESSAGE = os.getenv("WELCOME_MESSAGE", "Hello! How can I assist you today?")


class WelcomeView(APIView):
    permission_classes = [AllowAny]

    def post(self, request: Request) -> Response:
        name = request.data.get("name")
        if name is not None:
            name = str(name).strip()
        interests = request.data.get("interests")
        interest_list = []
        if interests is not None:
            interests = str(interests).strip()
            interest_list = interests.split(",")
        company = request.data.get("company")
        if company is not None:
            company = str(company).strip()

        session_id = request.session.session_key
        if not session_id:
            request.session.create()
            session_id = request.session.session_key

        visitor, _ = Visitor.objects.get_or_create(session_id=session_id)
        visitor.name = name
        visitor.interests = interest_list
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

        # Limit to 5 active conversations
        active_conversations = Conversation.objects.filter(
            visitor=visitor, ended_at__isnull=True
        ).count()
        if active_conversations >= 5:
            return Response(
                {"error": "You have reached the limit of 5 active conversations."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        conversation = Conversation.objects.create(visitor=visitor)
        return Response(
            {
                "message": WELCOME_MESSAGE,
                "conversation_id": conversation.id,
            },
            status=status.HTTP_200_OK,
        )


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
                }
                for msg in messages
            ]
        return Response({"messages": message_data}, status=status.HTTP_200_OK)

    def post(self, request: Request) -> Response:
        """
        Accepts a POST request with a 'message' key.
        Returns a dummy chatgpt-like response (echoes or returns a canned response).
        """
        try:
            visitor = get_visitor(request)
        except Visitor.DoesNotExist:
            return Response({"redirect": "/welcome"}, status=status.HTTP_302_FOUND)

        conversation_id = request.data.get("conversation_id")
        if conversation_id is None:
            return Response(
                {"error": "conversation_id query parameter is required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        conversation = self.find_conversation(visitor, conversation_id)
        if not conversation:
            return Response(
                {"error": "Conversation not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        user_message = request.data.get("message")
        if not isinstance(user_message, str):
            return Response(
                {"error": "Message is required."}, status=status.HTTP_400_BAD_REQUEST
            )

        # Placeholder for a real LLM backend
        response_text = f"You said: {user_message}"

        ChatMessage.objects.create(
            conversation=conversation,
            content=user_message,
            role="visitor",
            token_count=len(user_message.split()),
        )
        ChatMessage.objects.create(
            conversation=conversation,
            content=response_text,
            role="assistant",
            token_count=len(response_text.split()),
        )

        return Response({"response": response_text}, status=status.HTTP_200_OK)
