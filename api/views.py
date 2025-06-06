from rest_framework import status
from rest_framework.permissions import AllowAny
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView


class ChatAPIView(APIView):
    permission_classes = [AllowAny]

    def post(self, request: Request) -> Response:
        """
        Accepts a POST request with a 'message' key.
        Returns a dummy chatgpt-like response (echoes or returns a canned response).
        """
        user_message = request.data.get("message", "")
        if not user_message:
            return Response(
                {"error": "Message is required."}, status=status.HTTP_400_BAD_REQUEST
            )

        # Placeholder for a real LLM backend
        response_text = f"You said: {user_message}"
        return Response({"response": response_text}, status=status.HTTP_200_OK)
