from django.test import TestCase
from django.urls import reverse
from rest_framework.test import APIClient


class ChatAPITest(TestCase):
    def setUp(self) -> None:
        self.client = APIClient()

    def test_chat_post(self) -> None:
        url = reverse("chat")
        resp = self.client.post(url, {"message": "Hello, world!"}, format="json")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("response", resp.content)
