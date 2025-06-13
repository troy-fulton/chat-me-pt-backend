from django.urls import path

from .views import ChatAPIView, ConversationListView, WelcomeView

urlpatterns = [
    path("chat/", ChatAPIView.as_view(), name="chat"),
    path("conversations/", ConversationListView.as_view(), name="conversation_list"),
    path("welcome/", WelcomeView.as_view(), name="welcome"),
]
