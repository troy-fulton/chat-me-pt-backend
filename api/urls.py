from django.http import HttpRequest, HttpResponse
from django.urls import path

from .views import (
    ChatAPIView,
    ConversationListView,
    DocumentQueryAPIView,
    VisitorUsageAPIView,
    WelcomeView,
)


def health_check(request: HttpRequest) -> HttpResponse:
    return HttpResponse("OK")


urlpatterns = [
    path("chat/", ChatAPIView.as_view(), name="chat"),
    path("conversations/", ConversationListView.as_view(), name="conversation_list"),
    path("welcome/", WelcomeView.as_view(), name="welcome"),
    path("visitor_usage/", VisitorUsageAPIView.as_view(), name="visitor_usage"),
    path("health/", health_check, name="health_check"),
    path("document_query/", DocumentQueryAPIView.as_view(), name="document_query"),
]
