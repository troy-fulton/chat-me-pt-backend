from typing import Any

from django.db import models


class Visitor(models.Model):
    session_id = models.CharField(max_length=64, unique=True)
    name = models.CharField(max_length=100, null=True, blank=True)
    interests = models.CharField(max_length=500, null=True, blank=True)
    company = models.CharField(max_length=100, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)


class Conversation(models.Model):
    id = models.AutoField(primary_key=True)
    visitor = models.ForeignKey(
        Visitor, on_delete=models.SET_NULL, null=True, blank=True
    )
    title = models.CharField(max_length=200, blank=True)
    started_at = models.DateTimeField(auto_now_add=True)
    ended_at = models.DateTimeField(null=True, blank=True)

    def __str__(self) -> str:
        return self.title or f"Conversation {self.id}"


class ChatMessage(models.Model):
    ROLE_CHOICES = (
        ("visitor", "Visitor"),
        ("assistant", "Assistant"),
        ("system", "System"),
    )
    LLM_USE_CHOICES = (
        ("chat", "Chat"),
        ("document_query", "Document Query"),
    )

    conversation = models.ForeignKey(
        Conversation, related_name="messages", on_delete=models.CASCADE
    )
    content = models.TextField()
    referenced_documents: Any = models.JSONField(default=list, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    role = models.CharField(max_length=10, choices=ROLE_CHOICES)
    token_count = models.IntegerField(default=0)
    llm_use = models.CharField(max_length=20, choices=LLM_USE_CHOICES, default="chat")
