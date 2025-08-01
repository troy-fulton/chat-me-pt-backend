# Generated by Django 5.2.2 on 2025-07-14 02:22

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("api", "0004_chatmessage_referenced_documents"),
    ]

    operations = [
        migrations.AddField(
            model_name="chatmessage",
            name="llm_use",
            field=models.CharField(
                choices=[("chat", "Chat"), ("document_query", "Document Query")],
                default="chat",
                max_length=20,
            ),
        ),
    ]
