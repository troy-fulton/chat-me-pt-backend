# Generated by Django 5.2.2 on 2025-06-13 06:40

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("api", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="visitor",
            name="name",
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
    ]
