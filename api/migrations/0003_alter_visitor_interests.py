# Generated by Django 5.2.2 on 2025-06-17 02:14

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("api", "0002_visitor_name"),
    ]

    operations = [
        migrations.AlterField(
            model_name="visitor",
            name="interests",
            field=models.CharField(blank=True, max_length=500, null=True),
        ),
    ]
