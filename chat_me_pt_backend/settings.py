import os
from pathlib import Path
from typing import Any, Dict

import dj_database_url

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = "django-insecure-please-change-this-key"

DEBUG = True

ALLOWED_HOSTS = ["*"]

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "corsheaders",
    "rest_framework",
    "api",
]

MIDDLEWARE = [
    "corsheaders.middleware.CorsMiddleware",
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "chat_me_pt_backend.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "chat_me_pt_backend.wsgi.application"

IN_RAILWAY = os.getenv("RAILWAY_ENVIRONMENT", None) is not None

DATABASES: Dict[str, Any]

if IN_RAILWAY:
    DATABASES = {
        "default": dj_database_url.config(
            default=os.environ["DATABASE_URL"], conn_max_age=600
        )
    }
else:
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": BASE_DIR / "db.sqlite3",
        }
    }

AUTH_PASSWORD_VALIDATORS: list[str] = []

LANGUAGE_CODE = "en-us"

TIME_ZONE = "UTC"

USE_I18N = True

USE_TZ = True

STATIC_URL = "static/"

if IN_RAILWAY:
    FRONTEND_URL = os.getenv("FRONTEND_URL", "").strip()
    print(f"FRONTEND_URL: {FRONTEND_URL}")
    if FRONTEND_URL:
        CORS_ALLOWED_ORIGINS = [FRONTEND_URL]
    else:
        CORS_ALLOWED_ORIGINS = []
else:
    CORS_ALLOWED_ORIGIN_REGEXES = [
        r"^http://localhost:\d+$",
    ]

CORS_ALLOW_CREDENTIALS = True
SESSION_COOKIE_SAMESITE = "None"
SESSION_COOKIE_SECURE = True
