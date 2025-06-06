import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "chat_me_pt_backend.settings")

application = get_wsgi_application()
