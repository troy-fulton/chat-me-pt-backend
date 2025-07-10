import os

from django.core.management.base import BaseCommand

from api.document_indexer import DirectoryRAGIndexer


class Command(BaseCommand):
    help = "Rebuilds the vector index"

    def handle(self, *args: object, **options: object) -> None:
        DOC_DIRECTORY = os.environ["DOCUMENTS_DIRECTORY"]
        DOC_INDEX_PATH = os.environ["DOCUMENT_INDEX_PATH"]
        indexer = DirectoryRAGIndexer(
            doc_directory=DOC_DIRECTORY, doc_index_path=DOC_INDEX_PATH
        )

        indexer.build_whoosh_index()
        msg = f"Whoosh index built from {DOC_DIRECTORY} saved to {DOC_INDEX_PATH}"
        self.stdout.write(self.style.SUCCESS(msg))
