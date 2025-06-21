import json
from pathlib import Path
from typing import TypedDict

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredPowerPointLoader,
)
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings


class DocumentMetadata(TypedDict):
    file: str
    description: str


class DirectoryRAGIndexer:
    def __init__(
        self,
        doc_directory: str,
        embeddings: Embeddings | None = None,
        doc_index_path: str | None = None,
    ) -> None:
        self.doc_index_path = doc_index_path
        self.directory = Path(doc_directory)
        self.embeddings: Embeddings = embeddings or HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en"
        )
        self.meta_path = self.directory / "meta.json"
        if not self.directory.exists() or not self.directory.is_dir():
            raise ValueError(
                f"Directory {doc_directory} does not exist or is not a directory."
            )
        if not self.meta_path.exists():
            raise ValueError(
                f"Meta file {self.meta_path} does not exist in the directory."
            )
        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.file_metadata: list[DocumentMetadata] = json.load(f)

    def _load_documents(self) -> list[Document]:
        docs = []

        remaining_file_names = set()
        for file in self.directory.glob("**/*"):
            if file.suffix.lower() == ".pdf":
                docs.extend(PyPDFLoader(str(file)).load())
            elif file.suffix.lower() in [".txt", ".md"]:
                docs.extend(TextLoader(str(file)).load())
            elif file.suffix.lower() == ".pptx":
                docs.extend(UnstructuredPowerPointLoader(str(file)).load())
            elif file.name == "meta.json":
                # Skip the meta.json file itself
                continue
            else:
                remaining_file_names.add(file.name)

        for entry in self.file_metadata:
            remaining_file_names.discard(entry["file"])
            docs.append(
                Document(
                    page_content=entry["description"],
                    metadata={"filepath": self.directory / entry["file"]},
                )
            )

        if remaining_file_names:
            print(
                f"Warning: The following files were not indexed from {self.directory}:"
            )
            for file_name in remaining_file_names:
                print(f" - {file_name}")

        return docs

    def build_vectorstore(self) -> FAISS:
        docs = self._load_documents()
        if len(docs) == 0:
            raise ValueError("No documents found to index.")
        return FAISS.from_documents(docs, self.embeddings)

    def get_vectorstore(self) -> FAISS:
        if not self.doc_index_path:
            raise ValueError("Document index path is not set.")
        if not Path(self.doc_index_path).exists():
            raise ValueError(f"Index path {self.doc_index_path} does not exist.")
        return FAISS.load_local(self.doc_index_path, self.embeddings)

    def save_vectorstore(self, vectorstore: FAISS) -> None:
        if not self.doc_index_path:
            raise ValueError("Document index path is not set.")
        vectorstore.save_local(self.doc_index_path)
        print(f"Vector store saved to {self.doc_index_path}.")
