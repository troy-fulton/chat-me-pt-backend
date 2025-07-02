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
    filename: str
    description: str
    semantic: bool


class DirectoryRAGIndexer:
    def __init__(
        self,
        doc_directory: str,
        embeddings: Embeddings | None = None,
        doc_index_path: str | None = None,
    ) -> None:
        print("Initializing DirectoryRAGIndexer...")
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
        print(f"Loading documents from {self.directory}...")
        docs: list[Document] = []

        remaining_file_names = set()
        for document in self.file_metadata:
            filepath = self.directory / document["filename"]

            loaded_docs = list()
            semantic = document.get("semantic", False)
            if not semantic and filepath.suffix.lower() == ".pdf":
                print(f"Loading PDF: {filepath}")
                loaded_docs = PyPDFLoader(str(filepath)).load()
            elif not semantic and filepath.suffix.lower() in [".txt", ".md"]:
                print(f"Loading text file: {filepath}")
                loaded_docs = TextLoader(str(filepath)).load()
            elif not semantic and filepath.suffix.lower() == ".pptx":
                print(f"Loading PowerPoint: {filepath}")
                loaded_docs = UnstructuredPowerPointLoader(str(filepath)).load()
            else:
                remaining_file_names.add(document["filename"])

            for loaded_doc in loaded_docs:
                loaded_doc.metadata["filepath"] = filepath
                loaded_doc.metadata["file"] = document["filename"]
                loaded_doc.metadata["description"] = document["description"]
            docs.extend(loaded_docs)

        print(f"Loaded {len(docs)} documents from {self.directory}.")
        print(f"Indexing remaining {len(remaining_file_names)} files semantically...")
        for entry in self.file_metadata:
            if entry["filename"] not in remaining_file_names:
                continue
            remaining_file_names.discard(entry["filename"])
            docs.append(
                Document(
                    page_content=entry["description"],
                    metadata={
                        "filepath": self.directory / entry["filename"],
                        "file": entry["filename"],
                        "description": entry["description"],
                    },
                )
            )

        if remaining_file_names:
            print(
                f"Warning: The following files were not indexed from {self.directory}:"
            )
            for file_name in remaining_file_names:
                print(f" - {file_name}")

        print(f"Total documents loaded: {len(docs)}")
        return docs

    def build_vectorstore(self) -> FAISS:
        docs = self._load_documents()
        if len(docs) == 0:
            raise ValueError("No documents found to index.")
        print("Building vector store...")
        return FAISS.from_documents(docs, self.embeddings)

    def get_vectorstore(self) -> FAISS:
        if not self.doc_index_path:
            raise ValueError("Document index path is not set.")
        if not Path(self.doc_index_path).exists():
            raise ValueError(f"Index path {self.doc_index_path} does not exist.")
        return FAISS.load_local(
            self.doc_index_path, self.embeddings, allow_dangerous_deserialization=True
        )

    def save_vectorstore(self, vectorstore: FAISS) -> None:
        if not self.doc_index_path:
            raise ValueError("Document index path is not set.")
        print(f"Saving vector store to {self.doc_index_path}...")
        vectorstore.save_local(self.doc_index_path)
        print(f"Vector store saved to {self.doc_index_path}.")
