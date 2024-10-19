import faiss
import json
import os
import logging
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from typing import List, Dict, Optional
from langchain.schema import Document
from langchain_text_splitters import MarkdownTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import utils as chromautils
import hashlib
from langchain.schema import Document
from typing import Iterable

logging.basicConfig(filename='agent.log', level=logging.INFO)

class FAISSManager:
    def __init__(self, site_index_path: str, file_index_path: str, embeddings_model=None):
        self.site_index_path = site_index_path
        self.file_index_path = file_index_path
        self.embeddings_model = embeddings_model or OpenAIEmbeddings()
        
        # Inizializza gli indici separati per siti e file
        self.site_vectorstore = self.load_faiss_index(self.site_index_path)
        self.file_vectorstore = self.load_faiss_index(self.file_index_path)

        # Inizializza retriever solo se ci sono vectorstore
        self.retriever = None
        self.update_retriever()

    def generate_unique_id(self, doc: Document) -> str:
        """Genera un ID univoco basato sul contenuto del documento."""
        hash_object = hashlib.sha256(doc.page_content.encode('utf-8'))
        return hash_object.hexdigest()

    def assign_ids_to_docs(self, docs: List[Document]) -> List[Document]:
        """Assegna un ID univoco a ciascun documento."""
        for doc in docs:
            unique_id = self.generate_unique_id(doc)
            doc.id = unique_id
        return docs

    def load_faiss_index(self, index_path: str) -> Optional[FAISS]:
        """Carica un FAISS index locale."""
        if os.path.exists(index_path):
            logging.info(f"Loading FAISS index from {index_path}...")
            return FAISS.load_local(index_path, self.embeddings_model, allow_dangerous_deserialization=True)
        return None

    def process_new_docs(self, new_docs: List[Document], index_type: str):
        """Sincronizza i nuovi documenti con il vectorstore FAISS specifico per sito o file."""
        logging.info(f"Generating embeddings for {len(new_docs)} new documents ({index_type})...")
        
        # Filtro e split dei documenti
        new_docs = chromautils.filter_complex_metadata(new_docs)
        #text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
        text_splitter = MarkdownTextSplitter(chunk_size=2000, chunk_overlap=1000)
        splits = text_splitter.split_documents(new_docs)
        docs_with_ids = self.assign_ids_to_docs(splits)
        
        # Scegli quale indice aggiornare
        if index_type == 'site':
            vectorstore = self.site_vectorstore
            index_path = self.site_index_path
        elif index_type == 'file':
            vectorstore = self.file_vectorstore
            index_path = self.file_index_path
        else:
            raise ValueError("Invalid index type. Must be 'site' or 'file'.")

        # Ottieni gli ID dei nuovi documenti e degli esistenti
        new_doc_ids = {doc.id for doc in docs_with_ids}
        existing_ids = set(vectorstore.index_to_docstore_id.values()) if vectorstore else set()

        # Rimuovi i documenti non più presenti
        ids_to_remove = existing_ids - new_doc_ids
        if ids_to_remove and vectorstore:
            vectorstore.delete(ids_to_remove)
            logging.info(f"Removed {len(ids_to_remove)} old documents from {index_type} vectorstore.")

        # Aggiungi i nuovi documenti mancanti
        ids_to_add = new_doc_ids - existing_ids
        new_docs_to_add = [doc for doc in docs_with_ids if doc.id in ids_to_add]
        unique_docs = {doc.id: doc for doc in new_docs_to_add}.values()
        new_docs_to_add = list(unique_docs)

        if new_docs_to_add:
            logging.info(f"Adding {len(new_docs_to_add)} new documents to {index_type} vectorstore.")
            if vectorstore:
                vectorstore.add_documents(new_docs_to_add)
            else:
                vectorstore = FAISS.from_documents(new_docs_to_add, self.embeddings_model)


            # Aggiorna il vectorstore corrente
            if index_type == 'site':
                self.site_vectorstore = vectorstore
            else:
                self.file_vectorstore = vectorstore

            # Aggiorna il retriever ogni volta che il vectorstore cambia
            self.update_retriever()
        else:
            logging.info(f"No new documents to add to {index_type} vectorstore.")

        # Salva il vectorstore aggiornato
        vectorstore.save_local(index_path)             
        logging.info(f"{index_type.capitalize()} vectorstore saved successfully.")


    def merge_vectorstores(self) -> FAISS:
        """Unisce i due vectorstore (siti e file) in un unico FAISS index."""
        if self.site_vectorstore and self.file_vectorstore:
            logging.info("Merging site and file vectorstores.")
            self.site_vectorstore.merge_from(self.file_vectorstore)
            return self.site_vectorstore
        elif self.site_vectorstore:
            return self.site_vectorstore
        elif self.file_vectorstore:
            return self.file_vectorstore
        else:
            raise ValueError("No vectorstores to merge.")

    def update_retriever(self):
        """Aggiorna il retriever con il vectorstore più recente (dopo un merge o update)."""
        merged_store = self.merge_vectorstores() if self.site_vectorstore or self.file_vectorstore else None
        if merged_store:
            self.retriever = merged_store.as_retriever()
            logging.info("Retriever updated with merged vectorstore.")
        else:
            logging.info("No vectorstore to initialize retriever.")
