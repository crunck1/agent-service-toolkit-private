import hashlib
import os
import json
import logging

from service import FAISSManager
from langchain_community.document_loaders import (
     PyPDFLoader, UnstructuredHTMLLoader, UnstructuredWordDocumentLoader,
    CSVLoader, TextLoader
)
from langchain_unstructured import UnstructuredLoader



from .AgentConfigManager import AgentConfigManager
logging.basicConfig(filename='agent.log', level=logging.INFO)

class AgentFileHandler:
    def __init__(self, agent_id:str, faiss_manager: FAISSManager
                 #, hash_file_path='file_hashes.json'
                 ):
        self.agent_id = agent_id
        self.faiss_manager = faiss_manager  # Integrazione con FAISSManager
        #self.hash_file_path = hash_file_path
        #self.file_hashes = self.load_hashes()  # Carica il dizionario degli hash

    # Carica documenti dai file dell'agente
    def load_docs_from_agent_files(self):
        files = self.load_agent_files()  # Metodo che carica i file di un agente
        documents = []
        for file in files:
            file_path = '/files/' + file['path']
            file_extension = os.path.splitext(file_path)[1].lower()
            logging.info(f"Carico il file {file_path}")

            # Calcola l'hash del file per verificarne l'unicità
           # file_hash = self.calculate_file_hash(file_path)

            # Verifica se l'hash del file è già presente nel vectorstore
            #if file_hash in self.file_hashes:
            #    logging.info(f"Il file {file_path} è già stato processato, salto l'aggiunta.")
            #    continue  # Salta il file se è già stato processato

            # Usa un loader appropriato per ogni estensione di file
            if file_extension == ".pdf":
                loader = UnstructuredLoader(file_path)
            elif file_extension == ".html":
                loader = UnstructuredHTMLLoader(file_path)
            elif file_extension in [".docx", ".doc"]:
                loader = UnstructuredWordDocumentLoader(file_path)
            elif file_extension == ".csv" and file['path'] != 'eventi.csv':
                loader = CSVLoader(file_path)
            elif file_extension == ".txt":
                loader = TextLoader(file_path)
            else:
                continue

            documents.extend(loader.load())
            #self.file_hashes[file_hash] = file_path  # Aggiungi l'hash alla lista

        #self.save_hashes()  # Salva gli hash dopo aver processato i file
        return documents

    # Aggiunge documenti dai file esistenti al vectorstore
    def get_file_docs(self):
        docs_from_files = self.load_docs_from_agent_files()
        logging.info(f"Numero di documenti da file: {len(docs_from_files)}")
        return docs_from_files


    # Metodo per calcolare l'hash di un file
    """     def calculate_file_hash(self, file_path):
        hash_sha256 = hashlib.sha256()
        with open(file_path, 'rb') as file:
            # Leggi il file in blocchi per evitare problemi con file di grandi dimensioni
            for block in iter(lambda: file.read(4096), b""):
                hash_sha256.update(block)
        return hash_sha256.hexdigest()

    # Salva gli hash dei file su disco in formato JSON
    def save_hashes(self):
        with open(self.hash_file_path, 'w') as hash_file:
            json.dump(self.file_hashes, hash_file)
        logging.info(f"Hash dei file salvati in {self.hash_file_path}.")

    # Carica gli hash dei file da disco, se il file esiste
    def load_hashes(self):
        if os.path.exists(self.hash_file_path):
            with open(self.hash_file_path, 'r') as hash_file:
                logging.info(f"Caricamento degli hash da {self.hash_file_path}.")
                return json.load(hash_file)
        else:
            logging.info(f"Nessun file degli hash trovato, creazione di un nuovo dizionario.")
            return {} """

        # Metodo per caricare i file dell'agente (simulato qui)
    # Carica tutti i file di un agente
    def load_agent_files(self):
        agent_config_manager = AgentConfigManager(db_uri=f"postgresql://claudio:settanta9-a@postgres:5432/agentic")
        return agent_config_manager.load_agent_files(self.agent_id)
