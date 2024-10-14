import os
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchResults, OpenWeatherMapQueryRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.managed import IsLastStep
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.tools import BraveSearch
from typing import List, Dict
from agent.llama_guard import LlamaGuard, LlamaGuardOutput, SafetyAssessment
from dataclasses import dataclass
from bs4 import BeautifulSoup
from langchain.vectorstores import utils as chromautils
from langchain.schema import Document
import json
from typing import Iterable
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool
import re
from langchain.schema import Document
import psycopg2
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_core.tools import tool
from langchain import hub
from functools import partial
import asyncio
from langchain_community.document_loaders import SpiderLoader
from .AgentCheckpointManager import AgentCheckpointManager
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import utils as chromautils
from langchain_text_splitters.markdown import MarkdownTextSplitter
from .EventExtractor import EventExtractor
from langchain.vectorstores import FAISS
import faiss 
from langchain_community.document_loaders import (
PyPDFLoader, UnstructuredHTMLLoader, UnstructuredWordDocumentLoader, CSVLoader, TextLoader
)

import sqlite3
import csv


import os
import shutil
from .AgentConfigManager import AgentConfigManager
from .GoogleSearchTool import GoogleSearchTool


from pydantic import BaseModel
from langchain_core.tools import StructuredTool
from langchain_ollama import ChatOllama





import logging
logging.basicConfig(filename='errori.log', level=logging.ERROR)


""" models = {
    "gpt-4o-mini": ChatOpenAI(model="gpt-4o-mini", temperature=0.5, streaming=True),
} """

ollama_model = ChatOllama(model="llama3.1:8b")
models = {
    "gpt-4o-mini": ollama_model,
}
class AgentState(MessagesState):
    safety: LlamaGuardOutput
    is_last_step: IsLastStep

    # Definiamo uno schema per i parametri della ricerca
class SearchInput(BaseModel):
    query: str
    num_results: int = 5

class AgentManager:
    def __init__(self, id, model_name="gpt-4o-mini", streaming=True, temperature=0.5):
        # Impostare il modello predefinito
        self.model = ChatOpenAI(
            model=model_name, streaming=streaming, temperature=temperature)
        self.tools = []
        self.retriever = None
        self.database = None
        self.table_name = None
        self.current_date = datetime.now().strftime("%B %d, %Y")
        self.instructions = None
        self.db_uri = f"postgresql://claudio:settanta9-a@postgres:5432/agentic"
        self.checkpoint_manager = AgentCheckpointManager(db_uri=self.db_uri)
        self.agent = None
        self.site = None
        self.name = None
        self.persist_directory = None
        self.faiss_index = None
        self.embeddings_path = None
        self.id = id
        self.original_persist_directory = None
        self.hasSqlToolkit = False

    def add_model(self, model_name, temperature=0.5, streaming=True):
        """Aggiunge un nuovo modello all'agente."""
        if model_name == "gpt-4o-mini":
            self.model = ChatOpenAI(
                model="gpt-4o-mini", temperature=temperature, streaming=streaming)
        elif model_name == "llama-3.1-70b" and os.getenv("GROQ_API_KEY"):
            self.model = ChatGroq(
                model="llama-3.1-70b-versatile", temperature=temperature)
        else:
            raise ValueError(
                f"Modello {model_name} non supportato o chiave API non disponibile.")

    def add_name(self, name):
        """Aggiunge un nome."""
        self.name = name


    def recreateVectorstore(self):
        ## Cancelliamo il precedente vectorstore ##
        if os.path.exists(self.embeddings_path):
            print(f"Cancello il vectorstore in {self.embeddings_path}")
            shutil.rmtree(self.embeddings_path)
        
    def find_event_csv(self):
        """Cerca il file eventi.csv nella lista dei file."""
        files = self.load_agent_files()
        """Cerca il file eventi.csv nella lista dei file."""
        for file in files:
            if file['name'] == 'eventi.csv':
                print(f"Trovato file eventi.csv: {file['path']}")
                return '/files/' + file['path']
        print("Il file eventi.csv non è stato trovato.")
        return None

    def load_agent_files(self):
        """Carica tutti i file di un agente e li elabora con il loader appropriato."""
        
        # Carica i file dell'agente dal database
        db_uri = f"postgresql://claudio:settanta9-a@postgres:5432/agentic"
        agent_config_manager = AgentConfigManager(db_uri=db_uri)


        files = agent_config_manager.load_agent_files(self.id)
        return files

    def load_docs_from_agent_files(self):
        """Carica tutti i file di un agente e li elabora con il loader appropriato."""
        
        # Carica i file dell'agente dal database
        db_uri = f"postgresql://claudio:settanta9-a@postgres:5432/agentic"
        agent_config_manager = AgentConfigManager(db_uri=db_uri)


        files = agent_config_manager.load_agent_files(self.id)
        documents = []
        
        for file in files:
            file_path = '/files/' + file['path']
            file_extension = os.path.splitext(file_path)[1].lower()
            print(f"carico il file {file_path}")

            # Selezione del loader in base all'estensione del file
            if file_extension == ".pdf":
                loader = PyPDFLoader(file_path)
            elif file_extension == ".html":
                loader = UnstructuredHTMLLoader(file_path)
            elif file_extension in [".docx", ".doc"]:
                loader = UnstructuredWordDocumentLoader(file_path)
            elif file_extension == ".csv":
                if file['path'] == 'eventi.csv':  # Verifica se il file è "eventi.csv"
                    continue
                else:
                    loader = CSVLoader(file_path)
            elif file_extension == ".txt":
                loader = TextLoader(file_path)
            else:
                continue

            # Carica i documenti e li aggiunge all'elenco
            documents.extend(loader.load())
        
        return documents
        

    # Definiamo il tool per Langchain
    def search_tool(input_data: SearchInput):
        google_search_tool = GoogleSearchTool()
        return google_search_tool.google_search(input_data.query, input_data.num_results)    
        
    def add_tool(self, tool):
        """Aggiunge uno strumento all'agente."""
        self.tools.append(tool)

    def add_search_tools(self):
        """Aggiunge strumenti di ricerca (Brave o DuckDuckGo) all'agente."""
        print("aggiungo search tools")

        api_key = os.getenv('BRAVE_API_KEY')
        if api_key:
            brave_search = BraveSearch.from_api_key(name="BraveSearch", api_key=api_key, search_kwargs={"count": 20, "search_lang":"it","summary":True})
            self.add_tool(brave_search)


        duck_search = DuckDuckGoSearchResults(
            name="DuckDuckGoSearch", region="it-it", max_results=20)
        self.add_tool(duck_search)

        # Definiamo il tool per Langchain
        def search_tool(query):
            google_search_tool = GoogleSearchTool()
            return google_search_tool.google_search(query)

        google_search_tool = StructuredTool.from_function(
                                func=search_tool,
                                name="google_search",
                                description="Esegue una ricerca su Google e restituisce i primi risultati"
                            )
        self.add_tool(google_search_tool)




    def load_docs_from_jsonl(self, file_path) -> Iterable[Document]:
        array = []
        if  os.path.exists(file_path) :
            with open(file_path, 'r') as jsonl_file:
                for line in jsonl_file:
                    data = json.loads(line)
                    obj = Document(**data)
                    array.append(obj)
        return array

    def add_retriever(self, create_site_docs=False, create_files_docs=False):
        """Aggiunge un sistema di recupero basato su embeddings."""
        print("Aggiungo retriever")

        if create_site_docs:
            self._create_and_store_site_docs()
        if create_files_docs:
            self._add_file_docs_to_vectorstore()
        # Se esiste la directory degli embeddings, carica il vectorstore
        if os.path.exists(self.embeddings_path):
            self._load_vectorstore()

        if hasattr(self, 'vectorstore'):
            self._add_retriever_tool()

    # Funzione helper per caricare il vectorstore esistente
    def _load_vectorstore(self):
        """Carica il vectorstore esistente."""

        # Carica l'indice FAISS salvato

        self.vectorstore = FAISS.load_local(self.embeddings_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)


    def _add_file_docs_to_vectorstore(self):
        """Aggiunge documenti dai file esistenti al vectorstore salvato."""

        # Carica il vectorstore esistente dal disco
        if os.path.exists(self.embeddings_path):
            self.vectorstore = FAISS.load_local(self.embeddings_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
        else:
            raise FileNotFoundError("Il vectorstore non esiste. Crealo prima con i documenti del sito.")

        # Carica i documenti dai file
        docs_from_files = self.load_docs_from_agent_files()
        print(f"Numero di documenti da file: {len(docs_from_files)}")

        # Se ci sono nuovi documenti, calcola i loro embeddings e aggiungili al vectorstore
        if docs_from_files:
            embedding_model = OpenAIEmbeddings()
            text_splitter = MarkdownTextSplitter(chunk_size=1500, chunk_overlap=150)
            file_splits = text_splitter.split_documents(docs_from_files)

            # Crea un vectorstore per i documenti dai file e uniscilo a quello esistente
            file_store = FAISS.from_documents(file_splits, embedding_model)
            self.vectorstore.merge_from(file_store)

            # Salva il vectorstore aggiornato su disco
            self.vectorstore.save_local(self.embeddings_path)

            print("Documenti dai file aggiunti al vectorstore e salvati.")


    def _create_and_store_site_docs(self):
        """Crea e salva gli embeddings solo per i documenti del sito."""

        # Crea la directory di persistenza se non esiste
        if not os.path.exists(self.persist_directory):
            os.makedirs(self.persist_directory, exist_ok=True)

        # Se il file JSON non esiste, carica i documenti dal sito e salvali
        if not os.path.exists(self.jsonfile_path):
            docs = self._crawl_and_load_docs()
            self._save_docs_to_jsonl(docs)

        # Carica i documenti salvati dal sito
        if  os.path.exists(self.jsonfile_path) :
            site_docs = self._load_docs_from_jsonl()

        # Crea embeddings solo per i documenti del sito
        embedding_model = OpenAIEmbeddings()
        text_splitter = MarkdownTextSplitter(chunk_size=1500, chunk_overlap=150)
        site_splits = text_splitter.split_documents(site_docs)

        # Crea un nuovo vectorstore per i documenti del sito e salvalo su disco
        self.vectorstore = FAISS.from_documents(site_splits, embedding_model)
        self.vectorstore.save_local(self.embeddings_path)

        print("Embeddings del sito creati e salvati.")


    # Funzione helper per il crawling e il caricamento dei documenti
    def _crawl_and_load_docs(self):
        """Crawla e carica i documenti dal sito."""
        loader = SpiderLoader(
            api_key=os.getenv("SPIDER_TOKEN"),
            url=self.site,
            mode="crawl"
        )
        docs = loader.load()
        print(f"Numero di documenti scaricati: {len(docs)}")
        return docs

    # Funzione helper per salvare i documenti in un file JSONL
    def _save_docs_to_jsonl(self, docs):
        """Salva i documenti in un file JSONL."""
        def save_docs(array, file_path):
            with open(file_path, 'w') as jsonl_file:
                for doc in array:
                    jsonl_file.write(doc.json() + '\n')

        save_docs(docs, self.jsonfile_path)
        print(f"Documenti salvati in {self.jsonfile_path}")

    # Funzione helper per caricare i documenti dal file JSONL
    def _load_docs_from_jsonl(self):
        """Carica i documenti da un file JSONL."""
        print(f"Caricamento dei documenti da {self.jsonfile_path}")
        docs = self.load_docs_from_jsonl(self.jsonfile_path)
        print(f"Numero di documenti caricati: {len(docs)}")
        return docs

    # Funzione helper per creare gli embeddings dai documenti


    def _create_embeddings_from_docs(self, docs):
        """Crea embeddings dai documenti caricati."""

        # Usa MarkdownTextSplitter al posto di RecursiveCharacterTextSplitter
        text_splitter = MarkdownTextSplitter(
            chunk_size=1500, chunk_overlap=150
        )

        docs = chromautils.filter_complex_metadata(docs)
        splits = text_splitter.split_documents(docs)

        embedding_model = OpenAIEmbeddings()
        self.vectorstore = FAISS.from_documents(
            documents=splits,
            embedding=embedding_model
        )
        # Salva l'indice FAISS manualmente (persistenza)
        self.vectorstore.save_local(self.embeddings_path)

    # Funzione helper per aggiungere il retriever come strumento
    def _add_retriever_tool(self):
        """Aggiunge il retriever come strumento."""
        retriever = self.vectorstore.as_retriever()
        self.retriever = retriever
        retriever_tool = create_retriever_tool(
            retriever, "DocumentSearch", "Usa questo strumento per cercare documenti pertinenti."
        )
        self.add_tool(retriever_tool)

    def add_sql_tookit(self):
        """Configura una connessione al database SQLite e aggiunge gli strumenti relativi."""
        print(f"Configuro database")
        # Usa SQLite con un file locale
        db_uri = f"sqlite:///{self.original_persist_directory}.db"
        database = SQLDatabase.from_uri(db_uri)
        toolkit = SQLDatabaseToolkit(db=database, llm=self.model)
        self.tools += toolkit.get_tools()
        self.hasSqlToolkit = True

    def import_events_to_sqlite(self, csv_path):
        if os.path.exists(csv_path):
            self.load_events_from_csv(csv_path)

    def load_events_from_csv(self, csv_file):
        # Connessione a SQLite (file locale)
        conn = sqlite3.connect(f'{self.original_persist_directory}.db')
        cursor = conn.cursor()

        
        
        # Nome della tabella
        table_name = f"spettacoli"

        print(f"creo tabella {table_name}")
        
        # Creazione della tabella se non esiste già
        cursor.execute(f'''
            drop table if exists {table_name};
        ''')
        cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {table_name} (
                event_date DATE,
                event_time TEXT,
                event_title TEXT,
                event_author TEXT,
                content TEXT
            );
        ''')

        """ cursor.execute(f'''
            CREATE UNIQUE INDEX IF NOT EXISTS unique_event ON {table_name} (event_date, event_title);
        ''') """

        
        # Caricamento dati dal file CSV
        with open(csv_file, newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Salta l'intestazione
            for row in reader:
                print("inserisco riga ")
                print(row)
                cursor.execute(f'''
                    INSERT INTO {table_name} (event_date, event_time, event_title, event_author, content)
                    VALUES (?, ?, ?, ?, ?)
                ''', row)
        
        # Salva i cambiamenti e chiudi la connessione
        conn.commit()
        conn.close()


    def is_directory_exists(self, directory):
        # Controlla se la directory esiste
        return os.path.exists(directory)

    def add_instructions(self, instructions) -> str:
        print("istruzioni:")
        print(instructions)
        # Imposta la localizzazione in italiano
        from datetime import datetime

        # Dizionario per tradurre i mesi in italiano
        mesi_italiani = {
            "January": "gennaio", "February": "febbraio", "March": "marzo", "April": "aprile",
            "May": "maggio", "June": "giugno", "July": "luglio", "August": "agosto",
            "September": "settembre", "October": "ottobre", "November": "novembre", "December": "dicembre"
        }

        # Ottieni la data odierna in inglese e traducila
        current_date = datetime.now().strftime("%d %B %Y")
        mese_inglese = datetime.now().strftime("%B")
        mese_italiano = mesi_italiani[mese_inglese]

        # Sostituisci il mese inglese con quello italiano
        current_date_italiana = current_date.replace(mese_inglese, mese_italiano)
        self.instructions = instructions + f""" La data odierna è {current_date_italiana} Considera sempre questa data quando ti viene richiesto un parametro temporale."""
        print(self.hasSqlToolkit)
        if self.hasSqlToolkit:
            print("hasSqlToolkit")
            self.instructions += """ Nel caso ti vengano richieste informazioni sulle date di uno spettacolo:
            1) usa sempre lo strumento 'sql_db_schema', 'sql_db_list_tables' e poi 'sql_db_query' (nell'ordine)
            2) Quando usi lo strumento 'sql_db_query' applica sempre un  limite di 5 risultati a meno che non ti venga esplicitamente richiesto
            3) puoi fare al massimo 5 step.
            4) Non cercare mai nel passato a meno che non ti venga esplicitamente richiesto"""
        # print("ora instruction =")
        # print(self.instructions)

    def add_site(self, site) -> str:
        self.site = site

    def configure_paths(self, persist_directory) -> str:
        self.original_persist_directory = str(persist_directory) 
        self.persist_directory = '/data/' + str(persist_directory)
        self.jsonfile_path = self.persist_directory + '/json_file.jsonl'
        self.embeddings_path = os.path.join(self.persist_directory, 'embeddings')
        

    def _wrap_model(self, model: BaseChatModel):
        # potremmo anche non avere tools
        print("siamo in wrap_model")
        print(self.tools)
        if self.tools and len(self.tools) > 0 :
            print("aggiungo tools")
            model = model.bind_tools(self.tools)
        else:
            print("non aggiungo tools")
        # print("istruzioni in wrap_model")
        # print(self.instructions)
        preprocessor = RunnableLambda(
            lambda state: [SystemMessage(
                content=self.instructions)] + state["messages"],
            name="StateModifier",
        )
        return preprocessor | model

    def _create_agent(self) -> StateGraph:
        agent = StateGraph(AgentState)
        agent.add_node("model", self._acall_model)
        
        # Aggiungi il nodo tools solo se self.tools non è vuoto
        if self.tools and len(self.tools) > 0:
            agent.add_node("tools", ToolNode(self.tools))
            agent.add_edge("tools", "model")  # Aggiungi l'edge tools -> model solo se esistono tools
        
        agent.add_node("guard_input", self._llama_guard_input)
        agent.add_node("block_unsafe_content", self._block_unsafe_content)
        agent.set_entry_point("guard_input")

        def check_safety(state: AgentState):
            safety: LlamaGuardOutput = state["safety"]
            match safety.safety_assessment:
                case SafetyAssessment.UNSAFE:
                    return "unsafe"
                case _:
                    return "safe"

        agent.add_conditional_edges(
            "guard_input", check_safety, {
                "unsafe": "block_unsafe_content", "safe": "model"}
        )

        agent.add_edge("block_unsafe_content", END)

        def pending_tool_calls(state: AgentState):
            last_message = state["messages"][-1]
            if last_message.tool_calls:
                return "tools"
            else:
                return "done"

        # Condizionalmente aggiungi edges solo se tools è presente
        if self.tools and len(self.tools) > 0:
            agent.add_conditional_edges("model", pending_tool_calls, {
                                        "tools": "tools", "done": END})
        else:
            agent.add_edge("model", END)

        return agent.compile(checkpointer=MemorySaver())


    async def _acall_model(self, state: AgentState, config: RunnableConfig):
        m = models[config["configurable"].get("model", "gpt-4o-mini")]
        model_runnable = self._wrap_model(m)
        response = await model_runnable.ainvoke(state, config)

        llama_guard = LlamaGuard()
        safety_output = await llama_guard.ainvoke("Agent", state["messages"] + [response])
        if safety_output.safety_assessment == SafetyAssessment.UNSAFE:
            return {"messages": [self._format_safety_message(safety_output)], "safety": safety_output}

        if state["is_last_step"] and response.tool_calls:
            return {
                "messages": [
                    AIMessage(
                        id=response.id,
                        content="Sorry, need more steps to process this request.",
                    )
                ]
            }
        return {"messages": [response]}

    async def _llama_guard_input(self, state: AgentState, config: RunnableConfig):
        llama_guard = LlamaGuard()
        safety_output = await llama_guard.ainvoke("User", state["messages"])
        return {"safety": safety_output}

    async def _block_unsafe_content(self, state: AgentState, config: RunnableConfig):
        safety: LlamaGuardOutput = state["safety"]
        return {"messages": [self._format_safety_message(safety)]}

    def _format_safety_message(self, safety: LlamaGuardOutput) -> AIMessage:
        content = (
            f"This conversation was flagged for unsafe content: {
                ', '.join(safety.unsafe_categories)}"
        )
        return AIMessage(content=content)

    def save_agent_checkpoint(self, agent_id: str):
        """Salva il checkpoint per l'agente corrente."""
        """ if self.agent:
            self.checkpoint_manager.save_checkpoint(agent_id, self.agent.state) """

    def delete_agent_checkpoint(self, agent_id: str):
        """Elimina il checkpoint per l'agente."""
        # self.checkpoint_manager.delete_checkpoint(agent_id)

    def create_agent(self) -> StateGraph:
        """Crea un agente e salva il checkpoint iniziale."""
        agent = self._create_agent()
        # Salva il checkpoint iniziale
        """ self.checkpoint_manager.save_checkpoint(agent_id, {
            "agent_state": agent.state,
            "tools": self.tools
        }) """
        return agent

    async def add_calendar(self):

        docs = self.load_docs_from_jsonl(self.jsonfile_path)
        print("numero docs")
        print(len(docs))
        print(self.jsonfile_path)
        model = ChatOpenAI(model="gpt-4o-mini-2024-07-18",
                           temperature=0, streaming=True)
        event_extractor = EventExtractor(
            model, self.db_uri, self.name + '_eventi', )
        await event_extractor.run(docs)
