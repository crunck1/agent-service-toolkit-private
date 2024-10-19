import os
import shutil
import json
import csv
import re
import logging
import sqlite3
from datetime import datetime
from typing import Iterable
from functools import partial
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_community.tools import BraveSearch
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.managed import IsLastStep
from langgraph.prebuilt import ToolNode
from agent.llama_guard import LlamaGuard, LlamaGuardOutput, SafetyAssessment

from langchain.schema import Document
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.document_loaders import SpiderLoader
from langchain_core.tools import StructuredTool
from langchain.tools.retriever import create_retriever_tool

from .AgentFileHandler import AgentFileHandler
from .EventExtractor import EventExtractor
from .FAISSManager import FAISSManager
from .GoogleSearchTool import GoogleSearchTool
from .AgentConfigManager import AgentConfigManager


# Configurazione logging
logging.basicConfig(filename='errori.log', level=logging.INFO)

# Modelli predefiniti
models = {
    "gpt-4o-mini": ChatOpenAI(model="gpt-4o-mini", temperature=0.5, streaming=True),
}


class AgentState(MessagesState):
    safety: LlamaGuardOutput
    is_last_step: IsLastStep


# Input per la ricerca
class SearchInput(BaseModel):
    query: str
    num_results: int = 5


class AgentManager:
    def __init__(self, id, model_name="gpt-4o-mini", streaming=True, temperature=0.5):
        self.id = id
        self.model = ChatOpenAI(
            model=model_name, streaming=streaming, temperature=temperature)
        self.tools = []
        self.retriever = None
        self.database = None
        self.table_name = None
        self.current_date = datetime.now().strftime("%B %d, %Y")
        self.instructions = None
        self.db_uri = f"postgresql://claudio:settanta9-a@postgres:5432/agentic"
        # self.checkpoint_manager = AgentCheckpointManager(db_uri=self.db_uri)
        self.site = None
        self.name = None
        self.persist_directory = None
        self.faiss_index = None
        self.embeddings_path = None
        self.original_persist_directory = None
        self.hasSqlToolkit = False
        self.file_embeddings_path = None

    # Aggiunge un nuovo modello
    def add_model(self, model_name, temperature=0.5, streaming=True):
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
        self.name = name

    # Cancella il precedente vectorstore
    def recreate_vectorstore(self):
        if os.path.exists(self.embeddings_path):
            logging.info(f"Cancello il vectorstore in {self.embeddings_path}")
            shutil.rmtree(self.embeddings_path)

    # Trova il file eventi.csv
    def find_event_csv(self):
        agent_config_manager = AgentConfigManager(db_uri=self.db_uri)
        files = agent_config_manager.load_agent_files(self.id)
        for file in files:
            if file['name'] == 'eventi.csv':
                logging.info(f"Trovato file eventi.csv: {file['path']}")
                return '/files/' + file['path']
        logging.info("Il file eventi.csv non è stato trovato.")
        return None

    # Definisce il tool di ricerca

    def search_tool(self, input_data: SearchInput):
        google_search_tool = GoogleSearchTool()
        return google_search_tool.google_search(input_data.query, input_data.num_results)

    def add_tool(self, tool):
        self.tools.append(tool)

    # Aggiunge strumenti di ricerca
    def add_search_tools(self):
        logging.info("Aggiungo search tools")
        api_key = os.getenv('BRAVE_API_KEY')
        if api_key:
            brave_search = BraveSearch.from_api_key(
                name="BraveSearch", api_key=api_key, search_kwargs={"count": 20, "search_lang": "it", "summary": True})
            self.add_tool(brave_search)

        google_search_tool = StructuredTool.from_function(
            func=self.search_tool,
            name="google_search",
            description="Esegue una ricerca su Google e restituisce i primi risultati"
        )
        self.add_tool(google_search_tool)

    # Carica documenti da un file JSONL
    def load_docs_from_jsonl(self, file_path) -> Iterable[Document]:
        array = []
        if os.path.exists(file_path):
            with open(file_path, 'r') as jsonl_file:
                for line in jsonl_file:
                    data = json.loads(line)
                    obj = Document(**data)
                    array.append(obj)
        return array

    # Aggiunge un retriever basato su embeddings
    def add_retriever(self, create_site_docs=False, create_files_docs=False):
        logging.info("Aggiungo retriever")

        faissmanager = FAISSManager(site_index_path=self.embeddings_path, file_index_path=self.file_embeddings_path)

        site_docs = []
        files_docs = []
        if create_site_docs and self.site:
            logging.info("ricreo sito")
            site_docs = self._crawl_and_load_docs()
        if create_files_docs:
            agentFileManager = AgentFileHandler(agent_id=self.id, faiss_manager=faissmanager)
            files_docs = agentFileManager.get_file_docs()

        if len(site_docs) > 0:
            faissmanager.process_new_docs(site_docs, "site")
        #if len(files_docs) > 0:
        faissmanager.process_new_docs(files_docs, "file")


        if hasattr(faissmanager, 'retriever'):
            self._add_retriever_tool(manager=faissmanager)

    # Funzione helper per il crawling e il caricamento dei documenti

    def _crawl_and_load_docs(self):
        """Crawla e carica i documenti dal sito."""
        loader = SpiderLoader(
            api_key=os.getenv("SPIDER_TOKEN"),
            url=self.site,
            mode="crawl",
            params={
                "return_format": "text",
                #"stealth": True,
                #"smart_mode": True,
                "readability": True,
                "metadata": True,
                #"headers": {"user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
            }
        )
        docs = loader.load()
        logging.info(f"Numero di documenti scaricati da spider: {len(docs)}")
        logging.info(docs)
        return docs

    # Funzione helper per salvare i documenti in un file JSONL
    def _save_docs_to_jsonl(self, docs):
        """Salva i documenti in un file JSONL."""
        def save_docs(array, file_path):
            with open(file_path, 'w') as jsonl_file:
                for doc in array:
                    jsonl_file.write(doc.json() + '\n')

        save_docs(docs, self.jsonfile_path)
        logging.info(f"Documenti salvati in {self.jsonfile_path}")

    # Funzione helper per caricare i documenti dal file JSONL
    def _load_docs_from_jsonl(self):
        """Carica i documenti da un file JSONL."""
        logging.info(f"Caricamento dei documenti da {self.jsonfile_path}")
        docs = self.load_docs_from_jsonl(self.jsonfile_path)
        logging.info(f"Numero di documenti caricati: {len(docs)}")
        return docs

    # Funzione helper per aggiungere il retriever come strumento

    def _add_retriever_tool(self, manager):
        logging.info("Aggiungo retriever tool")
        retriever_tool = create_retriever_tool(
            manager.retriever,
            "document_search",
            f"Input of this tool is any query needing an information about {self.name} "
            )
        self.add_tool(retriever_tool)

    def add_sql_tookit(self):
        """Configura una connessione al database SQLite e aggiunge gli strumenti relativi."""
        logging.info(f"Configuro database")
        # Usa SQLite con un file locale
        db_uri = f"sqlite:///{self.original_persist_directory}.db"
        logging.info(f"db_uri = {db_uri}")
        database = SQLDatabase.from_uri(db_uri)
        toolkit = SQLDatabaseToolkit(db=database, llm=self.model)
        self.tools += toolkit.get_tools()
        logging.info(toolkit.get_tools())
        self.hasSqlToolkit = True

    def load_events_from_csv(self, csv_file):
        # Connessione a SQLite (file locale)
        conn = sqlite3.connect(f'{self.original_persist_directory}.db')
        cursor = conn.cursor()

        # Nome della tabella
        table_name = f"spettacoli"

        logging.info(f"creo tabella {table_name}")

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
                logging.info("inserisco riga ")
                logging.info(row)
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
        logging.info("istruzioni:")
        logging.info(instructions)
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
        current_date_italiana = current_date.replace(
            mese_inglese, mese_italiano)
        self.instructions = instructions + f""" La data odierna è {
            current_date_italiana} Considera sempre questa data quando ti viene richiesto un parametro temporale."""
        logging.info(self.hasSqlToolkit)
        if self.hasSqlToolkit:
            logging.info("hasSqlToolkit")
            self.instructions += """ Nel caso ti vengano richieste informazioni sulle date di uno spettacolo:
            1) usa sempre lo strumento 'sql_db_schema', 'sql_db_list_tables' e poi 'sql_db_query' (nell'ordine)
            2) Quando usi lo strumento 'sql_db_query' applica sempre un  limite di 5 risultati a meno che non ti venga esplicitamente richiesto
            3) puoi fare al massimo 5 step.
            4) Non cercare mai nel passato a meno che non ti venga esplicitamente richiesto"""
        logging.info("ora instruction =")
        logging.info(self.instructions)

    def add_site(self, site) -> str:
        self.site = site

    def configure_paths(self, persist_directory) -> str:
        self.original_persist_directory = str(persist_directory)
        self.persist_directory = '/data/' + str(persist_directory)
        self.jsonfile_path = self.persist_directory + '/json_file.jsonl'
        self.embeddings_path = os.path.join(
            self.persist_directory, 'site_embeddings')
        self.file_embeddings_path = os.path.join(
            self.persist_directory, 'file_embeddings')

    def _wrap_model(self, model: BaseChatModel):
        # potremmo anche non avere tools
        logging.info("siamo in wrap_model")
        logging.info(self.tools)
        if self.tools and len(self.tools) > 0:
            logging.info("aggiungo tools")
            model = model.bind_tools(self.tools)
        else:
            logging.info("non aggiungo tools")
        logging.info("istruzioni in wrap_model")
        logging.info(self.instructions)
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
            # Aggiungi l'edge tools -> model solo se esistono tools
            agent.add_edge("tools", "model")

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

    def create_agent(self) -> StateGraph:
        """Crea un agente """
        agent = self._create_agent()
        return agent

    async def add_calendar(self):

        docs = self.load_docs_from_jsonl(self.jsonfile_path)
        logging.info("numero docs")
        logging.info(len(docs))
        logging.info(self.jsonfile_path)
        model = ChatOpenAI(model="gpt-4o-mini-2024-07-18",
                           temperature=0, streaming=True)
        event_extractor = EventExtractor(
            model, self.db_uri, self.name + '_eventi', )
        await event_extractor.run(docs)

    def delete(self):
        """
        Cancella i dati e le risorse associate a un agente specifico.
        """
        # Cancella il vectorstore se esiste
        if os.path.exists(self.embeddings_path):
            logging.info(f"Cancello il vectorstore in {self.embeddings_path}")
            shutil.rmtree(self.embeddings_path)

        # Cancella i documenti JSONL
        if os.path.exists(self.jsonfile_path):
            logging.info(f"Cancello il file JSONL in {self.jsonfile_path}")
            os.remove(self.jsonfile_path)

        # Cancella i file nella directory persistente
        if os.path.exists(self.persist_directory):
            logging.info(f"Cancello la directory di persistenza {
                         self.persist_directory}")
            shutil.rmtree(self.persist_directory)

        # Cancella la tabella dal database
        conn = sqlite3.connect(f'{self.original_persist_directory}.db')
        cursor = conn.cursor()
        table_name = f"spettacoli"
        logging.info(f"Cancello la tabella {table_name} dal database")
        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
        conn.commit()
        conn.close()

        logging.info(
            f"Agente {self.id} e tutte le risorse collegate sono state cancellate.")
