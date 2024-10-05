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

import logging
logging.basicConfig(filename='errori.log', level=logging.ERROR)


models = {
    "gpt-4o-mini": ChatOpenAI(model="gpt-4o-mini", temperature=0.5, streaming=True),
}


class AgentState(MessagesState):
    safety: LlamaGuardOutput
    is_last_step: IsLastStep


class AgentManager:
    def __init__(self, model_name="gpt-4o-mini", streaming=True, temperature=0.5):
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

    def add_tool(self, tool):
        """Aggiunge uno strumento all'agente."""
        self.tools.append(tool)

    def add_search_tools(self, use_brave=True, use_duckduckgo=True):
        """Aggiunge strumenti di ricerca (Brave o DuckDuckGo) all'agente."""
        print("aggiungo search tools")

        if use_brave:
            api_key = os.getenv('BRAVE_API_KEY')
            if api_key:
                brave_search = BraveSearch(api_key=api_key)
                self.add_tool(brave_search)

        if use_duckduckgo:
            duck_search = DuckDuckGoSearchResults(
                name="DuckDuckGoSearch", region="it-it", max_results=20)
            self.add_tool(duck_search)

    def load_docs_from_jsonl(self, file_path) -> Iterable[Document]:
        array = []
        with open(file_path, 'r') as jsonl_file:
            for line in jsonl_file:
                data = json.loads(line)
                obj = Document(**data)
                array.append(obj)
        return array

    def add_retriever(self, create_docs=False):
        """Aggiunge un sistema di recupero basato su embeddings."""
        print("Aggiungo retriever")

        embeddings_path = os.path.join(self.persist_directory, 'embeddings')

        # Se esiste la directory degli embeddings, carica il vectorstore
        if os.path.exists(embeddings_path):
            self._load_vectorstore(embeddings_path)
        elif create_docs:
            self._create_and_store_docs()

        # Se è stato creato o caricato il vectorstore, aggiungi il retriever
        if hasattr(self, 'vectorstore'):
            self._add_retriever_tool()

    # Funzione helper per caricare il vectorstore esistente
    def _load_vectorstore(self, embeddings_path):
        """Carica il vectorstore esistente."""
        self.vectorstore = Chroma(
            persist_directory=embeddings_path,
            embedding_function=OpenAIEmbeddings()
        )

    # Funzione helper per creare i documenti e salvarli in jsonl
    def _create_and_store_docs(self):
        """Crea documenti e salva embeddings."""
        # Crea la directory di persistenza se non esiste
        if not os.path.exists(self.persist_directory):
            os.makedirs(self.persist_directory, exist_ok=True)

        # Crea i documenti se il file JSON non esiste
        if not os.path.exists(self.jsonfile_path):
            docs = self._crawl_and_load_docs()
            self._save_docs_to_jsonl(docs)

        # Carica i documenti e crea gli embeddings
        docs = self._load_docs_from_jsonl()
        self._create_embeddings_from_docs(docs)

    # Funzione helper per il crawling e il caricamento dei documenti
    def _crawl_and_load_docs(self):
        """Crawla e carica i documenti dal sito."""
        loader = SpiderLoader(
            api_key=os.getenv("SPIDER_TOKEN"),
            url=self.site,
            mode="crawl",
            params={"return_format": "markdown", "readability": True,
                    "metadata": True, "request": "smart_mode"}
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
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200, chunk_overlap=20
        )

        docs = chromautils.filter_complex_metadata(docs)
        splits = text_splitter.split_documents(docs)

        embedding_model = OpenAIEmbeddings()
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embedding_model,
            persist_directory=os.path.join(
                self.persist_directory, 'embeddings')
        )

    # Funzione helper per aggiungere il retriever come strumento
    def _add_retriever_tool(self):
        """Aggiunge il retriever come strumento."""
        retriever = self.vectorstore.as_retriever()
        self.retriever = retriever
        retriever_tool = create_retriever_tool(
            retriever, "DocumentSearch", "Usa questo strumento per cercare documenti pertinenti."
        )
        self.add_tool(retriever_tool)

    def configure_database(self, table_name):
        """Configura una connessione al database SQL e aggiunge gli strumenti relativi."""
        print(f"configuro database con tabella {table_name}")
        self.table_name = table_name
        self.db_uri = f"postgresql://claudio:settanta9-a@postgres:5432/agentic"
        self.database = SQLDatabase.from_uri(self.db_uri)
        toolkit = SQLDatabaseToolkit(db=self.database, llm=self.model)
        self.tools += toolkit.get_tools()

    def import_to_postgres(self, event_docs, table_name):
        """Importa i documenti nel database PostgreSQL."""
        print("importo dati su postgres")
        if not self.db_params:
            raise ValueError("Database PostgreSQL non configurato.")

        # Qui puoi chiamare una funzione di importazione eventi al database
        self.import_events_to_postgres(self, event_docs)

    def is_directory_exists(self, directory):
        # Controlla se la directory esiste
        return os.path.exists(directory)

    def add_instructions(self, instructions) -> str:
        self.instructions = instructions + f""" La data odierna è {
            self.current_date}. Considera sempre questa data quando ti viene richiesto un parametro temporale."""
        # print("ora instruction =")
        # print(self.instructions)

    def add_site(self, site) -> str:
        self.site = site

    def add_persist_directory(self, persist_directory) -> str:
        self.persist_directory = '/data/' + persist_directory
        self.jsonfile_path = self.persist_directory + '/json_file.jsonl'

    def _wrap_model(self, model: BaseChatModel):
        model = model.bind_tools(self.tools)
        # print("istruzioni in wrap_model")
        # print(self.instructions)
        preprocessor = RunnableLambda(
            lambda state: [SystemMessage(
                content=self.instructions)] + state["messages"],
            name="StateModifier",
        )
        return preprocessor | model

    def _create_agent(self) -> StateGraph:
        """Crea o riprende un agente."""
        # Verifica se esiste un checkpoint per questo agente
        """
        saved_state = self.checkpoint_manager.load_checkpoint(agent_id)
        
        if saved_state:
            print(f"Ripristinando lo stato salvato per l'agente {agent_id}")
            agent_state = saved_state
        else:
            print(f"Creando un nuovo agente per {agent_id}")
            agent_state = AgentState()

        logging.info(agent_state)
        print("agent_state")
        print(agent_state) 
        """

        # Definisci l'agente con lo stato corrente o salvato
        agent = StateGraph(AgentState)
        agent.add_node("model", self._acall_model)
        agent.add_node("tools", ToolNode(self.tools))
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
        agent.add_edge("tools", "model")

        def pending_tool_calls(state: AgentState):
            last_message = state["messages"][-1]
            if last_message.tool_calls:
                return "tools"
            else:
                return "done"

        agent.add_conditional_edges("model", pending_tool_calls, {
                                    "tools": "tools", "done": END})

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
