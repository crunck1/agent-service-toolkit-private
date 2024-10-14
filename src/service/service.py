import asyncio
from contextlib import asynccontextmanager
import json
import os
from typing import AsyncGenerator, Dict, Any, Tuple, List
from uuid import uuid4
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.graph import CompiledGraph
from langsmith import Client as LangsmithClient
from fastapi.middleware.cors import CORSMiddleware
from agent import  qa_assistant_react
from schema import ChatMessage, Feedback, UserInput, StreamInput
from typing import Dict, Any
from .AgentManager import AgentManager
from .AgentStoreManager import AgentStoreManager
from .AgentConfigManager import AgentConfigManager
import logging
from fastapi.responses import JSONResponse
from fastapi import File, UploadFile
from langchain.document_loaders import PyPDFLoader, TextLoader
from typing import Dict, Any, Optional
from pydantic import BaseModel



db_uri = f"postgresql://claudio:settanta9-a@postgres:5432/agentic"
agent_store_manager = AgentStoreManager(db_uri=db_uri)
agent_config_manager = AgentConfigManager(db_uri=db_uri)
# Dizionario per tenere gli agenti in memoria
agents_cache = {}

async def create_agent(
                       model_name="gpt-4o-mini", 
                       id=None,
                       name=None,
                       instructions=None,
                       site=None,
                       create_calendar=False,
                       use_search_engines=False, 
                       recreateSite=False,
                       recreateFiles=False) -> Tuple[str, CompiledGraph]:
    """
    Crea un agente configurato con diversi strumenti e modelli.

    Args:
        model_name (str): Il nome del modello da utilizzare.
        persist_directory (str): La directory per il retriever basato su embeddings.
        db_uri (str): URI del database SQL.
        db_params (dict): Parametri per la connessione PostgreSQL (opzionale).
        create_docs (bool): Se creare documenti nel caso in cui non ci siano embeddings.

    Returns:
        Tuple[str, CompiledGraph]: Un ID dell'agente e l'istanza dell'agente configurata.
    """
    # Inizializza l'agente
    agent_manager = AgentManager(id, model_name=model_name)

    agent_manager.add_name(name)
    
    # Aggiungi strumenti di ricerca
    if use_search_engines:
        agent_manager.add_search_tools()

    # Creo documenti del sito
    if site and id:
        agent_manager.configure_paths(persist_directory=id)
        agent_manager.add_site(site=site)
        # Aggiungi il retriever basato su embeddings
        agent_manager.add_retriever(create_site_docs=recreateSite,create_files_docs=recreateFiles)
        csvpath = agent_manager.find_event_csv() 
        print("csvpath:")
        print(csvpath)
        ## se esiste un cvs eventi lo carico nel database per i tools
        if csvpath:
            agent_manager.load_events_from_csv(csvpath)
            agent_manager.add_sql_tookit()
            

    # Aggiungi istruzioni
    if instructions:
        agent_manager.add_instructions(instructions=instructions)
    


    # Crea il calendario se necessario
    if create_calendar:
        print("Creo il calendario")
        await agent_manager.add_calendar()

    # Crea l'agente
    agent = agent_manager.create_agent()
    print("Agente creato correttamente")
    
    return agent


class TokenQueueStreamingHandler(AsyncCallbackHandler):
    """LangChain callback handler for streaming LLM tokens to an asyncio queue."""

    def __init__(self, queue: asyncio.Queue):
        self.queue = queue

    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        if token:
            await self.queue.put(token)


        
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestore del ciclo di vita dell'applicazione FastAPI."""

    try:
        global agents_cache
        print("Avvio del ciclo di vita (lifespan): caricamento degli agenti")

        # Carica tutte le configurazioni degli agenti
        all_agent_configs = agent_config_manager.load_all_agent_configs()
        print("all_agent_configs")
        print(all_agent_configs)
        agent_config_manager.close()

        # Ricrea ogni agente
        for agent_config in all_agent_configs:
            id = agent_config["id"]
            agent = await create_agent(
                model_name=agent_config["model_name"],
                id=id,
                name=agent_config["name"],
                instructions=agent_config["instructions"],
                site=agent_config["site"],
                create_calendar=agent_config["create_calendar"],
                use_search_engines=agent_config["use_search_engines"],
                recreateSite=False,  # non creare i documenti al momento del caricamento  in app devono esserci già
                recreateFiles=False  # non creare i documenti al momento del caricamento devono esserci già
            )
            # Aggiungi l'agente nella cache
            agents_cache[id] = agent

        print("Agenti caricati correttamente:")
        print(agents_cache)

        # Assegna il checkpointer agli agenti (se necessario)
        """for agent_id, agent in agents_cache.items():
            async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as saver:
                agent.checkpointer = saver"""

        yield  # L'app è pronta a ricevere richieste
    except Exception as e:
        logging.error("Errore durante il caricamento degli agenti all'avvio", exc_info=True)
        raise

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTP Exception: {exc.detail}", exc_info=True)
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )

@app.middleware("http")
async def check_auth_header(request: Request, call_next):
    if auth_secret := os.getenv("AUTH_SECRET"):
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return Response(status_code=401, content="Missing or invalid token")
        if auth_header[7:] != auth_secret:
            return Response(status_code=401, content="Invalid token")
    return await call_next(request)


def _parse_input(user_input: UserInput) -> Tuple[Dict[str, Any], str]:
    run_id = uuid4()
    thread_id = user_input.thread_id or str(uuid4())
    input_message = ChatMessage(type="human", content=user_input.message)
    kwargs = dict(
        input={"messages": [input_message.to_langchain()]},
        config=RunnableConfig(
            configurable={"thread_id": thread_id, "model": user_input.model},
            run_id=run_id,
        ),
    )
    return kwargs, run_id


@app.post("/invoke")
async def invoke(user_input: UserInput) -> ChatMessage:
    agent: CompiledGraph = app.state.agent
    kwargs, run_id = _parse_input(user_input)
    try:
        response = await agent.ainvoke(**kwargs)
        output = ChatMessage.from_langchain(response["messages"][-1])
        output.run_id = str(run_id)
        return output
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.options("/invoke")
async def options_handler():
    return {
        "Allow": "OPTIONS, POST",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "OPTIONS, POST",
        "Access-Control-Allow-Headers": "Content-Type, Authorization",
    }
async def message_generator(user_input: StreamInput) -> AsyncGenerator[str, None]:
    # Ottieni l'agente in base all'ID
    agent: CompiledGraph = agents_cache.get(int(user_input.id))
    print("agent")
    print(agent)
    if agent is None:
        yield f"data: {json.dumps({'type': 'error', 'content': 'Agent not found'})}\n\n"
        return
    
    kwargs, run_id = _parse_input(user_input)

    output_queue = asyncio.Queue(maxsize=10)
    if user_input.stream_tokens:
        kwargs["config"]["callbacks"] = [TokenQueueStreamingHandler(queue=output_queue)]

    async def run_agent_stream():
        async for s in agent.astream(**kwargs, stream_mode="updates"):
            print("faccio output queu")
            await output_queue.put(s)
        await output_queue.put(None)

    print("prima asyncio.create_task")
    stream_task = asyncio.create_task(run_agent_stream())
    print("dopo asyncio.create_task")

    while s := await output_queue.get():
        if isinstance(s, str):
            yield f"data: {json.dumps({'type': 'token', 'content': s})}\n\n"
            continue

        new_messages = []
        for _, state in s.items():

            if "messages" in state:
                new_messages.extend(state["messages"])
        for message in new_messages:
            print(message)
            try:
                chat_message = ChatMessage.from_langchain(message)
                chat_message.run_id = str(run_id)
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'content': f'Error parsing message: {e}'})}\n\n"
                continue
            if chat_message.type == "human" and chat_message.content == user_input.message:
                continue
            yield f"data: {json.dumps({'type': 'message', 'content': chat_message.dict()})}\n\n"

    await stream_task
    yield "data: [DONE]\n\n"



@app.post("/stream")
async def stream_agent(user_input: StreamInput):
    print("agent cache")
    print(agents_cache)
    agent = agents_cache.get(int(user_input.id))
    if agent is None:
        raise HTTPException(status_code=404, detail=f"stream endpoint, Agent not found, richiesta:{user_input.id}")
    return StreamingResponse(message_generator(user_input), media_type="text/event-stream")




# Endpoint di esempio per verificare gli agenti caricati
@app.get("/agents")
async def get_agents():
    return {"agents": list(agents_cache.keys())}

@app.post("/feedback")
async def feedback(feedback: Feedback):
    client = LangsmithClient()
    kwargs = feedback.kwargs or {}
    client.create_feedback(
        run_id=feedback.run_id,
        key=feedback.key,
        score=feedback.score,
        **kwargs,
    )
    return {"status": "success"}

class AgentConfig(BaseModel):
    agent_id: int
    recreateFiles: Optional[bool] = False
    recreateSite: Optional[bool] = False

# Route per creare una nuova configurazione di agente
@app.post("/agents/create")
async def create_agent_config(aconfig: AgentConfig):
    print(f"id {aconfig.agent_id}")
    id = aconfig.agent_id
    try:
        # Ottieni il valore dalla configurazione
        config = agent_config_manager.load_agent_config(id)  # Esempio: carica l'agente con ID 1
        print(config)

        # Poi genero tutto l'agente compreso documenti e tabella eventi
        agent = await create_agent(
            model_name=config.get("model_name", "gpt-4o-mini"),
            id=id,
            name=config.get("name"),
            instructions=config.get("instructions", ""),
            site=config.get("site"),
            create_calendar=config.get("create_calendar", False),
            use_search_engines=config.get("use_search_engines", False),
            recreateSite=aconfig.recreateSite,
            recreateFiles=aconfig.recreateFiles
            )
        print("finita creazione agent")
        agent_config_manager.close()

        agents_cache[id] = agent

        return {"id": id, "message": "Configurazione salvata con successo"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# Route per creare una nuova configurazione di agente
@app.post("/agents/add_calendar")
async def create_agent_calendar(config: Dict[str, Any]) -> Dict[str, Any]:
    
    try:
        agent_manager.add_name(config.get("name"))
        agent_manager.configure_paths(config.get("persist_directory"))
        await agent_manager.add_calendar()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"agent_name": config.get("name"), "message": "Calendario salvato con successo"}