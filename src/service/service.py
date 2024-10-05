import asyncio
from contextlib import asynccontextmanager
import json
import os
from typing import AsyncGenerator, Dict, Any, Tuple
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

db_uri = f"postgresql://claudio:settanta9-a@postgres:5432/agentic"
agent_store_manager = AgentStoreManager(db_uri=db_uri)
agent_manager = AgentManager()
agent_config_manager = AgentConfigManager(db_uri=db_uri)
# Dizionario per tenere gli agenti in memoria
agents_cache = {}

async def create_agent(model_name="gpt-4o-mini", 
                       name=None,
                       use_brave=True, 
                       use_duckduckgo=True, 
                       persist_directory=None, 
                       instructions=None,
                       site=None,
                       create_calendar=True,
                       db_uri=None, 
                       db_params=None,
                       create_docs=False) -> Tuple[str, CompiledGraph]:
    """
    Crea un agente configurato con diversi strumenti e modelli.

    Args:
        model_name (str): Il nome del modello da utilizzare.
        use_brave (bool): Se usare Brave come motore di ricerca.
        use_duckduckgo (bool): Se usare DuckDuckGo come motore di ricerca.
        persist_directory (str): La directory per il retriever basato su embeddings.
        db_uri (str): URI del database SQL.
        db_params (dict): Parametri per la connessione PostgreSQL (opzionale).
        create_docs (bool): Se creare documenti nel caso in cui non ci siano embeddings.

    Returns:
        Tuple[str, CompiledGraph]: Un ID dell'agente e l'istanza dell'agente configurata.
    """
    # Inizializza l'agente
    agent_manager = AgentManager(model_name=model_name)

    agent_manager.add_name(name)
    
    # Aggiungi strumenti di ricerca
    agent_manager.add_search_tools(use_brave=use_brave, use_duckduckgo=use_duckduckgo)

    # Configura il database SQL (se fornito)
    if create_calendar:
        agent_manager.configure_database(table_name=name)
    
    # Aggiungi il retriever solo se esiste già, senza creare documenti
    if site and persist_directory:
        agent_manager.add_persist_directory(persist_directory=persist_directory)
        agent_manager.add_site(site=site)
        # Aggiungi il retriever basato su embeddings
        agent_manager.add_retriever(create_docs=create_docs)  # Non creare documenti

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
                name=agent_config["name"],
                use_brave=agent_config["use_brave"],
                use_duckduckgo=agent_config["use_duckduckgo"],
                persist_directory=agent_config["persist_directory"],
                instructions=agent_config["instructions"],
                site=agent_config["site"],
                create_calendar=False,  # Non creare calendari al momento del caricamento
                create_docs=False  # Carica solo gli embeddings esistenti
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




""" @app.on_event("startup")
async def startup_event():
    print("parte startup event")
    all_agent_configs = agent_config_manager.load_all_agent_configs()

    # Ricrea ogni agente
    for agent_config in all_agent_configs:
        print("agent config")
        print(agent_config)
        agent_id = agent_config["agent_id"]
        agent = await create_agent(
            model_name=agent_config["model_name"],
            name=agent_config["name"],
            use_brave=agent_config["use_brave"],
            use_duckduckgo=agent_config["use_duckduckgo"],
            persist_directory=agent_config["persist_directory"],
            instructions=agent_config["instructions"],
            site=agent_config["site"],
            create_calendar=agent_config["create_calendar"]
        )

        # Salva l'agente nel dizionario degli agenti attivi
        agents_cache[agent_id] = agent
 """

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
""" 
@app.post("/agents")
async def create_agent_route(config: Dict[str, Any]) -> Dict[str, str]:
    try:
        agent_id, agent = create_agent(
            model_name=config.get("model_name", "gpt-4o-mini"),
            name=config.get("name", "esempio"),
            use_brave=config.get("use_brave", True),
            use_duckduckgo=config.get("use_duckduckgo", True),
            persist_directory=config.get("persist_directory", "./embeddings"),
            site=config.get("site", None),
            db_uri=config.get("db_uri"),
            db_params=config.get("db_params"),
            create_calendar=config.get("create_calendar")

        )
        

        # Salva l'agente nel database
        agent_store_manager.save_agent(agent_id, agent)  # Presumendo che save_agent accetti agent_id e agent

        return {"agent_id": agent_id}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
 """

""" @app.get("/agents/{agent_id}")
async def get_agent_route(agent_id: str) -> CompiledGraph:
    agent = agents.get(agent_id)
    if agent is None:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent
 """

from fastapi import Query


# Route per creare una nuova configurazione di agente
@app.post("/agents/create/{id}")
async def create_agent_config(id: int) -> Dict[str, Any]:
    print(f"id {id}")
    try:
        config = agent_config_manager.load_agent_config(id)  # Esempio: carica l'agente con ID 1


        # Poi genero tutto l'agente compreso documenti e tabella eventi
        agent = await create_agent(
            model_name=config.get("model_name", "gpt-4o-mini"),
            name=config.get("name", "esempio"),
            use_brave=config.get("use_brave", True),
            use_duckduckgo=config.get("use_duckduckgo", True),
            persist_directory=config.get("persist_directory"),
            instructions=config.get("instructions", ""),
            site=config.get("site"),
            create_calendar=config.get("create_calendar"),
            create_docs=config.get("create_docs"),
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
        agent_manager.add_persist_directory(config.get("persist_directory"))
        await agent_manager.add_calendar()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"agent_name": config.get("name"), "message": "Calendario salvato con successo"}