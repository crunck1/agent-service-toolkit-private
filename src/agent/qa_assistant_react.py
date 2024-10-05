from datetime import datetime
import os
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
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.tools import BraveSearch
from typing import List, Dict
from bs4 import BeautifulSoup
from agent.llama_guard import LlamaGuard, LlamaGuardOutput, SafetyAssessment
from dataclasses import dataclass
import json
from bs4 import BeautifulSoup
from langchain.vectorstores import utils as chromautils
from  langchain.schema import Document
import json
from typing import Iterable
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool
import re
from datetime import datetime
from langchain.schema import Document
import psycopg2
from langchain_community.utilities import  SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import   SQLDatabaseToolkit
from langchain_core.tools import tool
from langchain import hub
import asyncio
from functools import partial
import asyncio




class AgentState(MessagesState):
    safety: LlamaGuardOutput
    is_last_step: IsLastStep


# NOTE: models with streaming=True will send tokens as they are generated
# if the /stream endpoint is called with stream_tokens=True (the default)
models = {
    "gpt-4o-mini": ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0.5, streaming=True),
   # "ollama":OllamaLLM(model="llama3.1")
}

if os.getenv("GROQ_API_KEY") is not None:
    models["llama-3.1-70b"] = ChatGroq(model="llama-3.1-70b-versatile", temperature=0.5)

web_search = BraveSearch.from_api_key(name="BraveSearch", api_key='BSAb-crb_t58vXgmzxSRsfFRb2Z2nXO', search_kwargs={"count": 20, "search_lang":"it","summary":True})

duck_search = DuckDuckGoSearchResults(name="DuckDuckGoSearch", region="it-it",max_results=20)


def load_docs_from_jsonl(file_path)->Iterable[Document]:
    array = []
    with open(file_path, 'r') as jsonl_file:
        for line in jsonl_file:
            data = json.loads(line)
            obj = Document(**data)
            array.append(obj)
    return array
docs = load_docs_from_jsonl('/app/agent/scala2.jsonl')


def filter_calendario_docs(docs: List[Document]) -> List[Document]:
    return [doc for doc in docs if "calendario" in doc.metadata.get("source", "").lower()]

def extract_json_from_response(response_text):
    # Usa una regex per estrarre il blocco JSON dalla risposta
    json_match = re.search(r'```json\n({.*?})\n```', response_text, re.DOTALL)
    if json_match:
        json_text = json_match.group(1)
        try:
            # Converti il testo JSON in un dizionario Python
            json_data = json.loads(json_text)
            return json_data
        except json.JSONDecodeError:
            print("Errore nella decodifica del JSON")
            return None
    else:
        print("Nessun JSON trovato nella risposta")
        return None

async def ask_model_for_selectors(model, document: Document):
    # Usa il modello per ottenere i s0elettori dal documento
    response = await model.ainvoke(
        [
            SystemMessage(content="""Tu sei un esperto parser HTML.
                          All'interno di questa pagina HTML si trovano molte date  di spettacoli di vario genere.
                          Il tuo compito è trovare i selettori HTML per estrarre da ogni data tutti gli eventi al suo interno di questo contenuto HTML.
                          Prima troverai la collezioni di nodi html che rappresentano tutte le date (possono essere molti).
                          Poi da questi gli eventi al suo interno con orario, il titolo, autore.
                          Fai delle prove per verificare che funzionino, se non funziona rispondi che non ci riesci: cerchi prima la data che ha il selettore 
                          date_selector e poi cerchi   il titolo , l'orario, e l'autore all'interno del testo (innerText) dell'elemento.
                          I selettori devono essere in grado di fornire   gli elementi HTML al cui interno trovare questi dati.
                          I selettori titolo , l'orario, e l'autore devono essere relativi all'elemento html date_selector, per esempio 
                          Rispondi con un JSON valido e ben formattato con questi campi:
                          { 
                            'date_selector': 'testo di esempio',
                            'time_selector': 'testo di esempio',
                            'title_selector': 'testo di esempio',
                            'author_selector': 'testo di esempio'
                          } Rispondi con json valido perchè lo userò poi in seguito. 
                          Non inserire commenti o altro testo che non sia json.
                         """),
            AIMessage(content=document.page_content[:10000])
        ]
    )

    # Estrai il contenuto della risposta e prova a interpretarlo come JSON
    selectors = extract_json_from_response(response.content)

    return selectors

# Funzione per estrarre gli eventi in base ai selettori ricevuti
def extract_events_with_selectors(html_content: str, selectors: Dict[str, str]) -> List[Dict[str, str]]:

    soup = BeautifulSoup(html_content, 'html.parser')
    dates = soup.select(selectors.get('date_selector', ''))

    extracted_events = []
    for date in dates:
        try:
            extracted_event = {
                'date': date.get('id') if date else None,
                'time': date.select_one(selectors.get('time_selector', '')).text.strip() if date.select_one(selectors.get('time_selector', '')) else None,
                'title': date.select_one(selectors.get('title_selector', '')).text.strip() if date.select_one(selectors.get('title_selector', '')) else None,
                'author': date.select_one(selectors.get('author_selector', '')).text.strip() if date.select_one(selectors.get('author_selector', '')) else None
            }
            extracted_events.append(extracted_event)
        except:
            print("Errore nel trovare evento")

    return extracted_events



@dataclass
class Document:
    page_content: str
    metadata: dict

async def process_and_extract_events_old(model, docs: List[Document]):
    events_document = []

    for doc in docs:
        # Ottenere i selettori dal modello
        selectors = await ask_model_for_selectors(model, doc)
        # Estrarre eventi usando i selettori
        events = extract_events_with_selectors(doc.page_content, selectors)

        # Generare il contenuto HTML
        html_content = "<html><head><title>Eventi</title></head><body>"
        html_content += f"<h1>Eventi estratti dal documento: {doc.metadata['source']}</h1>"
        
        for event in events:
            html_content += "<div class='event'>"
            html_content += f"<p><strong>Data:</strong> {event['date']}</p>"
            html_content += f"<p><strong>Orario:</strong> {event['time']}</p>"
            html_content += f"<p><strong>Titolo:</strong> {event['title']}</p>"
            html_content += f"<p><strong>Autore:</strong> {event['author']}</p>"
            html_content += "</div><hr>"

        html_content += "</body></html>"

        # Creare un documento HTML
        events_document.append(Document(
            page_content=html_content,
            metadata={"source": doc.metadata["source"], "type": "event_extraction_html"}
        ))

    return events_document

def replace_non_alphanumeric_with_space(s: str) -> str:
    return re.sub(r'[^a-zA-Z0-9]', ' ', s)



# Dizionario per mappare i mesi italiani ai numeri dei mesi
mesi_italiani = {
    'gennaio': '01',
    'febbraio': '02',
    'marzo': '03',
    'aprile': '04',
    'maggio': '05',
    'giugno': '06',
    'luglio': '07',
    'agosto': '08',
    'settembre': '09',
    'ottobre': '10',
    'novembre': '11',
    'dicembre': '12'
}

def convert_date(data_italiana):
    #print(f"Inizio a convertire: {data_italiana} (lunghezza: {len(data_italiana)}), Inizia con underscore: {data_italiana.startswith('_')}")
    
    # Controlla se la data inizia con l'underscore e ha il formato corretto
    if not data_italiana.startswith("_") or len(data_italiana) <= 1:
        return None
    
    # Rimuovi l'underscore iniziale e gli spazi extra
    data_italiana = data_italiana.lstrip("_").strip()
    
    try:
        # Splitta la data in giorno, mese e anno
        giorno, mese, anno = data_italiana.split('-')
        
        # Controlla se il mese è valido
        if mese not in mesi_italiani:
            return None
        
        # Costruisci la data nel formato inglese (YYYY-MM-DD)
        mese_numero = mesi_italiani[mese]
        data_inglese = f"{anno}-{mese_numero}-{giorno.zfill(2)}"
        
        return data_inglese
    
    except ValueError as e:
        print(f"Errore nella conversione: {e}")
        return None



async def process_and_extract_events_md(model, docs: List[Document]) -> List[Document]:
    event_documents = []

    for doc in docs:
        # Ottenere i selettori dal modello
        selectors = await ask_model_for_selectors(model, doc)
        
        # Estrarre eventi usando i selettori
        events = extract_events_with_selectors(doc.page_content, selectors)

        # Creare un documento per ogni evento estratto in formato Markdown
        for event in events:
            date = event.get('date', 'Data non disponibile')
            rdate = replace_non_alphanumeric_with_space(date or 'Data non disponibile')
            # Creare il contenuto in Markdown per il singolo evento
            markdown_content = f"# Evento estratto dal documento: {doc.metadata['source']}\n\n"
            markdown_content += f"**Data:** {rdate}\n\n"
            markdown_content += f"**Orario:** {event['time']}\n\n"
            markdown_content += f"**Titolo:** {event['title']}\n\n"
            markdown_content += f"**Autore:** {event['author']}\n\n"
            markdown_content += "---\n"

            # Aggiungi ogni evento come un documento separato
            event_documents.append(Document(
                page_content=markdown_content,
                metadata={
                    "source": doc.metadata["source"], 
                    "type": "event_extraction_markdown",
                    "event_date": event['date'],
                    "event_time": event['time'],
                    "event_title": event['title'],
                    "event_author": event['author']
                }
            ))

    return event_documents


# Funzione per importare eventi in un database PostgreSQL
def import_events_to_postgres(event_docs: List[Document], db_params: dict):
    # Connessione al database PostgreSQL
    conn = psycopg2.connect(
        host=db_params['host'],
        port=db_params['port'],
        dbname=db_params['dbname'],
        user=db_params['user'],
        password=db_params['password']
    )
    cursor = conn.cursor()

    # Creare la tabella se non esiste
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS eventi (
            id SERIAL PRIMARY KEY,
            source TEXT,
            event_date DATE,
            event_time TEXT,
            event_title TEXT,
            event_author TEXT,
            content TEXT
        )
    ''')
    cursor.execute('''
        CREATE UNIQUE INDEX IF NOT EXISTS unique_event ON eventi (event_date, event_title)
    ''')


    # Iterare su tutti gli eventi estratti
    for doc in event_docs:
        # Estrarre i metadati e il contenuto del documento
        source = doc.metadata.get('source', None)
        event_date = doc.metadata.get('event_date', None)
        event_time = doc.metadata.get('event_time', None)
        event_title = doc.metadata.get('event_title', None)
        event_author = doc.metadata.get('event_author', None)
        content = doc.page_content

        if not event_date:
            continue
        event_date = convert_date(event_date)  # Assicurati che la funzione convert_date restituisca una data valida

        # Inserire l'evento nel database
        if event_date and event_title:
            cursor.execute('''
                INSERT INTO eventi (source, event_date, event_time, event_title, event_author, content)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (event_date, event_title) DO NOTHING
            ''', (source, event_date, event_time, event_title, event_author, content))

    # Salvare le modifiche nel database
    conn.commit()

    # Chiudere la connessione
    cursor.close()
    conn.close()


async def process_and_extract_events(model, docs: List[Document]) -> List[Document]:
    event_documents = []

    for doc in docs:
        # Ottenere i selettori dal modello
        selectors = await ask_model_for_selectors(model, doc)
        
        # Estrarre eventi usando i selettori
        events = extract_events_with_selectors(doc.page_content, selectors)

        # Creare un documento per ogni evento estratto
        for event in events:
            # Creare il contenuto HTML per il singolo evento
            html_content = "<html><head><title>Evento</title></head><body>"
            html_content += f"<h1>Evento estratto dal documento: {doc.metadata['source']}</h1>"
            html_content += "<div class='event'>"
            html_content += f"<p><strong>Data:</strong> {re.sub(r'[^a-zA-Z0-9]', ' ', event['date'])}</p>"
            html_content += f"<p><strong>Orario:</strong> {event['time']}</p>"
            html_content += f"<p><strong>Titolo:</strong> {event['title']}</p>"
            html_content += f"<p><strong>Autore:</strong> {event['author']}</p>"
            html_content += "</div><hr>"
            html_content += "</body></html>"

            # Aggiungi ogni evento come un documento separato
            event_documents.append(Document(
                page_content=html_content,
                metadata={
                    "source": doc.metadata["source"], 
                    "type": "event_extraction_html",
                    "event_date": event['date'],
                    "event_time": event['time'],
                    "event_title": event['title'],
                    "event_author": event['author']
                }
            ))

    return event_documents
    

db = SQLDatabase.from_uri("postgresql://claudio:settanta9-a@postgres:5432/agentic")
docs = chromautils.filter_complex_metadata(docs)
vectorstore = Chroma(
    persist_directory="./embeddings",  # Specifica la directory di caricamento
    embedding_function=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()

tool = create_retriever_tool(
    retriever,
    "scala_search",
    """
    Usa questo tool per ricercare le informazioni piu pertinenti alla query nei documenti a tua disposizione.
   
   """,
)  

event_docs = []
# Funzione per processare i documenti e ottenere gli eventi
async def create_event_docs(model, docs: List[Document]):
    calendario_docs = filter_calendario_docs(docs)
    ev = await process_and_extract_events_md(model, calendario_docs)  # Funzione asincrona che estrae gli eventi
    return ev  # Restituisci i documenti estratti

# Wrapper per gestire la chiamata asincrona e popolare event_docs
async def main():
    global event_docs  # Rende event_docs accessibile
    model = models["gpt-4o-mini"]  # Il tuo modello
    event_docs = await create_event_docs(model, docs)


# Esegui il codice
asyncio.run(main())
event_docs = chromautils.filter_complex_metadata(event_docs)

db_params = {
    'host': 'postgres',
    'port': '5432',
    'dbname': 'agentic',
    'user': 'claudio',
    'password': 'settanta9-a'
}
import_events_to_postgres(event_docs, db_params)


toolkit = SQLDatabaseToolkit(db=db, llm=models["gpt-4o-mini"])
tools = [ tool, web_search, duck_search] + toolkit.get_tools()

# Add weather tool if API key is set
# Register for an API key at https://openweathermap.org/api/
if os.getenv("OPENWEATHERMAP_API_KEY") is not None:
    tools.append(OpenWeatherMapQueryRun(name="Weather"))

current_date = datetime.now().strftime("%B %d, %Y")

instructions = f"""
Sei un utile assistente di ricerca per il sito https://www.teatroallascala.org/ del teatro la scala
 di milano con la capacità di cercare sul web e utilizzare altri strumenti per aiutare gli utenti
 nel loro processo di ricerca spettacoli e acquisto biglietti, anche sul sito vivaticket.com.

La data odierna è {current_date}. Considera sempre questa data quando ti viene richiesto un parametro temporale.

Nel caso ti vengano richieste informazioni sulle date di uno spettacolo:
1) usa sempre lo strumento  "sql_db_schema", "sql_db_list_tables" e poi "sql_db_query" (nell'ordine) 
2) Quando usi lo strumento "sql_db_query"  applica sempre un limite di 5 risultati a meno che non ti venga esplicitamente richiesto
3) puoi fare al massimo 5 step 
4) non cercare mai nel passato a meno che non ti venga esplicitamente richiesto.

FORMATTARE SEMPRE LA RISPOSTA IN UN BLOCCO DIV DEL FORMATO HTML (senza immettere il testo ```html)
Non usare elenchi <ul><li>, utilizza dei paragrafi <p> e separa ogni evento in maniera chiara e precisa

L'utente ha 3 domande che non riguardano il teatro la scala o vivaticket. Alla quarta domanda rispondi che non puoi rispondere a questo tipo di domande
che esulano dal servizio di assistenza spettacoli o biglietteria.

NOTA: L'UTENTE NON PUÒ VEDERE LA RISPOSTA DELLO STRUMENTO se richiesto

Lo strumento "BraveSearch" può fare ricerche unicamente sul dominio teatroallascala.org, vivaticket.com

Usa lo strumento chiamato "DuckDuckGoSearch" quando devi fare ricerche su vivaticket.com utilizzando il parametro site:vivaticket.com

"""


prompt = hub.pull("hwchase17/react")
def wrap_model(model: BaseChatModel):
    model = model.bind_tools(tools)
    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(content=instructions)] + state["messages"],
        name="StateModifier",
    )
    return preprocessor | model


def format_safety_message(safety: LlamaGuardOutput) -> AIMessage:
    content = (
        f"This conversation was flagged for unsafe content: {', '.join(safety.unsafe_categories)}"
    )
    return AIMessage(content=content)


async def acall_model(state: AgentState, config: RunnableConfig):
    m = models[config["configurable"].get("model", "gpt-4o-mini")]
    model_runnable = wrap_model(m)
    response = await model_runnable.ainvoke(state, config)

    # Run llama guard check here to avoid returning the message if it's unsafe
    llama_guard = LlamaGuard()
    safety_output = await llama_guard.ainvoke("Agent", state["messages"] + [response])
    if safety_output.safety_assessment == SafetyAssessment.UNSAFE:
        return {"messages": [format_safety_message(safety_output)], "safety": safety_output}

    if state["is_last_step"] and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, need more steps to process this request.",
                )
            ]
        }
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


async def llama_guard_input(state: AgentState, config: RunnableConfig):
    llama_guard = LlamaGuard()
    safety_output = await llama_guard.ainvoke("User", state["messages"])
    return {"safety": safety_output}


async def block_unsafe_content(state: AgentState, config: RunnableConfig):
    safety: LlamaGuardOutput = state["safety"]
    return {"messages": [format_safety_message(safety)]}


# Define the graph
agent = StateGraph(AgentState)
agent.add_node("model", acall_model)
agent.add_node("tools", ToolNode(tools))
agent.add_node("guard_input", llama_guard_input)
agent.add_node("block_unsafe_content", block_unsafe_content)
agent.set_entry_point("guard_input")


# Check for unsafe input and block further processing if found
def check_safety(state: AgentState):
    safety: LlamaGuardOutput = state["safety"]
    match safety.safety_assessment:
        case SafetyAssessment.UNSAFE:
            return "unsafe"
        case _:
            return "safe"


agent.add_conditional_edges(
    "guard_input", check_safety, {"unsafe": "block_unsafe_content", "safe": "model"}
)

# Always END after blocking unsafe content
agent.add_edge("block_unsafe_content", END)

# Always run "model" after "tools"
agent.add_edge("tools", "model")


# After "model", if there are tool calls, run "tools". Otherwise END.
def pending_tool_calls(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    else:
        return "done"


agent.add_conditional_edges("model", pending_tool_calls, {"tools": "tools", "done": END})

qa_assistant_react = agent.compile(
    checkpointer=MemorySaver(),
)


if __name__ == "__main__":
    import asyncio
    from uuid import uuid4
    from dotenv import load_dotenv

    load_dotenv()

    async def main():
        inputs = {"messages": [("user", "Find me a recipe for chocolate chip cookies")]}
        result = await qa_assistant_react.ainvoke(
            inputs,
            config=RunnableConfig(configurable={"thread_id": uuid4()}),
        )
        result["messages"][-1].pretty_print()

        # Draw the agent graph as png
        # requires:
        # brew install graphviz
        # export CFLAGS="-I $(brew --prefix graphviz)/include"
        # export LDFLAGS="-L $(brew --prefix graphviz)/lib"
        # pip install pygraphviz
        #
        # qa_assistant.get_graph().draw_png("agent_diagram.png")

    # Set up the event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Run the main function
    loop.run_until_complete(main())
