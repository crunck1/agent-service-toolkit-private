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
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.tools import BraveSearch



from agent.tools import calculator
from agent.llama_guard import LlamaGuard, LlamaGuardOutput, SafetyAssessment


class AgentState(MessagesState):
    safety: LlamaGuardOutput
    is_last_step: IsLastStep


# NOTE: models with streaming=True will send tokens as they are generated
# if the /stream endpoint is called with stream_tokens=True (the default)
models = {
    "gpt-4o-mini": ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0.5, streaming=True),
}

if os.getenv("GROQ_API_KEY") is not None:
    models["llama-3.1-70b"] = ChatGroq(model="llama-3.1-70b-versatile", temperature=0.5)

 #web_search = TavilySearchResults(include_answer=True,include_raw_content=True)


## la roba mia:
from langchain.vectorstores import utils as chromautils
from  langchain.schema import Document
import json
from typing import Iterable
from langchain_text_splitters import  MarkdownTextSplitter
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool
 
web_search = BraveSearch.from_api_key(name="BraveSearch", api_key='BSAb-crb_t58vXgmzxSRsfFRb2Z2nXO', search_kwargs={"count": 20, "search_lang":"it","summary":True})

duck_search = DuckDuckGoSearchResults(name="DuckDuckGoSearch", region="it-it",max_results=20)


#tools = [web_search, duck_search]



def load_docs_from_jsonl(file_path)->Iterable[Document]:
    array = []
    with open(file_path, 'r') as jsonl_file:
        for line in jsonl_file:
            data = json.loads(line)
            obj = Document(**data)
            array.append(obj)
    return array
docs = load_docs_from_jsonl('/app/agent/scala2.jsonl')
docs = chromautils.filter_complex_metadata(docs)




text_splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()


tool = create_retriever_tool(
    retriever,
    "scala_search",
    "Trova tutte le informazioni sul teatro la scala di Milano. Per qualsiasi domanda sul teatro la scala, devi usare questo strumento!",
)  

## fine roba mia


tools = [ tool, web_search, duck_search]
"""
Nota:
- Controlla tutti i documenti che hai a disposizione nello strumento venezia_search per completare la risposta, non ti fermare al primo.
- Non inserire il testo "Teatro La Fenice" nella query nello strumento venezia_search.

Se non trovi informazioni utili sul primo strumento utilizza tutti gli altri strumenti per ottenere l'informazione.
Per quanto riguarda informazioni sugli orari e date degli spettacoli utilizza il documento con "pathname" = "/calendario/"
 """

# Add weather tool if API key is set
# Register for an API key at https://openweathermap.org/api/
if os.getenv("OPENWEATHERMAP_API_KEY") is not None:
    tools.append(OpenWeatherMapQueryRun(name="Weather"))

current_date = datetime.now().strftime("%B %d, %Y")
instructions = f"""
Sei un utile assistente di ricerca per il sito https://www.teatroallascala.org/ del teatro la scala
 di milano con la capacità di cercare sul web e utilizzare altri strumenti per aiutare gli utenti
 nel loro processo di ricerca spettacoli e acquisto biglietti, anche sul sito vivaticket.com.
 Rispondi sempre con educazione e con informazioni utili per l'utente .

La data odierna è {current_date}. Considera sempre questa data quando ti viene richiesto un parametro temporale.


FORMATTARE SEMPRE LA RISPOSTA IN UN BLOCCO DIV DEL FORMATO HTML (senza immettere il testo ```html)

L'utente ha 3 domande che non riguardano il teatro la scala o vivaticket. Alla quarta domanda rispondi che non puoi rispondere a questo tipo di domande
che esulano dal servizio di assistenza spettacoli o biglietteria.

NOTA: L'UTENTE NON PUÒ VEDERE LA RISPOSTA DELLO STRUMENTO.

Lo strumento "BraveSearch" può fare ricerche unicamente sul dominio teatroallascala.org, vivaticket.com

Usa lo strumento chiamato "DuckDuckGoSearch" quando devi fare ricerche su vivaticket.com utilizzando il parametro site:vivaticket.com


"""


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
"""
llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0)
prompt = ChatPromptTemplate.from_messages(
             [
                 ("system", instructions),
                 ("placeholder", "{messages}"),
             ]
         )
 memory = MemorySaver()

qa_assistant_react = create_react_agent(
        model=llm,  # Utilizza il modello fornito
        tools=tools,  # Gli strumenti che l'agente può utilizzare
        debug=True , # (opzionale) Abilita la verbosità per debug
        messages_modifier=instructions,
        checkpointer=memory
    ) 
"""


if __name__ == "__main__":
    import asyncio
    from uuid import uuid4
    from dotenv import load_dotenv

    load_dotenv()

    async def main():
        inputs = {"messages": [("user", "Find me a recipe for chocolate chip cookies")]}
        result = await qa_assistant.ainvoke(
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

    asyncio.run(main())
