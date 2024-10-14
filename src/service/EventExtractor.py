import re
import json
import psycopg2
import asyncio
from bs4 import BeautifulSoup
from typing import List, Dict, Iterable
from langchain.schema import Document
from langchain_community.agent_toolkits.sql.toolkit import   SQLDatabaseToolkit
from datetime import datetime
from langchain_core.messages import AIMessage, SystemMessage
import logging,os
logging.basicConfig(filename='errori.log', level=logging.DEBUG)


class EventExtractor:
    def __init__(self, model, db_uri, table_name):
        self.model = model
        self.db_uri = db_uri
        self.table_name = self.sanitize_table_name(table_name)

    """     @staticmethod
    def filter_calendario_docs(docs: List[Document]) -> List[Document]:
        return [doc for doc in docs if "calendario" in doc.metadata.get("source", "").lower()]
     """



    @staticmethod
    def filter_calendario_docs(docs: List[Document]) -> List[Document]:
        """Filtra i documenti che contengono la parola 'calendario'."""
        filtered_docs = []
        print(f"sono in filter_calendario_docs, len= {len(docs)}")
        logging.warning(f"sono in filter_calendario_docs, len= {len(docs)}")
        print("Current working directory:", os.getcwd())
        

        for doc in docs:
            try:
                # Controlla il tipo di oggetto
                if not isinstance(doc, Document):
                    print(f"Elemento non di tipo Document: {doc}")
                    continue
                
                # Controlla se 'metadata' e 'source' sono presenti
                pathname = doc.metadata.get("pathname", "")
                print(doc.metadata)
                if pathname is None or '':
                    print(f"Metadata mancante per il documento: {doc}")
                    continue
                
                # Logga il valore di 'source' per il debugging
                print(f"Verificando il documento con source: '{pathname}'")

                # Verifica se 'calendario' è presente
                if "calendario" in pathname.lower():
                    filtered_docs.append(doc)

            except Exception as e:
                print(f"Errore durante l'elaborazione del documento: {e}", exc_info=True)

        return filtered_docs



    @staticmethod
    def extract_json_from_response(response_text):
        """Estrae il blocco JSON da una risposta con una regex."""
        json_match = re.search(r'```json\n({.*?})\n```', response_text, re.DOTALL)
        if json_match:
            json_text = json_match.group(1)
            try:
                print(json_text)
                return json.loads(json_text)
            except json.JSONDecodeError:
                print("Errore nella decodifica del JSON")
                return None
        else:
            print("Nessun JSON trovato nella risposta")
            return None
    
    async def ask_model_for_selectors(self, document: Document):
        # Usa il modello per ottenere i s0elettori dal documento
        response = await self.model.ainvoke(
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
                            """),
                AIMessage(content=document.page_content[:10000])
            ]
        )
        # Estrai il contenuto della risposta e prova a interpretarlo come JSON
        selectors = self.extract_json_from_response(response.content)

        return selectors

    @staticmethod
    def extract_events_with_selectors(html_content: str, selectors: Dict[str, str]) -> List[Dict[str, str]]:
        """Estrae eventi usando i selettori ottenuti."""
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

    @staticmethod
    def replace_non_alphanumeric_with_space(s: str) -> str:
        """Sostituisce i caratteri non alfanumerici con spazi."""
        return re.sub(r'[^a-zA-Z0-9]', ' ', s)

    async def process_and_extract_events_md(self, docs: List[Document]) -> List[Document]:
        """Processa i documenti ed estrae gli eventi in formato Markdown."""
        event_documents = []

        for doc in docs:
            # Ottenere i selettori dal modello
            selectors = await self.ask_model_for_selectors(doc)
            print(f"selettori css trovati:")
            print(selectors)
            if selectors is None or '':
                continue
            
            # Estrarre eventi usando i selettori
            events = self.extract_events_with_selectors(doc.page_content, selectors)

            # Creare un documento per ogni evento estratto in formato Markdown
            for event in events:
                date = event.get('date', 'Data non disponibile')
                rdate = self.replace_non_alphanumeric_with_space(date or 'Data non disponibile')
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

    async def create_event_docs(self, docs: List[Document]):
        """Processa i documenti filtrando per calendario ed estrae gli eventi."""
        calendario_docs = self.filter_calendario_docs(docs)
        print(f"numero calendario docs:  {len(calendario_docs)}")
        return await self.process_and_extract_events_md(calendario_docs)
    
    import re

    def sanitize_table_name(self, input_string: str) -> str:

        # Lista di parole riservate del database (possono variare in base al database)
        RESERVED_WORDS = {
            "select", "insert", "update", "delete", "create", "drop", "alter", "table", "from", "where", "join", "null"
        }
        # 1. Rimuove tutti i caratteri non alfanumerici o underscore
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', input_string)

        # 2. Se inizia con un numero, aggiungi un underscore all'inizio
        if sanitized and sanitized[0].isdigit():
            sanitized = f"_{sanitized}"

        # 3. Converti in minuscolo
        sanitized = sanitized.lower()

        # 4. Verifica che non sia una parola riservata
        if sanitized in RESERVED_WORDS:
            sanitized = f"{sanitized}_tbl"

        # Limita la lunghezza del nome (ad esempio, 63 caratteri per PostgreSQL)
        sanitized = sanitized[:63]

        return sanitized

    
    def import_events_to_postgres(self, event_docs: List[Document]):
        # Connessione al database PostgreSQL
        print(f"importo su postrger num: {len(event_docs)}")
        conn = psycopg2.connect(self.db_uri)
        cursor = conn.cursor()


        # Creare la tabella se non esiste
        cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id SERIAL PRIMARY KEY,
                source TEXT,
                event_date DATE,
                event_time TEXT,
                event_title TEXT,
                event_author TEXT,
                content TEXT
            )
        ''')
        cursor.execute(f'''
            CREATE UNIQUE INDEX IF NOT EXISTS unique_event ON {self.table_name}  (event_date, event_title)
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
            # Assicurati che la funzione convert_date restituisca una data valida
            # event_date = convert_date(event_date)

            # Inserire l'evento nel database
            if event_date and event_title:
                cursor.execute(f'''
                    INSERT INTO {self.table_name} (source, event_date, event_time, event_title, event_author, content)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (event_date, event_title) DO NOTHING
                ''', (source, event_date, event_time, event_title, event_author, content))

        # Salvare le modifiche nel database
        conn.commit()

        # Chiudere la connessione
        cursor.close()
        conn.close()

    async def run(self, docs: List[Document]):
        """Esegue il processo di estrazione e importazione nel database."""
        event_docs = await self.create_event_docs(docs)
        print("eventi trovati da create_event_docs")
        print(event_docs)
        self.import_events_to_postgres(event_docs)
