import psycopg2
import logging


class AgentConfigManager:
    def __init__(self, db_uri):
        self.db_uri = db_uri
        self.conn = None  # Inizializza l'attributo conn a None

    def connect(self):
        """Crea una connessione al database."""
        if self.conn is None:
            self.conn = psycopg2.connect(self.db_uri)

    def close(self):
        """Chiude la connessione al database se è aperta."""
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def save_agent_config(self, name, persist_directory, instructions, site, create_calendar=True, model_name="gpt-4o-mini", use_search_engines=True):
        """Salva la configurazione dell'agente nel database."""
        self.connect()  # Assicurati di connetterti prima di eseguire operazioni
        try:
            with self.conn.cursor() as cursor:
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS agent_streams (
                        id SERIAL PRIMARY KEY,
                        model_name TEXT NOT NULL,
                        name TEXT NOT NULL,
                        use_brave BOOLEAN NOT NULL,
                        use_duckduckgo BOOLEAN NOT NULL,
                        persist_directory TEXT NOT NULL,
                        site TEXT DEFAULT NULL,
                        instructions TEXT NOT NULL,
                        create_calendar BOOLEAN DEFAULT TRUE,
                        file_names TEXT[],
                        CONSTRAINT unique_name UNIQUE (name)  
                    );

                ''')

                cursor.execute('''
                    INSERT INTO agent_streams (model_name, name, use_brave, use_duckduckgo, persist_directory, instructions, site, create_calendar)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id;
                ''', (model_name, name, use_search_engines, persist_directory, instructions, site, create_calendar))

                agent_id = cursor.fetchone()[0]

            self.conn.commit()
            return agent_id

        except Exception as e:
            logging.error(
                "Errore durante il salvataggio della configurazione dell'agente", exc_info=True)
            if self.conn:
                self.conn.rollback()
            raise

    def load_all_agent_configs(self):
        """Carica tutte le configurazioni degli agenti dal database."""
        self.connect()  # Assicurati di connetterti prima di eseguire operazioni
        try:
            print("Faccio load_all_config")
            with self.conn.cursor() as cursor:
                cursor.execute('SELECT * FROM agent_streams')
                agents = cursor.fetchall()

            loaded_agents = []
            for agent in agents:
                id, name,  site, model_name, create_calendar,  instructions, use_search_engines, created_at, updated_at = agent
                agent_data = {
                    "id": id,
                    "model_name": model_name,
                    "name": name,
                    "use_search_engines": use_search_engines,
                    "instructions": instructions,
                    "site": site,
                    "create_calendar": create_calendar,
                }
                loaded_agents.append(agent_data)

            return loaded_agents

        except Exception as e:
            logging.error(
                "Errore durante il caricamento delle configurazioni degli agenti", exc_info=True)
            raise

    def load_agent_config(self, agent_id):
        """Carica la configurazione di un singolo agente dal database usando l'agent_id."""
        self.connect()  # Assicurati di connetterti prima di eseguire operazioni
        try:
            print(f"Carico la configurazione per l'agente con ID {agent_id}")
            with self.conn.cursor() as cursor:
                cursor.execute(
                    'SELECT * FROM agent_streams WHERE id = %s', (agent_id,))
                agent = cursor.fetchone()

            print("agent")
            print(agent)
            if agent is None:
                raise ValueError(f"Nessun agente trovato con ID {agent_id}")

            # Destrutturazione dei valori recuperati dalla query
            id, name,  site, model_name,  create_calendar,  instructions, use_search_engine, created_at, updated_at = agent

            # Crea un dizionario con i dati dell'agente
            agent_data = {
                "id": id,
                "model_name": model_name,
                "name": name,
                "use_search_engine": use_search_engine,
                "site": site,
                "instructions": instructions,
                "create_calendar": create_calendar,
            }

            return agent_data

        except Exception as e:
            logging.error(f"Errore durante il caricamento della configurazione dell'agente con ID {agent_id}", exc_info=True)
            raise

    def load_agent_files(self, agent_id):
        """Carica i files di un singolo agente dal database usando l'agent_id."""
        self.connect()  # Assicurati di connetterti prima di eseguire operazioni
        try:
            print(f"Carico i files per l'agente con ID {agent_id}")
            with self.conn.cursor() as cursor:
                cursor.execute(
                    'SELECT * FROM agent_stream_files WHERE agent_stream_id = %s', (agent_id,))
                files = cursor.fetchall()



            all_files = []
            for file in files:
                id, agent_stream_id, name, path, created_at, updated_at = file
                all_files.append({
                    'id': id,
                    'agent_stream_id': agent_stream_id,
                    'name': name,
                    'path': path,
                    'created_at': created_at,
                    'updated_at': updated_at
                })

            print("Files recuperati:", all_files)
            return all_files

        except Exception as e:
            logging.error(f"Errore durante il caricamento dei files dell'agente con ID {agent_id}", exc_info=True)
            raise
            
    def __del__(self):
        """Assicurati che la connessione venga chiusa quando l'oggetto viene distrutto."""
        self.close()
