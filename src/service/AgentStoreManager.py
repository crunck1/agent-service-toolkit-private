import psycopg2
import pickle
import logging
import dill


logging.basicConfig(filename='agent.log', level=logging.INFO)

class AgentStoreManager:
    def __init__(self, db_uri):
        self.db_uri = db_uri
        try:
            # Connessione a PostgreSQL
            self.conn = psycopg2.connect(self.db_uri)
            cursor = self.conn.cursor()
            # Creazione della tabella se non esiste
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS agent_store (
                    agent_id TEXT PRIMARY KEY,
                    agent_data BYTEA
                );
            ''')
            self.conn.commit()
            cursor.close()
        except Exception as e:
            logging.error("Errore durante la connessione o creazione della tabella", exc_info=True)
            raise

    def save_agent(self, agent_id, agent):
        """Salva un agente serializzload_agentsato in PostgreSQL."""
        try:
            cursor = self.conn.cursor()
            # Serializza l'agente usando pickle
            agent_serialized = dill.dumps(agent)
            # Salva l'agente in formato binario (BYTEA)
            cursor.execute('''
                INSERT INTO agent_store (agent_id, agent_data)
                VALUES (%s, %s)
                ON CONFLICT (agent_id) DO UPDATE SET agent_data = EXCLUDED.agent_data
            ''', (agent_id, psycopg2.Binary(agent_serialized)))
            self.conn.commit()
            cursor.close()
            logging.info(f"Agente {agent_id} salvato con successo.")
        except Exception as e:
            logging.error("Errore durante il salvataggio dell'agente su PostgreSQL", exc_info=True)
            raise

    def load_agent(self, agent_id):
        """Carica un agente serializzato da PostgreSQL."""
        try:
            cursor = self.conn.cursor()
            # Recupera l'agente serializzato dal database
            cursor.execute('SELECT agent_data FROM agent_store WHERE id = %s', (agent_id,))
            row = cursor.fetchone()
            cursor.close()
            if row:
                agent_serialized = row[0]
                # Deserializza l'agente
                agent = dill.loads(agent_serialized)
                logging.info(f"Agente {agent_id} caricato con successo.")
                return agent
            else:
                logging.info(f"Agente {agent_id} non trovato.")
                return None
        except Exception as e:
            logging.error("Errore durante il caricamento dell'agente da PostgreSQL", exc_info=True)
            raise
    
    def load_all_agents(self):
        """Carica tutti gli agenti serializzati dal database."""
        try:
            cursor = self.conn.cursor()
            cursor.execute('SELECT id, agent_data FROM agent_store')
            agents = cursor.fetchall()
            cursor.close()
            loaded_agents = {}
            for agent_id, agent_serialized in agents:
                agent = dill.loads(agent_serialized)  # Deserializza l'agente
                loaded_agents[agent_id] = agent
            return loaded_agents
        except Exception as e:
            logging.error("Errore durante il caricamento degli agenti da PostgreSQL", exc_info=True)
            raise
