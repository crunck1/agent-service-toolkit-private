import psycopg2
import json
import logging

logging.basicConfig(filename='errori.log', level=logging.ERROR)

class AgentCheckpointManager:
    def __init__(self, db_uri):
        self.db_uri = db_uri
        try:
            self.conn = psycopg2.connect(self.db_uri)
            cursor = self.conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS agent_checkpoints (
                    agent_id TEXT PRIMARY KEY,
                    state TEXT NOT NULL
                );
            ''')
            self.conn.commit()
            cursor.close()
        except Exception as e:
            logging.error("Errore durante l'inizializzazione della connessione o creazione della tabella", exc_info=True)
            raise

    def save_checkpoint(self, agent_id, state):
        """Salva lo stato dell'agente nel database."""
        try:
            cursor = self.conn.cursor()
            state_json = json.dumps(state)  # Serializza il dizionario in JSON
            cursor.execute('''
                INSERT INTO agent_checkpoints (agent_id, state)
                VALUES (%s, %s)
                ON CONFLICT (agent_id) DO UPDATE SET state = EXCLUDED.state
            ''', (agent_id, state_json))
            self.conn.commit()
            cursor.close()
        except Exception as e:
            logging.error("Errore durante il salvataggio del checkpoint", exc_info=True)
            raise

    def load_checkpoint(self, agent_id):
        """Carica lo stato dell'agente dal database."""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT state FROM agent_checkpoints WHERE agent_id = %s
            ''', (agent_id,))
            row = cursor.fetchone()
            cursor.close()
            if row:
                state_json = row[0]  # Ottieni la stringa JSON
                return json.loads(state_json)  # Deserializza la stringa JSON in un dizionario
            else:
                return None
        except Exception as e:
            logging.error("Errore durante il caricamento del checkpoint", exc_info=True)
            raise
