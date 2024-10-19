from googlesearch import search
import logging
logging.basicConfig(filename='agent.log', level=logging.INFO)


class GoogleSearchTool:
    def __init__(self):
        pass

    def google_search(self, query: str, num_results: int = 5):
        """
        Esegue una ricerca su Google e restituisce i risultati.

        Args:
        query (str): La query di ricerca.
        num_results (int): Il numero di risultati da restituire.

        Returns:
        List[str]: Una lista di URL risultanti dalla ricerca.
        """
        results = []
        try:
            results = list(search(query, num_results=num_results, advanced=True))
        except Exception as e:
            logging.info(f"Errore durante la ricerca: {e}")
        return results
