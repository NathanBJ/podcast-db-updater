# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "chromadb",
#     "mcp[cli]",
#     "sentence-transformers",
# ]
# ///

import os
# 1. Prevent the tokenizer from forking multiple threads (fixes the hang)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 2. Limit the math library to one thread per core (prevents CPU choking)
os.environ["OMP_NUM_THREADS"] = "1"
import chromadb
from chromadb.utils import embedding_functions
import logging
from fastmcp import FastMCP
from pathlib import Path

# Configure logging to show dice roll results
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class podcastKnowledgeBase:
    """Fast knowledge base interface"""
    
    def __init__(self):
        # Use the shared KB at the chapter root level
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up to chapter root: rules_agent -> agents -> 5_a2a_integration
        chapter_root = os.path.dirname(os.path.dirname(current_dir))
        self.db_path = Path(__file__).parent.parent.parent / "podcast_db_local_fr"
        self._client = None
        self._collection = None
        self.MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        self.collection_name = "BKHK_podcast"
        self.local_ef = self.get_local_ef()
        logging.info(f"KB path: {self.db_path}")
    
    def _get_collection(self, name: str, embedding_function):
        if self._collection is None:
            try:
                logging.info(f"Attempting to connect to ChromaDB at: {self.db_path}")
                logging.info(f"Path exists: {os.path.exists(self.db_path)}")
                self._client = chromadb.PersistentClient(path=self.db_path)
                logging.info("ChromaDB client created successfully")
                
                # List available collections
                collections = self._client.list_collections()
                logging.info(f"Available collections: {[c.name for c in collections]}")
                
                self._collection = self._client.get_collection(name=name, embedding_function=embedding_function)
                logging.info("Collection 'BKHK_podcast' found successfully")
            except Exception as e:
                logging.info(f"Error connecting to KB: {e}")
                return None
        return self._collection
    
    def get_local_ef(self):
        """Sets up the local Hugging Face embedding function"""
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.MODEL_NAME
        )

    def quick_query(self, query: str) -> str:
        """Fast query with minimal processing"""
        logging.info(f"Querying KB with: {query}")
        collection = self._get_collection(name=self.collection_name,embedding_function=self.local_ef)
        if not collection:
            logging.info("Collection is None - KB unavailable")
            return "KB unavailable"
        
        try:
            results = collection.query(query_texts=[query], n_results=5)
            if results['documents'][0]:
                sources = []
                docs=[]
                for i, doc in enumerate(results['documents'][0]):
                    meta = results['metadatas'][0][i] if results['metadatas'] else {}
                    source = meta.get('source_file', 'Unknown')
                    sources.append(f"{source}")
                    docs.append(doc.replace("\n", " "))
                # Very short response
                return f"Sources {sources}: {docs}..."
            return "No advice found"
        except:
            return "KB error"

podcast_kb = podcastKnowledgeBase()

# TODO: Create an MCP server with name "D&D Dice Roll Service" on port 8080
mcp = FastMCP(
    name="BKHK teacher Service",
    port=8080
)


@mcp.tool
def query_podcast_advices(query: str) -> str:
    """Specialized childhood and parenthood agent that provides clear explainations and advices for the parents to understand their child and their associated problems."""
    return podcast_kb.quick_query(query)


# Start the MCP server
if __name__ == "__main__":
    logging.info("Starting BKHK teacher Server...")
    # TODO: run the MCP server
    mcp.run(transport="streamable-http")