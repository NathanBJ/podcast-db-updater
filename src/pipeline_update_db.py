from pathlib import Path
import chromadb
import download_podcasts
import transcribe_podcast
import store_podcast
from datetime import datetime
import os

chromadb_path=Path(__file__).parent.parent / "podcast_db_local_fr"
RSS_FEED = os.getenv("RSS_FEED")
if not RSS_FEED:
    raise RuntimeError("RSS_FEED environment variable is required")
TEMP_FOLDER="mp3_downloads"
DB_PATH="podcast_db_local_fr"
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "podcast_collection")

def get_last_modified_timestamp(chromadb_path):
    """
    Returns the last published date from ChromaDB as a datetime object.
    Returns None if the database or timestamp doesn't exist.
    """
    try:
        client = chromadb.PersistentClient(path=chromadb_path)
        sys_collection = client.get_collection("system_metadata")
        
        result = sys_collection.get(ids=["last_published_date"])

        # Check if we actually got data back
        if result['metadatas'] and result['metadatas'][0]:
            date_str = result['metadatas'][0]['published_date']
            
            # Convert ISO String -> Datetime Object
            last_date_obj = datetime.fromisoformat(date_str)
            
            print(f"📅 Last Published Date: {last_date_obj}")
            return last_date_obj
            
    except Exception as e:
        # If collection doesn't exist or DB is empty
        default_date = datetime(2000, 1, 1)  # Very old date to get all episodes
        print(f"⚠️ Exception occurred: {e}")
        print(f"⚠️ No previous update found. Defaulting to {default_date.date()}.")
        return default_date

    # Fallback (shouldn't be reached)
    return datetime(2000, 1, 1)

if __name__ == "__main__":
    last_modified_date=get_last_modified_timestamp(chromadb_path)
    print(last_modified_date)
    print(f"⬇️  Downloading new episodes since {last_modified_date}...")
    download_podcasts.download_all_episode(
        rss_url=RSS_FEED,
        last_update_date=last_modified_date,
        output_folder=TEMP_FOLDER
    )
    print("📝 Transcribing downloaded episodes...")
    transcribe_podcast.transcribe_audio(
        target_folder=TEMP_FOLDER,
        model_size="base"
    )
    print("🗄️  Updating the podcast database...")
    store_podcast.store_all_transcripts(
        TEMP_FOLDER,
        collection_name=COLLECTION_NAME,
        db_path=DB_PATH
    )