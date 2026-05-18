import chromadb
import os
import glob
import json
import re
import sys
import datetime
from hf_embeddings import get_embedding_function

def chunk_text(text, chunk_size=1000, overlap=100):
    # Same chunking logic as before
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        if current_length + len(sentence) > chunk_size and current_chunk:
            full_chunk = " ".join(current_chunk)
            chunks.append(full_chunk)
            
            overlap_text = ""
            overlap_len = 0
            new_start = []
            for s in reversed(current_chunk):
                if overlap_len + len(s) < overlap:
                    new_start.insert(0, s)
                    overlap_len += len(s)
                else:
                    break
            current_chunk = new_start
            current_length = overlap_len

        current_chunk.append(sentence)
        current_length += len(sentence)

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def store_all_transcripts(folder_path, collection_name="BKHK_podcast", db_path="podcast_db_local_fr"):
    # Create a NEW folder for this specific model's database
    print(f"Loading chromadb at path {db_path}")
    print(f"Contents of {db_path}: {os.listdir(db_path) if os.path.exists(db_path) else 'Path does not exist'}")
    #time.sleep(100000)
    client = chromadb.PersistentClient(path=db_path)
    
    print(f"⏳ Getting embedding function...")
    embedding_fn = get_embedding_function()
    
    collection = client.get_or_create_collection(
        name=collection_name, 
        embedding_function=embedding_fn
    )
    os.makedirs(folder_path, exist_ok=True)
    if not os.path.exists(folder_path):
        print(f"❌ Error: Folder '{folder_path}' not found.")
        return

    txt_files = glob.glob(os.path.join(folder_path, "*.txt"))
    print(f"📂 Found {len(txt_files)} files. Processing locally...")

    for file_path in txt_files:
        filename = os.path.basename(file_path)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                full_text = f.read()

            if not full_text.strip(): continue

            chunks = chunk_text(full_text)
            json_filepath = file_path.replace(".txt", ".json")
            with open(json_filepath) as f:
                data = json.load(f)

            # Create IDs
            clean_name = filename.replace(" ", "_").replace(".txt", "")
            ids = [f"{clean_name}_{i}" for i in range(len(chunks))]
            metadatas = [data for _ in range(len(chunks))]

            collection.upsert(
                documents=chunks,
                ids=ids,
                metadatas=metadatas
            )
            #TODO: retrieve all "published_date" from the metadata and store the most recent one in the system_metadata collection to track when the database was last updated with new episodes
            if "published_date" in data and data["published_date"]:
                # Store the most recent published_date in system_metadata
                sys_collection = client.get_or_create_collection("system_metadata")
                existing = sys_collection.get(ids=["last_published_date"])
                existing_date = existing['metadatas'][0]['published_date'] if existing['metadatas'] else None
                if not existing_date or data["published_date"] > existing_date:
                    sys_collection.upsert(
                        ids=["last_published_date"],
                        documents=[f"Last published date: {data['published_date']}"],
                        metadatas=[{
                            "published_date": data["published_date"]
                        }]
                    )
                    print(f"   📅 Updated last published date to {data['published_date']}")
            print(f"   ✅ {filename}: {len(chunks)} chunks stored.")

        except Exception as e:
            print(f"   ❌ Error on {filename}: {e}")
    
    print("✅ All transcripts processed.")

def query_database(query_text, collection_name="BKHK_podcast"):
    client = chromadb.PersistentClient(path="podcast_db_local_fr")
    
    # Must use the SAME model for querying
    local_ef = get_embedding_function()
    collection = client.get_collection(name=collection_name, embedding_function=local_ef)
    
    print(f"\n🔍 Searching for: '{query_text}'")
    
    results = collection.query(
        query_texts=[query_text],
        n_results=3
    )
    
    if not results['documents'] or not results['documents'][0]:
        print("❌ No results found.")
        return

    for i, doc in enumerate(results['documents'][0]):
        meta = results['metadatas'][0][i] if results['metadatas'] else {}
        source = meta.get('source_file', 'Unknown')
        print(f"\n--- Result {i+1} (Source: {source}) ---")
        print(doc)

if __name__ == "__main__":
    target_folder = sys.argv[1] if len(sys.argv) > 1 else "mp3_downloads"
    
    store_all_transcripts(target_folder)
    query_database("Comment gérer une crise en public ?")