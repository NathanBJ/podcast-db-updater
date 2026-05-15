import feedparser
import requests
import sys
import json
import os
from tqdm import tqdm
import concurrent.futures
import multiprocessing
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from datetime import datetime
import time

def get_spotify_url(episode_title, podcast_title, show_name=None):
    """
    Recherche un épisode sur Spotify et retourne son lien URL.
    
    Args:
        episode_title (str): Le titre de l'épisode (ex: "Gérer la colère").
        show_name (str, optional): Le nom du podcast pour filtrer (ex: "Éducation Positive").
    
    Returns:
        str: L'URL Spotify (ex: 'https://open.spotify.com/episode/...') ou None si non trouvé.
    """
    # Configuration de l'authentification (idéalement via variables d'environnement)
    # Vous pouvez aussi mettre les clés en dur pour tester, mais ne les partagez pas.
    CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID")
    CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET")

    # 1. Connexion à l'API (Client Credentials Flow - pas besoin de login utilisateur)
    auth_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
    sp = spotipy.Spotify(auth_manager=auth_manager)

    # 2. Construction de la requête de recherche
    # La syntaxe Spotify permet de filtrer par show avec "show:NomDuPodcast"
    query = f'"{episode_title}"' # Les guillemets aident à chercher la phrase exacte
    if show_name:
        query += f" show:{show_name}"

    print(f"🔍 Recherche Spotify pour : {query} ...")

    try:
        # 3. Exécution de la recherche (type='episode')
        results = sp.search(q=query, type='episode', limit=1, market='FR')
        
        items = results['episodes']['items']
        
        if items:
            # On récupère le premier résultat
            found_url = items[0]['external_urls']['spotify']
            found_name = items[0]['name']
            print(f"   ✅ Trouvé : {found_name}")
            return found_url
        else:
            print("   ❌ Aucun résultat trouvé sur Spotify.")
            return None

    except Exception as e:
        print(f"   ⚠️ Erreur API Spotify : {e}")
        return None

def download_episode(episode, headers, podcast_title, output_folder="mp3_downloads"):
    title = episode.title
    print(f"🎙️  Found: {title}")

    # 2. Find Audio Link
    audio_url = None
    for link in episode.links:
        if link['type'].startswith('audio'):
            audio_url = link['href']
            break
    
    if not audio_url:
        print("❌ No audio link found in this episode.")
        return

    # 3. Request the File (Stream Mode)
    print(f"🔄 Connecting to: {audio_url}")
    try:
        # allow_redirects=True is crucial for podcast tracking links (e.g. Podtrac)
        response = requests.get(audio_url, stream=True, headers=headers, allow_redirects=True)
        response.raise_for_status() # Stop if we get a 404 or 403 error directly

        # 4. Verify Content Type (Crucial Check)
        content_type = response.headers.get('Content-Type', '').lower()
        if 'text/html' in content_type:
            print("❌ ERROR: The server returned a webpage (HTML) instead of audio.")
            print("   This usually means they are blocking bots or the link is expired.")
            print("   Snippet of response:", response.text[:200]) # Print first 200 chars to debug
            return
        
        # 5. Download
        total_size = int(response.headers.get('content-length', 0))
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
        safe_title = "".join([c for c in title if c.isalpha() or c.isdigit() or c==' ']).rstrip()
        mp3_filename = os.path.join(output_folder, f"{safe_title}.mp3")
        json_filename = os.path.join(output_folder, f"{safe_title}.json")

        print(f"⬇️  Downloading ({total_size / (1024*1024):.2f} MB)...")
        
        with open(mp3_filename, 'wb') as file, tqdm(
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
        spotify_url=get_spotify_url(title, podcast_title)
        
        # Extract publication date
        published_date = None
        if hasattr(episode, 'published_parsed') and episode.published_parsed:
            published_date = datetime.fromtimestamp(time.mktime(episode.published_parsed)).isoformat()
        
        metadata = {
            "title": title,
            "podcast": podcast_title,
            "audio_url": audio_url,
            "spotify_url": spotify_url,
            "published_date": published_date
        }
        with open(json_filename, 'w', encoding='utf-8') as json_file:
            json.dump(metadata, json_file, ensure_ascii=False, indent=4)

        print(f"\n✅ Successfully saved: {mp3_filename} and {json_filename}")
        return mp3_filename, json_filename

    except Exception as e:
        print(f"❌ Download Failed: {e}")
        return None

def download_all_episode(rss_url, last_update_date, output_folder="mp3_downloads"):
    cpu_count = multiprocessing.cpu_count()
    # 1. Setup Request Headers (The "Disguise")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    print(f"📡 Parsing feed: {rss_url}...")
    # feedparser handles headers internally usually, but sometimes we might need to pass agent there too
    feed = feedparser.parse(rss_url, agent=headers["User-Agent"])

    if not feed.entries:
        print("❌ No episodes found. The feed might be blocked or empty.")
        return

    podcast_title = feed.feed.title
    episodes_to_download =[]
    #filter already downloaded episodes
    for episode in feed.entries:
        if hasattr(episode, 'published_parsed') and episode.published_parsed:
            # Convert struct_time to a datetime object
            published_date = datetime.fromtimestamp(time.mktime(episode.published_parsed))
            # 3. The Comparison
            if published_date > last_update_date:
          
                episodes_to_download.append(episode)
    
    # Sort by oldest first and limit to MAX_EPISODES_PER_RUN to avoid exceeding API rate limits
    episodes_to_download.sort(key=lambda ep: ep.published_parsed if hasattr(ep, 'published_parsed') else time.gmtime(0))
    max_episodes = int(os.getenv("MAX_DOWNLOADED_EPISODES_PER_RUN", 5))
    episodes_to_download = episodes_to_download[:max_episodes]

    # Start the processing of each episode with different threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_count) as executor:
        # Start the load operations and mark each future with its log_name
        future_to_downloaded_episode = {executor.submit(download_episode, episode, headers, podcast_title, output_folder): episode for episode in episodes_to_download}
        for future in concurrent.futures.as_completed(future_to_downloaded_episode):
            log = future_to_downloaded_episode[future]
            try:
                data = future.result()
            except Exception as exc:
                print('%r generated an exception: %s' % (log, exc))
                
            else:
                print('log %r has succeed' % (log))

def download_latest_episode(rss_url, output_folder="downloads"):
    # 1. Setup Request Headers (The "Disguise")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    print(f"📡 Parsing feed: {rss_url}...")
    # feedparser handles headers internally usually, but sometimes we might need to pass agent there too
    feed = feedparser.parse(rss_url, agent=headers["User-Agent"])

    if not feed.entries:
        print("❌ No episodes found. The feed might be blocked or empty.")
        return

    latest_episode = feed.entries[0]
    title = latest_episode.title
    print(f"🎙️  Found: {title}")

    # 2. Find Audio Link
    audio_url = None
    for link in latest_episode.links:
        if link['type'].startswith('audio'):
            audio_url = link['href']
            break
    
    if not audio_url:
        print("❌ No audio link found in this episode.")
        return

    # 3. Request the File (Stream Mode)
    print(f"🔄 Connecting to: {audio_url}")
    try:
        # allow_redirects=True is crucial for podcast tracking links (e.g. Podtrac)
        response = requests.get(audio_url, stream=True, headers=headers, allow_redirects=True)
        response.raise_for_status() # Stop if we get a 404 or 403 error directly

        # 4. Verify Content Type (Crucial Check)
        content_type = response.headers.get('Content-Type', '').lower()
        if 'text/html' in content_type:
            print("❌ ERROR: The server returned a webpage (HTML) instead of audio.")
            print("   This usually means they are blocking bots or the link is expired.")
            print("   Snippet of response:", response.text[:200]) # Print first 200 chars to debug
            return
        
        # 5. Download
        total_size = int(response.headers.get('content-length', 0))
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
        safe_title = "".join([c for c in title if c.isalpha() or c.isdigit() or c==' ']).rstrip()
        filename = os.path.join(output_folder, f"{safe_title}.mp3")

        print(f"⬇️  Downloading ({total_size / (1024*1024):.2f} MB)...")
        
        with open(filename, 'wb') as file, tqdm(
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)

        print(f"\n✅ Successfully saved: {filename}")
        return filename

    except Exception as e:
        print(f"❌ Download Failed: {e}")
        return None

if __name__ == "__main__":
    RSS_FEED = os.getenv("RSS_FEED")
    if not RSS_FEED:
        raise RuntimeError("RSS_FEED environment variable is required")
    last_update_date = datetime(2023, 6, 1)

    download_all_episode(RSS_FEED, last_update_date, output_folder="mp3_downloads")