import os
# Prevent tokenizer parallelism issues in containers
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch
# Force PyTorch to use single thread
torch.set_num_threads(1)

import whisper
import sys
import warnings
import concurrent.futures
import multiprocessing
import glob

# Filter out some noisy warnings from PyTorch/Whisper
warnings.filterwarnings("ignore")

def transcribe_audio(target_folder, model_size="base"):
    """
    Transcribes an audio file using OpenAI Whisper.
    
    Args:
        file_path (str): Path to the .mp3 file.
        model_size (str): Model size ('tiny', 'base', 'small', 'medium', 'large').
                          'base' is a good balance of speed vs accuracy.
    """
    if not os.path.exists(target_folder):
        print(f"Warning: Folder not found - {target_folder}")
        print("creating the folder...")
        os.makedirs(target_folder)
        return None

    print(f"⏳ Loading Whisper model ('{model_size}')... this may take a moment.")
    # The first run will download the model weights (~140MB for base)
    
    # Retry logic for corrupted downloads
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print("📥 Downloading/loading model...")
            print("🔧 Initializing Whisper...")
            sys.stdout.flush()  # Force output to appear immediately
            model = whisper.load_model(model_size, device="cpu")
            print("✅ Model loaded successfully!")
            sys.stdout.flush()
            break
        except RuntimeError as e:
            if "SHA256 checksum" in str(e) and attempt < max_retries - 1:
                print(f"⚠️  Model download corrupted (attempt {attempt + 1}/{max_retries})")
                print("🗑️  Clearing cache and retrying...")
                # Clear the corrupted model cache
                cache_dir = os.path.expanduser("~/.cache/whisper")
                if os.path.exists(cache_dir):
                    import shutil
                    shutil.rmtree(cache_dir)
                    print("✅ Cache cleared")
            else:
                raise
    
    print("🔍 Searching for MP3 files...")
    sys.stdout.flush()
    mp3_files = glob.glob(os.path.join(target_folder, "*.mp3"))
    print(f"Found {len(mp3_files)} audio files to transcribe.")
    sys.stdout.flush()
    
    if len(mp3_files) == 0:
        print("⚠️ No MP3 files found. Exiting.")
        return
    
    for file_path in mp3_files:
        print(f"🎧 Found audio file: {file_path}")
        sys.stdout.flush()
        transcribe_one_by_one_audio(model, file_path, fp16=False)
    '''
    cpu_count = multiprocessing.cpu_count()

    # Start the processing of each episode with different threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_count) as executor:
        # Start the load operations and mark each future with its log_name
        future_to_downloaded_episode = {executor.submit(transcribe_one_by_one_audio, model, episode): episode for episode in mp3_files}
        for future in concurrent.futures.as_completed(future_to_downloaded_episode):
            log = future_to_downloaded_episode[future]
            try:
                data = future.result()
            except Exception as exc:
                print('%r generated an exception: %s' % (log, exc))
                
            else:
                print('log %r has succeed' % (log))

    '''
    
    
def transcribe_one_by_one_audio(model, file_path, fp16=False):
    # The 'transcribe' function handles chunking and processing automatically
    # fp16=False is safer for CPU usage to avoid warnings
    print(f"🎙️  Transcribing '{file_path}'...")
    print("   (This can take 1-10 minutes depending on your CPU/GPU and audio length)")
    result = model.transcribe(file_path, fp16=False)
    
    # Save to file
    output_txt = file_path.replace(".mp3", ".txt")
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(result["text"])

if __name__ == "__main__":
    # 1. Get the filename from command line OR use a default
    if len(sys.argv) > 1:
        target_folder = sys.argv[1]
    else:
        # REPLACE THIS with the actual file you downloaded in the previous step
        target_folder = "mp3_downloads"

    transcribe_audio(target_folder, model_size="base")

    
