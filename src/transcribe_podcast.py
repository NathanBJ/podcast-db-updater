import os
import sys
import glob
from groq_transcription import get_transcriber

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

    print("🔍 Searching for MP3 files...")
    sys.stdout.flush()
    mp3_files = glob.glob(os.path.join(target_folder, "*.mp3"))
    print(f"Found {len(mp3_files)} audio files to transcribe.")
    sys.stdout.flush()
    
    if len(mp3_files) == 0:
        print("⚠️ No MP3 files found. Exiting.")
        return
    transcriber = get_transcriber()
    for file_path in mp3_files:
        print(f"🎧 Found audio file: {file_path}")
        sys.stdout.flush()
        transcribe_one_by_one_audio(transcriber, file_path, fp16=False)
    
    
def transcribe_one_by_one_audio(transcriber, file_path, fp16=False):
    # The 'transcribe' function handles chunking and processing automatically
    # fp16=False is safer for CPU usage to avoid warnings
    print(f"🎙️  Transcribing '{file_path}'...")
    outputfile = file_path.replace(".mp3", ".txt")
    result = transcriber.transcribe_file(file_path, outputfile)
    
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

    
