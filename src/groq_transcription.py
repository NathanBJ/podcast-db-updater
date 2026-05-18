# =============================================================================
# groq_transcription.py - Groq Whisper API for Speech-to-Text
# =============================================================================
# Uses Groq's FREE Whisper API for high-quality French transcription.
#
# Free Tier Limits:
# - 20 hours of audio per month
# - Whisper large-v3 model (best quality)
# - Very fast inference (~10x real-time)
#
# Get your free API key at: https://console.groq.com/keys
# =============================================================================

import os
import time
import subprocess
import tempfile
from pathlib import Path
from typing import List

from groq import Groq


class AudioSplitter:
    """
    Split large audio files into smaller chunks using ffmpeg.
    """
    
    def __init__(self, chunk_duration_seconds: int = 600):
        """
        Initialize the audio splitter.
        
        Args:
            chunk_duration_seconds: Duration of each chunk in seconds (default: 10 minutes)
        """
        self.chunk_duration = chunk_duration_seconds
        self._check_ffmpeg()
    
    def _check_ffmpeg(self):
        """Check if ffmpeg is available."""
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                check=True
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError(
                "ffmpeg not found. Install it with:\n"
                "  Ubuntu/Debian: apt-get install ffmpeg\n"
                "  macOS: brew install ffmpeg\n"
                "  Alpine: apk add ffmpeg"
            )
    
    def get_duration(self, audio_path: str) -> float:
        """Get the duration of an audio file in seconds."""
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(audio_path)
            ],
            capture_output=True,
            text=True,
            check=True
        )
        return float(result.stdout.strip())
    
    def get_file_size_mb(self, audio_path: str) -> float:
        """Get file size in MB."""
        return Path(audio_path).stat().st_size / (1024 * 1024)
    
    def needs_splitting(self, audio_path: str, max_size_mb: float = 24) -> bool:
        """Check if a file needs to be split."""
        return self.get_file_size_mb(audio_path) > max_size_mb
    
    def split_audio(
        self,
        audio_path: str,
        output_dir: str = None,
        chunk_duration: int = None
    ) -> List[str]:
        """
        Split an audio file into smaller chunks.
        
        Args:
            audio_path: Path to the audio file
            output_dir: Directory for output chunks (default: temp dir)
            chunk_duration: Duration per chunk in seconds (default: self.chunk_duration)
            
        Returns:
            List of paths to the chunk files
        """
        audio_path = Path(audio_path)
        chunk_duration = chunk_duration or self.chunk_duration
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = Path(tempfile.mkdtemp(prefix="audio_chunks_"))
        
        # Get audio duration
        total_duration = self.get_duration(audio_path)
        num_chunks = int(total_duration // chunk_duration) + 1
        
        print(f"🔪 Splitting {audio_path.name} ({total_duration:.0f}s) into {num_chunks} chunks...")
        
        chunk_paths = []
        
        for i in range(num_chunks):
            start_time = i * chunk_duration
            chunk_name = f"{audio_path.stem}_chunk{i:03d}{audio_path.suffix}"
            chunk_path = output_dir / chunk_name
            
            # Use ffmpeg to extract chunk
            cmd = [
                "ffmpeg",
                "-y",  # Overwrite
                "-i", str(audio_path),
                "-ss", str(start_time),
                "-t", str(chunk_duration),
                "-c", "copy",  # Fast copy without re-encoding
                str(chunk_path)
            ]
            
            subprocess.run(cmd, capture_output=True, check=True)
            
            if chunk_path.exists() and chunk_path.stat().st_size > 0:
                chunk_paths.append(str(chunk_path))
                print(f"   ✓ Created chunk {i+1}/{num_chunks}: {chunk_name}")
        
        return chunk_paths
    
    def cleanup_chunks(self, chunk_paths: List[str]):
        """Remove temporary chunk files."""
        for path in chunk_paths:
            try:
                Path(path).unlink()
            except:
                pass


class GroqTranscriber:
    """
    Transcribe audio files using Groq's free Whisper API.
    Automatically splits large files that exceed the 25MB limit.
    """
    
    # Supported audio formats
    SUPPORTED_FORMATS = {'.mp3', '.mp4', '.mpeg', '.mpga', '.m4a', '.wav', '.webm', '.ogg', '.flac'}
    
    # Max file size: 25MB
    MAX_FILE_SIZE = 25 * 1024 * 1024
    MAX_FILE_SIZE_MB = 24  # Use 24MB to be safe
    RPM_LIMIT = 20  # requests per minute 
    RPD_LIMIT = 2000  # requests per day
    ASH_LIMIT = 7200  # audio seconds per hour
    ASD_LIMIT = 28800  # audio seconds per day

    def __init__(
        self,
        api_key: str = None,
        model: str = "whisper-large-v3",
        language: str = "fr",
        max_retries: int = 3,
        auto_split: bool = True,
        chunk_duration: int = 600  # 10 minutes
    ):
        """
        Initialize the Groq transcriber.
        
        Args:
            api_key: Groq API key (or set GROQ_API_KEY env var)
            model: Whisper model to use (whisper-large-v3 is best)
            language: Language code (fr for French)
            max_retries: Number of retries on failure
            auto_split: Automatically split large files
            chunk_duration: Duration per chunk in seconds when splitting
        """
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.model = model
        self.language = language
        self.max_retries = max_retries
        self.auto_split = auto_split

        if not self.api_key:
            raise ValueError(
                "Groq API key required. "
                "Set GROQ_API_KEY environment variable or pass api_key parameter. "
                "Get a free key at: https://console.groq.com/keys"
            )
        
        self.client = Groq(api_key=self.api_key)
        
        # Initialize splitter if auto_split enabled
        if self.auto_split:
            try:
                self.splitter = AudioSplitter(chunk_duration_seconds=chunk_duration)
            except RuntimeError as e:
                print(f"⚠️ Audio splitting disabled: {e}")
                self.splitter = None
        else:
            self.splitter = None
    
    def transcribe_file(
        self,
        audio_path: str,
        output_path: str = None,
        prompt: str = None,
        temperature: float = 0.0
    ) -> dict:
        """
        Transcribe an audio file to text.
        Automatically splits large files if they exceed 25MB.
        
        Args:
            audio_path: Path to the audio file
            output_path: Optional path to save transcription text
            prompt: Optional prompt to guide transcription style
            temperature: Sampling temperature (0.0 = deterministic)
            
        Returns:
            dict with 'text', 'duration', 'language', etc.
        """
        audio_path = Path(audio_path)
        
        # Validate file
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        suffix = audio_path.suffix.lower()
        if suffix not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format: {suffix}. "
                f"Supported: {', '.join(self.SUPPORTED_FORMATS)}"
            )
        
        file_size = audio_path.stat().st_size
        file_size_mb = file_size / (1024 * 1024)
        
        # Check if file needs splitting
        if file_size > self.MAX_FILE_SIZE:
            if self.splitter and self.auto_split:
                return self._transcribe_large_file(audio_path, output_path, prompt, temperature)
            else:
                raise ValueError(
                    f"File too large: {file_size_mb:.1f}MB. "
                    f"Max size: {self.MAX_FILE_SIZE / 1024 / 1024:.0f}MB. "
                    "Enable auto_split or manually split the file."
                )
        
        return self._transcribe_single_file(audio_path, output_path, prompt, temperature)
    
    def _transcribe_large_file(
        self,
        audio_path: Path,
        output_path: str = None,
        prompt: str = None,
        temperature: float = 0.0
    ) -> dict:
        """Transcribe a large file by splitting it into chunks."""
        file_size_mb = audio_path.stat().st_size / (1024 * 1024)
        print(f"📝 Large file detected: {audio_path.name} ({file_size_mb:.1f}MB)")
        print(f"   Splitting into chunks...")
        
        # Split the file
        chunk_paths = self.splitter.split_audio(str(audio_path))
        
        if not chunk_paths:
            raise RuntimeError("Failed to split audio file")
        
        # Transcribe each chunk
        all_text = []
        all_segments = []
        total_duration = 0
        
        try:
            for i, chunk_path in enumerate(chunk_paths):
                print(f"\n   📝 Transcribing chunk {i+1}/{len(chunk_paths)}...")
                
                result = self._transcribe_single_file(
                    Path(chunk_path),
                    output_path=None,  # Don't save individual chunks
                    prompt=prompt,
                    temperature=temperature
                )
                
                all_text.append(result["text"])
                
                # Adjust segment timestamps
                if result.get("segments"):
                    for seg in result["segments"]:
                        seg["start"] = seg.get("start", 0) + total_duration
                        seg["end"] = seg.get("end", 0) + total_duration
                    all_segments.extend(result["segments"])
                
                if result.get("duration"):
                    total_duration += result["duration"]
        
        finally:
            # Cleanup chunk files
            print(f"\n   🧹 Cleaning up temporary chunks...")
            self.splitter.cleanup_chunks(chunk_paths)
        
        # Combine results
        combined_text = " ".join(all_text)
        
        result = {
            "text": combined_text,
            "language": self.language,
            "duration": total_duration,
            "segments": all_segments,
            "file": str(audio_path),
            "chunks": len(chunk_paths)
        }
        
        # Save combined output
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(combined_text, encoding="utf-8")
            print(f"\n   ✓ Saved to: {output_path}")
        
        print(f"\n   ✓ Transcribed {total_duration:.0f}s of audio from {len(chunk_paths)} chunks")
        return result
    
    def _transcribe_single_file(
        self,
        audio_path: Path,
        output_path: str = None,
        prompt: str = None,
        temperature: float = 0.0
    ) -> dict:
        """Transcribe a single audio file (must be under 25MB)."""
        file_size_mb = audio_path.stat().st_size / (1024 * 1024)
        print(f"📝 Transcribing: {audio_path.name} ({file_size_mb:.1f}MB)")
        
        # Transcribe with retries
        for attempt in range(self.max_retries):
            try:
                with open(audio_path, "rb") as audio_file:
                    response = self.client.audio.transcriptions.create(
                        file=(audio_path.name, audio_file),
                        model=self.model,
                        language=self.language,
                        prompt=prompt,
                        temperature=temperature,
                        response_format="verbose_json"
                    )
                
                result = {
                    "text": response.text,
                    "language": getattr(response, 'language', self.language),
                    "duration": getattr(response, 'duration', None),
                    "segments": getattr(response, 'segments', []),
                    "file": str(audio_path)
                }
                
                # Save to file if requested
                if output_path:
                    output_path = Path(output_path)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_text(result["text"], encoding="utf-8")
                    print(f"   ✓ Saved to: {output_path}")
                
                print(f"   ✓ Transcribed {result.get('duration', '?')}s of audio")
                return result
                
            except Exception as e:
                error_str = str(e).lower()
                
                if "rate" in error_str or "429" in error_str:
                    wait_time = 60 * (attempt + 1)
                    print(f"   ⚠️ Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                
                elif attempt < self.max_retries - 1:
                    wait_time = 5 * (attempt + 1)
                    print(f"   ⚠️ Error: {e}, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise RuntimeError(f"Failed to transcribe: {e}")
        
        raise RuntimeError(f"Failed after {self.max_retries} attempts")
    
    def transcribe_directory(
        self,
        input_dir: str,
        output_dir: str = None,
        recursive: bool = False
    ) -> list:
        """
        Transcribe all audio files in a directory.
        
        Args:
            input_dir: Directory containing audio files
            output_dir: Directory to save transcriptions (optional)
            recursive: Whether to search subdirectories
            
        Returns:
            List of transcription results
        """
        input_dir = Path(input_dir)
        
        if not input_dir.exists():
            raise FileNotFoundError(f"Directory not found: {input_dir}")
        
        # Find audio files
        pattern = "**/*" if recursive else "*"
        audio_files = [
            f for f in input_dir.glob(pattern)
            if f.suffix.lower() in self.SUPPORTED_FORMATS
        ]
        
        if not audio_files:
            print(f"No audio files found in {input_dir}")
            return []
        
        print(f"📂 Found {len(audio_files)} audio files")
        
        results = []
        for i, audio_file in enumerate(audio_files, 1):
            print(f"\n[{i}/{len(audio_files)}]")
            
            try:
                # Determine output path
                if output_dir:
                    relative = audio_file.relative_to(input_dir)
                    out_path = Path(output_dir) / relative.with_suffix(".txt")
                else:
                    out_path = None
                
                result = self.transcribe_file(audio_file, out_path)
                results.append(result)
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                results.append({"file": str(audio_file), "error": str(e)})
        
        # Summary
        success = sum(1 for r in results if "text" in r)
        print(f"\n✅ Completed: {success}/{len(audio_files)} files")
        
        return results


def get_transcriber(language: str = "fr") -> GroqTranscriber:
    """Factory function to create transcriber."""
    return GroqTranscriber(language=language)


# =============================================================================
# CLI Interface
# =============================================================================

if __name__ == "__main__":
    import sys
    import json
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python groq_transcription.py <audio_file>")
        print("  python groq_transcription.py <audio_file> <output.txt>")
        print("  python groq_transcription.py --dir <input_dir> [output_dir]")
        sys.exit(1)
    
    transcriber = get_transcriber()
    
    if sys.argv[1] == "--dir":
        input_dir = sys.argv[2] if len(sys.argv) > 2 else "."
        output_dir = sys.argv[3] if len(sys.argv) > 3 else None
        results = transcriber.transcribe_directory(input_dir, output_dir)
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        audio_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        result = transcriber.transcribe_file(audio_file, output_file)
        print(f"\n--- Transcription ---\n{result['text']}")