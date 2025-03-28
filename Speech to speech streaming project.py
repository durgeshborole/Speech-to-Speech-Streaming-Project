import os
import streamlit as st
import tempfile
import subprocess
from pathlib import Path
import whisper
from deep_translator import GoogleTranslator
from gtts import gTTS
import yt_dlp as youtube_dl

# Load Whisper model (for transcribing audio)
model = whisper.load_model("base")

# Function to extract audio from video using ffmpeg
def extract_audio_from_video(video_path, audio_output_path):
    """Extracts audio from the uploaded video using ffmpeg."""
    command = f'ffmpeg -i "{video_path}" -q:a 0 -map a "{audio_output_path}" -y'
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    if result.returncode != 0:
        st.error("❌ Error extracting audio from video.")
        return None
    return audio_output_path

# Transcribe audio and get word timestamps
def transcribe_audio_with_timestamps(audio_path):
    """Transcribes audio and provides word-level timestamps."""
    try:
        result = model.transcribe(audio_path, word_timestamps=True)
        return result["text"], result["segments"]
    except Exception as e:
        st.error(f"❌ Error during transcription: {e}")
        return None, None

# Translate transcribed text to the target language
def translate_text(text, target_language):
    """Translates the transcribed text into the target language."""
    try:
        translated = GoogleTranslator(source="auto", target=target_language).translate(text)
        return translated
    except Exception as e:
        return f"❌ Translation failed: {e}"

# Generate speech from translated text using gTTS
def generate_speech_from_text(text, output_audio_path, language):
    """Generates speech from translated text and saves it as an audio file."""
    try:
        tts = gTTS(text=text, lang=language)
        tts.save(output_audio_path)
        return output_audio_path
    except Exception as e:
        st.error(f"❌ Error generating speech: {e}")
        return None

# Function to sync audio with video by adjusting speed (to match durations)
def sync_audio_with_video(video_path, audio_path, output_path):
    """Syncs the generated audio to the video while adjusting for duration mismatch."""
    try:
        # Get video duration
        video_duration = float(subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True, check=True
        ).stdout.strip())

        # Get audio duration
        audio_duration = float(subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
            capture_output=True, text=True, check=True
        ).stdout.strip())

        # Calculate speed adjustment factor to match video duration
        speed_factor = audio_duration / video_duration
        adjusted_audio_path = os.path.splitext(audio_path)[0] + "_adjusted.mp3"

        # Adjust audio speed to match video duration
        result = subprocess.run(
            ["ffmpeg", "-i", audio_path, "-filter:a", f"atempo={speed_factor}", "-vn", adjusted_audio_path, "-y"],
            check=True, capture_output=True, text=True
        )
        if result.returncode != 0:
            st.error(f"❌ Error adjusting audio speed: {result.stderr}")
            return None

        # Merge adjusted audio with video
        result = subprocess.run(
            ["ffmpeg", "-i", video_path, "-i", adjusted_audio_path, "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0", "-shortest", output_path, "-y"],
            capture_output=True, text=True
        )

        if result.returncode != 0:
            st.error(f"❌ Error merging video and audio: {result.stderr}")
            return None

        return output_path

    except subprocess.CalledProcessError as e:
        st.error(f"❌ Error in syncing audio and video: {e}")
        return None
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        return None

# Download YouTube video using yt-dlp
def download_video_from_url(url, temp_path):
    """Downloads video from YouTube URL and saves it in a temporary directory."""
    try:
        # Set options for downloading best available audio and video
        ydl_opts = {
            'format': 'bestaudio/best',  # Download best audio or video
            'noplaylist': True,  # Avoid downloading playlists
            'quiet': False,  # Display logs for debugging
            'outtmpl': str(temp_path / 'downloaded_video.%(ext)s'),  # Save video in temp_path
        }

        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=False)
            available_formats = info_dict.get('formats', [])
            
            if not available_formats:
                st.error("❌ No available formats found for this video.")
                return None

            # Log available formats
            st.write("Available Video Formats:")
            for format in available_formats:
                format_note = format.get('format_note', 'N/A')
                st.write(f"{format['format_id']}: {format_note} ({format['ext']})")

            # Download the best available format (mp4 preferred)
            best_video_format = "mp4"
            for format in available_formats:
                if 'mp4' in format['ext']:
                    best_video_format = format
                    break

            ydl_opts['format'] = best_video_format['format_id']
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                result = ydl.download([url])

                if result == 0:
                    return temp_path / f"downloaded_video.{best_video_format['ext']}"
                else:
                    st.error("❌ Error downloading the video.")
                    return None
    except Exception as e:
        st.error(f"❌ Error downloading video: {e}")
        return None

# Streamlit UI
st.set_page_config(page_title="AI Video Translator", page_icon="🎬", layout="wide")
st.title("🎬 AI Video Translator with Accurate Sync")

# Upload video or provide YouTube URL
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])
video_url = st.text_input("Or provide a YouTube video URL (optional)")

# Create temporary directory for storing files
temp_dir = tempfile.TemporaryDirectory()
temp_path = Path(temp_dir.name)

# Handle file upload or YouTube URL download
if video_url:
    st.write("Downloading video from URL...")
    input_video_path = download_video_from_url(video_url, temp_path)
    if input_video_path:
        st.success("✅ Video downloaded successfully!")
        st.video(str(input_video_path))
else:
    if uploaded_file:
        input_video_path = temp_path / "input_video.mp4"
        with open(input_video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("✅ Video uploaded successfully!")
        st.video(str(input_video_path))

# Select target language for translation
LANGUAGE_OPTIONS = {
    "English": "en", "Hindi": "hi", "Marathi": "mr", "Bengali": "bn", "Telugu": "te", 
    "Malayalam": "ml", "Kannada": "kn", "Spanish": "es", "French": "fr", "German": "de", 
    "Italian": "it", "Tamil": "ta"
}
target_language = st.selectbox("Choose the language you want to translate to:", list(LANGUAGE_OPTIONS.keys()))

# Convert button triggers the full process
if st.button("Convert Video"):
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Step 1: Extract Audio
    status_text.text("Step 1/5: Extracting audio from video...")
    extracted_audio_path = temp_path / "extracted_audio.wav"
    extracted_audio = extract_audio_from_video(str(input_video_path), str(extracted_audio_path))
    if not extracted_audio:
        st.error("Failed to extract audio.")
        temp_dir.cleanup()
        st.stop()
    progress_bar.progress(20)

    # Step 2: Transcribe Audio with Timestamps
    status_text.text("Step 2/5: Transcribing audio...")
    transcript, segments = transcribe_audio_with_timestamps(str(extracted_audio))
    if not transcript:
        st.error("Failed to transcribe audio.")
        temp_dir.cleanup()
        st.stop()
    progress_bar.progress(40)

    st.text_area("Original Transcript", transcript, height=100)

    # Step 3: Translate Text
    status_text.text(f"Step 3/5: Translating text to {target_language}...")
    translated_text = translate_text(transcript, LANGUAGE_OPTIONS[target_language])
    if not translated_text:
        st.error("Translation failed.")
        temp_dir.cleanup()
        st.stop()
    progress_bar.progress(60)

    st.text_area(f"Translated Text ({target_language})", translated_text, height=100)

    # Step 4: Generate Speech from Translated Text
    status_text.text("Step 4/5: Generating translated speech...")
    cloned_audio_path = temp_path / f"translated_audio_{target_language.lower()}.mp3"
    cloned_audio = generate_speech_from_text(translated_text, str(cloned_audio_path), LANGUAGE_OPTIONS[target_language])
    if not cloned_audio:
        st.error("Failed to generate translated speech.")
        temp_dir.cleanup()
        st.stop()
    progress_bar.progress(80)

    # Step 5: Sync Audio with Video
    status_text.text("Step 5/5: Synchronizing audio with video...")
    output_video_path = temp_path / f"output_video_{target_language.lower()}.mp4"
    final_video = sync_audio_with_video(str(input_video_path), str(cloned_audio), str(output_video_path))
    if not final_video:
        st.error("Failed to sync audio with video.")
        temp_dir.cleanup()
        st.stop()
    progress_bar.progress(100)

    st.success("✅ Conversion complete! The translated audio is now synced with the video.")
    st.subheader("Converted Video")
    st.video(str(output_video_path))

    with open(output_video_path, "rb") as file:
        st.download_button("Download Final Video", file, file_name=f"translated_video_{target_language.lower()}.mp4", mime="video/mp4")

# Cleanup temporary directory
temp_dir.cleanup()
