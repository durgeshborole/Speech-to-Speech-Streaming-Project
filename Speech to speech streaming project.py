import os
import pytube
import ffmpeg
import whisper
from googletrans import Translator
from gtts import gTTS
from pydub import AudioSegment
import subprocess
import json
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from resemblyzer import VoiceEncoder, preprocess_wav
import pickle
from scipy.spatial.distance import cosine

# Function to download both video and audio from URL using pytube
def download_video(url, output_path):
    try:
        yt = pytube.YouTube(url)
        # Download highest resolution video
        video_stream = yt.streams.get_highest_resolution()
        video_file = os.path.join(output_path, f"{yt.title}.mp4")
        video_stream.download(output_path=output_path, filename=f"{yt.title}.mp4")
        print(f"Video downloaded successfully to {video_file}")
        return video_file
    except Exception as e:
        print(f"Error downloading video: {e}")
        return None

# Function to extract audio from video using ffmpeg
def extract_audio(input_video, output_audio):
    try:
        ffmpeg.input(input_video).output(output_audio, ac=1, ar='16000').run(quiet=False, overwrite_output=True)
        return True
    except ffmpeg.Error as e:
        print("Error extracting audio:")
        if e.stderr:
            print(f"FFmpeg error: {e.stderr.decode()}")
        else:
            print("Unknown FFmpeg error occurred")
        return False

# Function to get audio duration
def get_audio_duration(audio_file):
    try:
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', audio_file]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return float(json.loads(result.stdout)['format']['duration'])
    except Exception as e:
        print("Error getting duration:", e)
        return None

# Function to adjust audio duration to match original
def adjust_audio(input_audio, output_audio, target_duration):
    try:
        audio = AudioSegment.from_file(input_audio)
        speed_factor = len(audio) / 1000.0 / target_duration
        atempo = f"atempo={speed_factor*2},atempo=0.5" if speed_factor < 0.5 else f"atempo={speed_factor}"
        subprocess.run(['ffmpeg', '-y', '-i', input_audio, '-filter:a', atempo, output_audio], check=True)
        return True
    except Exception as e:
        print("Error adjusting audio:", e)
        return False

# Function to transcribe audio to text using Whisper
def transcribe_audio(audio_file):
    try:
        return whisper.load_model("base").transcribe(audio_file)["text"]
    except Exception as e:
        print("Error transcribing audio:", e)
        return None

# Function to translate text using googletrans
def translate_text(text, target_language):
    try:
        return Translator().translate(text, dest=target_language).text
    except Exception as e:
        print("Error translating text:", e)
        return None

# Function to convert text to speech using gTTS
def text_to_speech(text, output_file, language):
    try:
        gTTS(text, lang=language).save(output_file)
        return True
    except Exception as e:
        print("Error converting text to speech:", e)
        return False

# Function to browse and select file using file chooser dialog
def browse_file():
    root = Tk()
    root.withdraw()
    root.update()
    file_path = askopenfilename(title="Select Video File", filetypes=[("Video Files", "*.mp4 *.mkv *.avi *.mov *.flv")])
    root.quit()
    return file_path

# Function to load or initialize a known voices database
def load_voice_database():
    try:
        with open("known_voices.pkl", "rb") as db_file:
            known_voices = pickle.load(db_file)
        return known_voices
    except FileNotFoundError:
        return {}

# Function to save new voices to the database
def save_voice_database(known_voices):
    with open("known_voices.pkl", "wb") as db_file:
        pickle.dump(known_voices, db_file)

# Function for voice identification using Resemblyzer
def identify_speaker(audio_file, known_voices):
    try:
        encoder = VoiceEncoder()
        wav = preprocess_wav(audio_file)
        audio_embedding = encoder.embed_utterance(wav)

        closest_match = None
        min_distance = float('inf')

        for speaker, embedding in known_voices.items():
            distance = cosine(audio_embedding, embedding)
            if distance < min_distance:
                min_distance = distance
                closest_match = speaker

        return closest_match if min_distance < 0.5 else None
    except Exception as e:
        print(f"Error identifying speaker: {e}")
        return None

# Function to merge translated audio with the original video (Remove original audio)
def merge_audio_with_video(original_video, translated_audio, output_video):
    try:
        video_stream = ffmpeg.input(original_video)
        audio_stream = ffmpeg.input(translated_audio)

        # Replace original audio with translated audio
        ffmpeg.output(video_stream.video, audio_stream, output_video, vcodec='copy', acodec='aac', strict='experimental').run()
        print(f"Video with translated audio saved to {output_video}")
    except ffmpeg.Error as e:
        print(f"Error during video/audio merge: {e}")

# Main function to process video and audio
def process_video(input_video, target_language="es"):
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    audio_file = os.path.join(output_dir, 'extracted_audio.wav')
    temp_audio_file = os.path.join(output_dir, 'temp_audio.mp3')
    translated_audio_file = os.path.join(output_dir, f'translated_audio_{target_language}.mp3')
    output_video_file = os.path.join(output_dir, 'final_video.mp4')

    if isinstance(input_video, str) and input_video.startswith(('http://', 'https://')):
        print(f"Downloading video from URL: {input_video}")
        video_file = download_video(input_video, output_dir)
        if not video_file:
            print("Failed to download video.")
            return
    else:
        video_file = input_video  # Local file path

    if extract_audio(video_file, audio_file):
        speaker = identify_speaker(audio_file, load_voice_database())
        
        if speaker:
            print(f"Identified speaker: {speaker}")
        else:
            print("Speaker not recognized.")
            user_choice = input("Would you like to add this speaker to the database? (y/n): ").strip().lower()
            if user_choice == 'y':
                speaker_name = input("Enter the name of the speaker: ").strip()
                known_voices = load_voice_database()
                encoder = VoiceEncoder()
                wav = preprocess_wav(audio_file)
                audio_embedding = encoder.embed_utterance(wav)
                known_voices[speaker_name] = audio_embedding
                save_voice_database(known_voices)
                print(f"Speaker {speaker_name} added to the database.")
        
        transcribed_text = transcribe_audio(audio_file)
        if transcribed_text:
            translated_text = translate_text(transcribed_text, target_language)
            if translated_text and text_to_speech(translated_text, temp_audio_file, target_language):
                original_duration = get_audio_duration(audio_file)
                if original_duration and adjust_audio(temp_audio_file, translated_audio_file, original_duration):
                    print(f"Translated audio saved to {translated_audio_file}")
                    merge_audio_with_video(video_file, translated_audio_file, output_video_file)
                os.remove(temp_audio_file)
        os.remove(audio_file)
    print("\nProcess completed.")

if __name__ == "__main__":
    print("Top 12 Languages in India:") 
    print("1. English (en)") 
    print("2. Hindi (hi)") 
    print("3. Bengali (bn)") 
    print("4. Telugu (te)") 
    print("5. Marathi (mr)") 
    print("6. Tamil (ta)") 
    print("7. Urdu (ur)") 
    print("8. Gujarati (gu)") 
    print("9. Malayalam (ml)") 
    print("10. Kannada (kn)") 
    print("11. Odia (or)") 
    print("12. Punjabi (pa)")

    target_language_choice = input("Enter the number corresponding to the language you want (1-12): ").strip()

    language_map = {
        '1': 'en',
        '2': 'hi',  # Hindi
        '3': 'bn',  # Bengali
        '4': 'te',  # Telugu
        '5': 'mr',  # Marathi
        '6': 'ta',  # Tamil
        '7': 'ur',  # Urdu
        '8': 'gu',  # Gujarati
        '9': 'ml',  # Malayalam
        '10': 'kn',  # Kannada
        '11': 'or', # Odia
        '12': 'pa', # Punjabi
    }

    if target_language_choice in language_map:
        target_language = language_map[target_language_choice]
        print(f"Selected language code: {target_language}")
    else:
        print("Invalid choice. Defaulting to Hindi (hi).")
        target_language = 'hi'

    choice = input("Enter '1' for URL or '2' for local file: ").strip()
    if choice == '1':
        input_video = input("Enter video URL: ").strip()
        process_video(input_video, target_language)
    elif choice == '2':
        input_video = browse_file()
        if input_video:
            process_video(input_video, target_language)
        else:
            print("No file selected. Exiting...")
    else:
        print("Invalid choice. Exiting...")
