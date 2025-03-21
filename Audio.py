import os
import ffmpeg
import whisper
from googletrans import Translator
from gtts import gTTS
from pydub import AudioSegment
import subprocess
import json

# Function to extract audio from video using ffmpeg
def extract_audio_from_video(input_video, output_audio):
    try:
        # Extract audio from the video using ffmpeg
        (
            ffmpeg
            .input(input_video)
            .output(output_audio, ac=1, ar='16000')
            .run(quiet=True, overwrite_output=True)
        )
        print(f"Audio extracted and saved to {output_audio}")
        return True
    except Exception as e:
        print("Error extracting audio:", str(e))
        return False

# Function to get audio duration using ffprobe
def get_audio_duration(audio_file):
    try:
        # Run ffprobe command to get duration
        cmd = [
            'ffprobe', 
            '-v', 'error', 
            '-show_entries', 'format=duration', 
            '-of', 'json', 
            audio_file
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        data = json.loads(result.stdout)
        duration = float(data['format']['duration'])
        return duration
    except Exception as e:
        print(f"Error getting audio duration: {str(e)}")
        return None

# Function to adjust audio duration
def adjust_audio_duration(input_audio, output_audio, target_duration):
    try:
        # Load audio file
        audio = AudioSegment.from_file(input_audio)
        
        # Get current duration in seconds
        current_duration = len(audio) / 1000.0
        
        # Calculate speed factor
        speed_factor = current_duration / target_duration
        
        # Use ffmpeg to change tempo without affecting pitch
        if speed_factor > 1.0:  # Need to speed up
            subprocess.run([
                'ffmpeg', '-y',
                '-i', input_audio,
                '-filter:a', f'atempo={speed_factor}',
                output_audio
            ], check=True)
        elif speed_factor < 1.0:  # Need to slow down
            # For significant slowdowns, we may need to use multiple atempo filters
            # as ffmpeg's atempo filter works best in the range 0.5 - 2.0
            if speed_factor < 0.5:
                atempo_val = f"atempo={speed_factor*2},atempo=0.5"
            else:
                atempo_val = f"atempo={speed_factor}"
            
            subprocess.run([
                'ffmpeg', '-y',
                '-i', input_audio,
                '-filter:a', atempo_val,
                output_audio
            ], check=True)
        else:  # No adjustment needed
            subprocess.run([
                'ffmpeg', '-y',
                '-i', input_audio,
                output_audio
            ], check=True)
            
        print(f"Audio duration adjusted from {current_duration:.2f}s to {target_duration:.2f}s")
        return True
    except Exception as e:
        print(f"Error adjusting audio duration: {str(e)}")
        return False

# Function to transcribe audio to text using Whisper
def transcribe_audio(audio_file):
    try:
        # Load whisper model (you can choose model size: "tiny", "base", "small", "medium", "large")
        model = whisper.load_model("base")
        
        # Transcribe audio
        result = model.transcribe(audio_file)
        transcribed_text = result["text"]
        
        print(f"Transcribed Text: {transcribed_text}")
        return transcribed_text
    except Exception as e:
        print("Error during transcription:", str(e))
        return None

# Function to translate text using googletrans
def translate_text(text, target_language="es"):
    try:
        translator = Translator()
        translated = translator.translate(text, dest=target_language)
        return translated.text
    except Exception as e:
        print("Error during translation:", str(e))
        return None

# Function to convert text to speech using gTTS
def text_to_speech(text, output_file, language="en"):
    try:
        tts = gTTS(text=text, lang=language, slow=False)
        tts.save(output_file)
        print(f"Text converted to speech and saved to {output_file}")
        return True
    except Exception as e:
        print("Error converting text to speech:", str(e))
        return False

# Main function to automate the entire process
def process_video(input_video, target_language="es"):
    # Create an output directory if it doesn't exist
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # File paths
    audio_file = os.path.join(output_dir, 'extracted_audio.wav')
    translated_audio_temp = os.path.join(output_dir, f'translated_audio_temp_{target_language}.mp3')
    translated_audio_file = os.path.join(output_dir, f'translated_audio_{target_language}.mp3')
    
    # Step 1: Extract audio from the video
    if not extract_audio_from_video(input_video, audio_file):
        return
    
    # Step 2: Convert the extracted audio to text using Whisper
    transcribed_text = transcribe_audio(audio_file)
    
    if transcribed_text:
        print("Transcribed Text: ", transcribed_text)
        
        # Step 3: Translate the text to the target language
        translated_text = translate_text(transcribed_text, target_language)
        
        if translated_text:
            print(f"Translated Text ({target_language}): ", translated_text)
            
            # Step 4: Convert translated text to speech (temporary file)
            if text_to_speech(translated_text, translated_audio_temp, target_language):
                
                # Step 5: Get original audio duration
                original_duration = get_audio_duration(audio_file)
                if original_duration:
                    # Step 6: Adjust translated audio to match original duration
                    if adjust_audio_duration(translated_audio_temp, translated_audio_file, original_duration):
                        print(f"Translated audio adjusted to match original duration of {original_duration:.2f} seconds")
                        
                        # Clean up temporary translated audio file
                        if os.path.exists(translated_audio_temp):
                            os.remove(translated_audio_temp)
                    else:
                        print("Failed to adjust audio duration.")
                else:
                    print("Failed to get original audio duration.")
            else:
                print("Failed to convert text to speech.")
        else:
            print("Failed to translate text.")
    else:
        print("Failed to transcribe audio.")
    
    # Keep the extracted audio file for reference (uncomment to remove)
    # if os.path.exists(audio_file):
    #     os.remove(audio_file)
    #     print(f"Cleaned up temporary audio file: {audio_file}")

# Example usage:
if __name__ == "__main__":
    input_video = r'C:\Users\Deepak Borole\PycharmProjects\pythonProject\Infosys internship\ai.mp4'  # Replace with your video file path
    target_language = "mr"  # Marathi language code
    process_video(input_video, target_language)