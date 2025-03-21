#For audio extraction 
# import ffmpeg

# def process_video(input_file):
#     try:
#         # Get video information (e.g., codec, resolution, duration)
#         probe = ffmpeg.probe(input_file)
#         print("Video Information:")
#         print(probe)
        
#         # Example: Extract a specific portion of the video (first 10 seconds)
#         output_file = 'output.mp4'
#         ffmpeg.input(input_file, ss=0, t=10).output(output_file).run()
#         print(f"Processed video saved to {output_file}")
        
#     except ffmpeg.Error as e:
#         print("Error processing video:", e.stderr.decode())

# # Example usage:
# input_video = r'C:\Users\Deepak Borole\PycharmProjects\pythonProject\Infosys internship\Raghvendra-Solapur-Maharashtra-Solapur.mp4'  # Replace with your video file
# process_video(input_video)

# For audio extraction + text conversion

import ffmpeg
import os
import speech_recognition as sr
from textblob import TextBlob
from googletrans import Translator  # For translation

# Function to extract audio from video using ffmpeg
def extract_audio_from_video(input_video, output_audio):
    try:
        # Extract audio from the video using ffmpeg
        ffmpeg.input(input_video).output(output_audio, ac=1, ar='16000').run()
        print(f"Audio extracted and saved to {output_audio}")
    except ffmpeg.Error as e:
        print("Error extracting audio:", e.stderr.decode())

# Function to transcribe audio to text using SpeechRecognition
def transcribe_audio(audio_file):
    recognizer = sr.Recognizer()

    try:
        with sr.AudioFile(audio_file) as source:
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source)
            # Record the audio data from the file
            audio_data = recognizer.record(source)

            # Use Google's speech recognition API to convert the audio to text
            text = recognizer.recognize_google(audio_data)
            print(f"Transcribed Text: {text}")
            return text
    except Exception as e:
        print("Error during transcription:", str(e))
        return None

# Function to translate text using TextBlob and googletrans
def translate_text(text, target_language="es"):
    translator = Translator()
    translated = translator.translate(text, dest=target_language)
    return translated.text

# Main function to automate the process of extracting audio, transcribing, and translating
def process_video(input_video, target_language="es"):
    # Step 1: Extract audio from the video
    audio_file = 'extracted_audio.wav'
    extract_audio_from_video(input_video, audio_file)

    # Step 2: Convert the extracted audio to text
    transcribed_text = transcribe_audio(audio_file)

    if transcribed_text:
        print("Transcribed Text: ", transcribed_text)

        # Step 3: Translate the text to the target language
        translated_text = translate_text(transcribed_text, target_language)
        print(f"Translated Text ({target_language}): ", translated_text)

        # Step 4: Perform additional processing with TextBlob (e.g., sentiment analysis)
        # blob = TextBlob(transcribed_text)
        # sentiment = blob.sentiment
        # print(f"Sentiment Analysis: {sentiment}")

    else:
        print("Failed to transcribe audio.")

    # Clean up the extracted audio file
    if os.path.exists(audio_file):
        os.remove(audio_file)
        print(f"Cleaned up temporary audio file: {audio_file}")

# Example usage:
input_video = r'C:\Users\Deepak Borole\PycharmProjects\pythonProject\Infosys internship\ai.mp4'  # Replace with your video file path
target_language = "MR"  # Spanish (can change to any target language code like 'fr' for French, etc.)
process_video(input_video, target_language)


# text to speech
# import ffmpeg
# import os
# from googletrans import Translator  # For translation
# from gtts import gTTS  # For text-to-speech conversion
# import whisper
# from pydub import AudioSegment  # For audio manipulation (padding, concatenation)

# # Function to extract audio from video using ffmpeg
# def extract_audio_from_video(input_video, output_audio):
#     try:
#         # Extract audio from the video using ffmpeg
#         ffmpeg.input(input_video).output(output_audio, ac=1, ar='16000').run()
#         return True, output_audio  # Return success status and audio file path
#     except ffmpeg.Error as e:
#         return False, f"Error extracting audio: {e.stderr.decode() if e.stderr else str(e)}"  # Return failure status and error message

# # Function to transcribe audio to text using Whisper
# def transcribe_audio(audio_file):
#     try:
#         # Load Whisper model (can use small/medium/large depending on speed/accuracy needs)
#         model = whisper.load_model("base")  # You can use "small", "medium", or "large"
        
#         # Transcribe audio file
#         result = model.transcribe(audio_file)
        
#         # Return the transcribed text and timestamps
#         return True, result["text"], result["segments"]
    
#     except Exception as e:
#         return False, f"Error during transcription: {str(e)}"  # Return failure status and error message

# # Function to translate text using googletrans
# def translate_text(text, target_language="es"):
#     translator = Translator()
#     try:
#         translated = translator.translate(text, dest=target_language)
#         return True, translated.text  # Return success status and translated text
#     except Exception as e:
#         return False, f"Error during translation: {str(e)}"  # Return failure status and error message

# # Function to convert translated text to speech using gTTS
# def text_to_speech(text, lang='en'):
#     try:
#         # Initialize gTTS engine
#         tts = gTTS(text=text, lang=lang, slow=False)  # slow=False means the speech will be at normal speed
        
#         # Save the generated speech to a file
#         audio_file_path = 'translated_audio.mp3'
#         tts.save(audio_file_path)
        
#         if os.path.exists(audio_file_path):
#             return True, audio_file_path  # Return success status and audio file path
#         else:
#             return False, "Failed to create audio file."  # Return failure status and error message
#     except Exception as e:
#         return False, f"Error during text-to-speech conversion: {str(e)}"  # Return failure status and error message

# # Function to merge video with audio using ffmpeg (mute the original audio and add translated audio)
# # Function to merge video with audio using ffmpeg (mute the original audio and add translated audio)
# def merge_audio_with_video(input_video, input_audio, output_video):
#     try:
#         # Debug: Print FFmpeg command for verification
#         print(f"Running FFmpeg command: ffmpeg.input({input_video}, an=None).output({output_video}, audio={input_audio}, vcodec='copy', acodec='aac', strict='experimental').run()")
        
#         # Merge the video (without original audio) with the translated audio using ffmpeg
#         ffmpeg.input(input_video, an=None) \
#             .output(output_video, audio=input_audio, vcodec='copy', acodec='aac', strict='experimental') \
#             .run()
        
#         return True, output_video  # Return success status and output video file path
#     except ffmpeg.Error as e:
#         # Check for the error in stderr if available, otherwise use a generic error message
#         error_message = e.stderr.decode() if e.stderr else str(e)
#         print(f"FFmpeg error: {error_message}")  # Print the error for debugging
#         return False, f"Error during merging audio with video: {error_message}"  # Return failure status and error message


# # Function to generate translated audio for the full length of the video
# def generate_full_audio_from_translated_text(segments, target_language="en"):
#     full_audio = AudioSegment.silent(duration=0)  # Initialize with silence
    
#     for segment in segments:
#         text = segment["text"]
#         success, translated_text = translate_text(text, target_language)
#         if not success:
#             print(f"Error in translation for segment: {translated_text}")
#             continue
        
#         # Convert the translated text to speech (TTS)
#         success, audio_path = text_to_speech(translated_text, lang=target_language)
#         if not success:
#             print(f"Error in TTS for segment: {audio_path}")
#             continue
        
#         # Load the translated audio
#         segment_audio = AudioSegment.from_mp3(audio_path)
        
#         # Append this translated audio to the full audio
#         full_audio += segment_audio

#         # Clean up translated audio file after use
#         if os.path.exists(audio_path):
#             os.remove(audio_path)

#     return full_audio

# # Main function to automate the process of extracting audio, transcribing, and translating
# def process_video(input_video, target_language="es"):
#     # Step 1: Extract audio from the video
#     success, result = extract_audio_from_video(input_video, 'extracted_audio.wav')
#     if not success:
#         print(f"Error: {result}")
#         return
    
#     print("Audio extraction successful.")
    
#     # Step 2: Transcribe the audio to text using Whisper
#     success, transcribed_text, segments = transcribe_audio(result)
#     if not success:
#         print(f"Error: {transcribed_text}")
#         return
    
#     print(f"Transcribed Text: {transcribed_text}")
    
#     # Step 3: Generate translated audio matching the original audio's length
#     full_audio = generate_full_audio_from_translated_text(segments, target_language)

#     # Step 4: Save the final audio
#     final_audio_file = "final_translated_audio.mp3"
#     full_audio.export(final_audio_file, format="mp3")

#     print(f"Generated full translated audio: {final_audio_file}")
    
#     # Step 5: Merge the translated audio with the original video
#     output_video = 'final_output_video.mp4'
#     success, final_video = merge_audio_with_video(input_video, final_audio_file, output_video)
#     if not success:
#         print(f"Error: {final_video}")
#         return
    
#     print(f"Final video created successfully: {final_video}")
    
#     # Clean up the extracted audio file and translated audio file
#     if os.path.exists(result):
#         os.remove(result)
#         print("Cleaned up extracted audio file.")
    
#     if os.path.exists(final_audio_file):
#         os.remove(final_audio_file)
#         print("Cleaned up translated audio file.")

# # Example usage:
# input_video = r'C:\Users\Deepak Borole\PycharmProjects\pythonProject\Infosys internship\What is AI.mp4'  # Replace with your video file path
# target_language = "hi"  # Hindi language code (can change to any target language code like 'es' for Spanish)
# process_video(input_video, target_language)

# import ffmpeg
# import os
# from googletrans import Translator  # For translation
# from gtts import gTTS  # For text-to-speech conversion
# import whisperx  # Updated for WhisperX usage
# from pydub import AudioSegment  # For audio manipulation (speed adjustment)
# import subprocess  # For subprocess-based execution of ffmpeg
# import tempfile

# # Function to extract audio from video using ffmpeg
# def extract_audio_from_video(input_video, output_audio):
#     try:
#         # Extract audio from the video using ffmpeg
#         ffmpeg.input(input_video).output(output_audio, ac=1, ar='16000').run()
#         return True, output_audio  # Return success status and audio file path
#     except ffmpeg.Error as e:
#         return False, f"Error extracting audio: {e.stderr.decode() if e.stderr else str(e)}"  # Return failure status and error message

# # Function to transcribe audio to text using WhisperX
# def transcribe_audio_with_whisperx(audio_file):
#     try:
#         # Specify the device to use (either "cpu" or "cuda")
#         device = "cpu"  # Change to "cuda" if you have a compatible GPU

#         # Load the WhisperX model (use "base" or other model sizes)
#         model, metadata = whisperx.load_model("base", device=device)

#         # Perform transcription (this method is based on the latest WhisperX API)
#         result = model.transcribe(audio_file)  # Use the model directly for transcription
        
#         # Return the transcribed text and segments
#         return result['text'], result['segments']  # Return transcribed text and segments
#     except Exception as e:
#         return f"Error during transcription with WhisperX: {str(e)}", None  # Handle error and return appropriate message

# # Function to translate text using googletrans
# def translate_text(text, target_language="es"):
#     translator = Translator()
#     try:
#         translated = translator.translate(text, dest=target_language)
#         return True, translated.text  # Return success status and translated text
#     except Exception as e:
#         return False, f"Error during translation: {str(e)}"  # Return failure status and error message

# # Function to convert translated text to speech using gTTS
# def text_to_speech(text, lang='en'):
#     try:
#         # Initialize gTTS engine
#         tts = gTTS(text=text, lang=lang, slow=False)  # slow=False means the speech will be at normal speed
        
#         # Save the generated speech to a file
#         audio_file_path = 'translated_audio.mp3'
#         tts.save(audio_file_path)
        
#         if os.path.exists(audio_file_path):
#             return True, audio_file_path  # Return success status and audio file path
#         else:
#             return False, "Failed to create audio file."  # Return failure status and error message
#     except Exception as e:
#         return False, f"Error during text-to-speech conversion: {str(e)}"  # Return failure status and error message

# # Function to generate full translated audio from text segments
# def generate_full_audio_from_translated_text(segments, target_language="es"):
#     full_audio = AudioSegment.silent(duration=0)  # Start with a silent audio track
#     for segment in segments:
#         # Translate each segment
#         success, translated_text = translate_text(segment['text'], target_language)
#         if not success:
#             print(f"Error during translation: {translated_text}")
#             return None
#         # Convert translated text to speech
#         success, audio_path = text_to_speech(translated_text, lang=target_language)
#         if not success:
#             print(f"Error during text-to-speech: {audio_path}")
#             return None
#         # Load the generated audio and append it to the full audio
#         segment_audio = AudioSegment.from_mp3(audio_path)
#         full_audio += segment_audio
#         # Clean up the temporary audio file
#         os.remove(audio_path)
#     return full_audio

# # Function to adjust audio speed to match video duration
# def adjust_audio_speed_to_match_video_duration(audio, video_duration):
#     audio_duration = len(audio) / 1000  # Convert from milliseconds to seconds
#     speed_factor = audio_duration / video_duration  # Calculate the speed adjustment factor
#     adjusted_audio = audio.speedup(playback_speed=speed_factor)
#     return adjusted_audio

# # Function to merge video with audio using ffmpeg (mute the original audio and add translated audio)
# def merge_audio_with_video(input_video, input_audio, output_video):
#     try:
#         # Correctly create input streams for video and audio
#         video_stream = ffmpeg.input(input_video, an=None)  # 'an' option disables the original audio
#         audio_stream = ffmpeg.input(input_audio)  # Correct way to pass the audio stream
        
#         # Merge the video (without original audio) with the translated audio using ffmpeg
#         ffmpeg.output(video_stream, audio_stream, output_video, vcodec='copy', acodec='aac').run()
#         return True, output_video  # Return success status and output video file path
#     except ffmpeg.Error as e:
#         # Check for the error in stderr if available, otherwise use a generic error message
#         error_message = e.stderr.decode() if e.stderr else str(e)
#         print(f"FFmpeg error: {error_message}")  # Print the error for debugging
#         return False, f"Error during merging audio with video: {error_message}"  # Return failure status and error message

# # Main function to automate the process of extracting audio, transcribing, translating, and merging with video
# def process_video(input_video, target_language="es"):
#     # Step 1: Extract audio from the video
#     print("Step 1: Extracting audio from the video...")
#     success, extracted_audio = extract_audio_from_video(input_video, 'extracted_audio.wav')
#     if not success:
#         print(f"Error: {extracted_audio}")
#         return
#     print("Audio extraction successful.")
    
#     # Step 2: Transcribe the audio to text using WhisperX
#     print("Step 2: Transcribing audio to text...")
#     success, transcribed_text, segments = transcribe_audio_with_whisperx(extracted_audio)
#     if not success:
#         print(f"Error: {transcribed_text}")
#         return
#     print(f"Transcribed Text: {transcribed_text}")
    
#     # Step 3: Generate translated audio for the full length of the video
#     print("Step 3: Generating translated audio...")
#     full_translated_audio = generate_full_audio_from_translated_text(segments, target_language)
#     if full_translated_audio is None:
#         return
    
#     # Step 4: Get the duration of the video
#     video_duration = ffmpeg.probe(input_video, v='error', select_streams='v:0', show_entries='stream=duration')['streams'][0]['duration']
#     video_duration = float(video_duration)
#     print(f"Video duration: {video_duration} seconds")

#     # Step 5: Adjust the audio speed to match the video's duration using pydub
#     print("Step 5: Adjusting audio duration to match the video...")
#     full_translated_audio = adjust_audio_speed_to_match_video_duration(full_translated_audio, video_duration)
    
#     # Step 6: Save the final translated audio
#     final_audio_file = "final_translated_audio.wav"
#     full_translated_audio.export(final_audio_file, format="wav")
#     print(f"Generated full translated audio: {final_audio_file}")
    
#     # Step 7: Merge the translated audio with the original video
#     print("Step 6: Merging translated audio with the video...")
#     output_video = 'final_output_video.mp4'
#     success, final_video = merge_audio_with_video(input_video, final_audio_file, output_video)
#     if not success:
#         print(f"Error: {final_video}")
#         return
    
#     print(f"Final video created successfully: {final_video}")
    
#     # Clean up the extracted audio file and translated audio file
#     if os.path.exists(extracted_audio):
#         os.remove(extracted_audio)
#         print("Cleaned up extracted audio file.")
    
#     if os.path.exists(final_audio_file):
#         os.remove(final_audio_file)
#         print("Cleaned up translated audio file.")

# # Example usage:
# input_video = r'C:\Users\Deepak Borole\PycharmProjects\pythonProject\Infosys internship\ai.mp4'  # Replace with your video file path
# target_language = "hi"  # Hindi language code (can change to any target language code like 'es' for Spanish)

# # Run the video processing function
# process_video(input_video, target_language)

