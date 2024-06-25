import sys
import os

# Add the parent directory to the Python path so it can find the utils module
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from transformers import pipeline
from src.utils.audio_utils import convert_folder_to_wav

def transcribe_audio_with_whisper(audio_path):
    whisper = pipeline("automatic-speech-recognition", model="openai/whisper-small")
    transcription = whisper(audio_path)
    return transcription['text']

def transcribe_folder(source_folder):
    target_folder = source_folder + "_wav"
    convert_folder_to_wav(source_folder, target_folder)

    for filename in os.listdir(target_folder):
        if filename.endswith('.wav'):
            audio_path = os.path.join(target_folder, filename)
            transcription = transcribe_audio_with_whisper(audio_path)
            print(f"File: {filename}\nTranscription: {transcription}\n")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
        transcribe_folder(folder_path)
        print(f"All audio files in {folder_path} have been transcribed and saved in {folder_path}_wav.")
    else:
        print("Please provide the path to the folder containing audio files.")
