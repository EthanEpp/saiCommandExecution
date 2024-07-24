# import sys
# import os

# # Add the parent directory to the Python path so it can find the utils module
# current_dir = os.path.dirname(os.path.realpath(__file__))
# parent_dir = os.path.dirname(current_dir)
# sys.path.append(parent_dir)

# from transformers import pipeline
# from src.utils.audio_utils import convert_folder_to_wav

# def transcribe_audio_with_whisper(audio_path):
#     whisper = pipeline("automatic-speech-recognition", model="openai/whisper-small")
#     transcription = whisper(audio_path)
#     return transcription['text']

# def transcribe_folder(source_folder):
#     target_folder = source_folder + "_wav"
#     convert_folder_to_wav(source_folder, target_folder)

#     for filename in os.listdir(target_folder):
#         if filename.endswith('.wav'):
#             audio_path = os.path.join(target_folder, filename)
#             transcription = transcribe_audio_with_whisper(audio_path)
#             print(f"File: {filename}\nTranscription: {transcription}\n")

# # Example usage: python scripts/whisper_test.py audio_files
# if __name__ == "__main__":
#     import sys
#     if len(sys.argv) > 1:
#         folder_path = sys.argv[1]
#         transcribe_folder(folder_path)
#         print(f"All audio files in {folder_path} have been transcribed and saved in {folder_path}_wav.")
#     else:
#         print("Please provide the path to the folder containing audio files.")
import sys
import os
import numpy as np
import pyaudio
from transformers import pipeline

# Add the parent directory to the Python path so it can find the utils module
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.utils.audio_utils import convert_folder_to_wav

# Whisper setup
whisper = pipeline("automatic-speech-recognition", model="openai/whisper-base")

# Audio setup
CHUNK = 2048  # Number of audio frames per buffer
FORMAT = pyaudio.paInt16  # Audio format (16-bit PCM)
CHANNELS = 1  # Number of audio channels
RATE = 16000  # Sample rate (samples per second)
SEGMENT_DURATION = 2  # Segment length in seconds
OVERLAP_DURATION = 0.5  # Overlap length in seconds

def transcribe_audio_with_whisper(audio_data):
    audio_data = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    transcription = whisper(audio_data)
    return transcription['text']

def record_and_transcribe():
    p = pyaudio.PyAudio()

    # Open the stream
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Recording and transcribing...")

    previous_transcription = ""
    segment_frames = int(RATE * SEGMENT_DURATION)
    overlap_frames = int(RATE * OVERLAP_DURATION)
    buffer = b''

    try:
        while True:
            # Read data in chunks
            buffer += stream.read(CHUNK, exception_on_overflow=False)
            if len(buffer) >= segment_frames * 2:
                # Extract the segment for transcription
                segment = buffer[:segment_frames * 2]
                buffer = buffer[segment_frames * 2 - overlap_frames * 2:]

                # Transcribe the segment
                transcription = transcribe_audio_with_whisper(segment)
                previous_transcription += " " + transcription
                print(f"Transcription: {previous_transcription.strip()}\n")

    except KeyboardInterrupt:
        print("Stopping...")

    stream.stop_stream()
    stream.close()
    p.terminate()

# Example usage: python scripts/whisper_test.py
if __name__ == "__main__":
    record_and_transcribe()
