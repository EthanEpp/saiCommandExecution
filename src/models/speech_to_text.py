import whisper
import pyaudio
import numpy as np
import torch
from src.utils.timing import measure_time

class SpeechToText:
    def __init__(self, model_size="base"):
        self.model = whisper.load_model(model_size)
        self.audio_interface = pyaudio.PyAudio()
        self.stream = None

    def start_microphone_stream(self, rate=16000, chunk_size=1024):
        self.stream = self.audio_interface.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=rate,
            input=True,
            frames_per_buffer=chunk_size
        )

    def stop_microphone_stream(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.audio_interface.terminate()
            
    @measure_time
    def process_audio_stream(self, seconds=5):
        frames = []
        for _ in range(0, int(16000 / 1024 * seconds)):
            data = self.stream.read(1024, exception_on_overflow=False)
            frames.append(data)
        audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
        audio_data = audio_data.astype(np.float32)  # Convert to float32
        audio_data /= 32768.0                       # Normalize the data
        audio_data = torch.from_numpy(audio_data)   # Make tensor writable and convert to PyTorch tensor
        result = self.model.transcribe(audio_data)
        return result['text']
