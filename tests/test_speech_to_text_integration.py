# tests/test_speech_to_text_integration.py
import pytest
import pyaudio
import numpy as np
import torch
import wave
from unittest.mock import patch, MagicMock
from src.models.speech_to_text import SpeechToText  # Adjust the import path if necessary

@pytest.fixture
def speech_to_text():
    """Fixture to initialize the SpeechToText object for use in tests."""
    return SpeechToText(model_size="tiny")  # Use a smaller model for faster tests

def read_wav_file(file_path):
    """Utility function to read a WAV file and return the audio data as bytes."""
    with wave.open(file_path, 'rb') as wf:
        return wf.readframes(wf.getnframes())

@patch('pyaudio.PyAudio')
def test_process_prerecorded_audio_with_mock(mock_pyaudio, speech_to_text):
    """Integration test to process prerecorded audio and verify transcription using a mocked model."""
    # Mock the PyAudio instance and stream
    mock_instance = mock_pyaudio.return_value
    mock_stream = MagicMock()
    mock_instance.open.return_value = mock_stream

    # Path to the prerecorded audio file
    audio_file_path = "tests/data/wav/New Recording.wav"

    # Read audio data from the prerecorded file
    audio_data_bytes = read_wav_file(audio_file_path)
    audio_data_np = np.frombuffer(audio_data_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    # Mock the model's transcribe method to return a specific transcription
    speech_to_text.model.transcribe = MagicMock(return_value={'text': 'expected transcription text'})

    # Mock the behavior of the stream to return the prerecorded audio data
    mock_stream.read = MagicMock(side_effect=[audio_data_bytes[i:i+1024] for i in range(0, len(audio_data_bytes), 1024)])

    # Start and stop the stream to set up the audio interface
    speech_to_text.start_microphone_stream()
    speech_to_text.stream = mock_stream

    # Process the audio stream and check the result
    result = speech_to_text.process_audio_stream(seconds=5)
    assert result == 'expected transcription text'
    speech_to_text.model.transcribe.assert_called_once()

    # Clean up
    speech_to_text.stop_microphone_stream()

def test_prerecorded_audio():
    # Initialize the SpeechToText with the base model size
    stt = SpeechToText(model_size="base")

    # Load prerecorded audio file
    audio_path = 'tests/data/wav/this_is_a_test_audio.wav'
    wf = wave.open(audio_path, 'rb')
    assert wf.getsampwidth() == 2, "Expected 16-bit PCM"
    assert wf.getframerate() == 16000, "Expected 16 kHz sample rate"

    # Read the audio data
    frames = []
    chunk_size = 1024
    data = wf.readframes(chunk_size)
    while data:
        frames.append(data)
        data = wf.readframes(chunk_size)

    # Convert audio data to numpy array
    audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
    audio_data = audio_data.astype(np.float32)  # Convert to float32
    audio_data /= 32768.0                       # Normalize the data
    audio_data = torch.from_numpy(audio_data)   # Convert to PyTorch tensor

    # Transcribe the audio data
    result = stt.model.transcribe(audio_data)
    transcribed_text = result['text']

    # Define the expected transcription
    expected_text = "This is a test audio."

    # Assert the transcribed output equals the expected text
    assert transcribed_text.strip() == expected_text