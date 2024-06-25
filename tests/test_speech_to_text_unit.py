# tests/test_speech_to_text.py
import pytest
import numpy as np
import torch
from unittest.mock import MagicMock, patch
from src.models.speech_to_text import SpeechToText  # Adjust the import path if necessary

@pytest.fixture
def speech_to_text():
    """Fixture to initialize the SpeechToText object for use in tests."""
    return SpeechToText(model_size="tiny")  # Use a smaller model for faster tests

def test_initialization(speech_to_text):
    """Test if the SpeechToText object is initialized correctly."""
    assert speech_to_text.model is not None  # Check if the model is loaded
    assert speech_to_text.audio_interface is not None  # Check if the audio interface is initialized
    assert speech_to_text.stream is None  # Ensure the stream is initially set to None

@patch('pyaudio.PyAudio.open', return_value=MagicMock())
def test_start_microphone_stream(mock_open, speech_to_text):
    """Test the start_microphone_stream method to ensure it opens a stream."""
    speech_to_text.start_microphone_stream()
    assert speech_to_text.stream is not None  # Check if the stream is initialized
    mock_open.assert_called_once()  # Verify that the open method was called once

def test_stop_microphone_stream(speech_to_text):
    """Test the stop_microphone_stream method to ensure it stops and closes the stream properly."""
    # Mock the audio interface and stream
    speech_to_text.audio_interface = MagicMock()
    speech_to_text.stream = MagicMock()
    
    # Call the method to stop the stream
    speech_to_text.stop_microphone_stream()
    
    # Check if the stop_stream and close methods were called on the stream
    speech_to_text.stream.stop_stream.assert_called_once()
    speech_to_text.stream.close.assert_called_once()
    
    # Check if the terminate method was called on the audio interface
    speech_to_text.audio_interface.terminate.assert_called_once()

@patch('pyaudio.PyAudio.open', return_value=MagicMock())
@patch('whisper.load_model')
def test_process_audio_stream(mock_load_model, mock_open, speech_to_text):
    """Test the process_audio_stream method to ensure it processes audio data and returns the correct transcription."""
    # Mock the behavior of the stream to return random audio data
    mock_stream = MagicMock()
    mock_stream.read = MagicMock(return_value=np.random.randn(1024).astype(np.int16).tobytes())
    speech_to_text.stream = mock_stream

    # Mock the model's transcribe method to return a specific transcription
    mock_model = MagicMock()
    mock_model.transcribe = MagicMock(return_value={'text': 'transcribed text'})
    speech_to_text.model = mock_model

    # Process the audio stream and check the result
    result = speech_to_text.process_audio_stream(seconds=1)
    assert result == 'transcribed text'  # Verify the transcription result
    mock_stream.read.assert_called()  # Check if the read method was called on the stream
    mock_model.transcribe.assert_called_once()  # Ensure the transcribe method was called once

# Add more tests as needed
