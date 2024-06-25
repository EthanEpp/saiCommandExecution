# tests/test_command_processor.py
import pytest
import numpy as np
from unittest.mock import MagicMock
from src.services.command_processor import CommandProcessor
from src.models.embeddings import SentenceEmbedder

@pytest.fixture
def setup_command_processor():
    # Using the actual SentenceEmbedder class
    embedder = SentenceEmbedder()
    embedding_dim = embedder.get_embedding_dim()
    
    # Create real embeddings for the commands
    command_embeddings = {
        "Turn on the lights": embedder.encode(["Turn on the lights"]).squeeze(),
        "Turn off the lights": embedder.encode(["Turn off the lights"]).squeeze(),
        "Set the thermostat to 22 degrees": embedder.encode(["Set the thermostat to 22 degrees"]).squeeze()
    }
    
    # Mock encode method to return appropriate embeddings based on input
    def mock_encode(sentences):
        return np.array([command_embeddings.get(sentence, np.zeros(embedding_dim)) for sentence in sentences])
    
    embedder.encode = MagicMock(side_effect=mock_encode)
    
    # Sample commands
    commands = ["Turn on the lights", "Turn off the lights", "Set the thermostat to 22 degrees"]
    
    # Instantiating the CommandProcessor with the actual embedder
    processor = CommandProcessor(commands)
    processor.embedder = embedder  # Replacing the embedder with the mock
    return processor

def test_find_closest_command(setup_command_processor):
    processor = setup_command_processor
    
    # Testing with a user input that should match the first command
    user_input = "Turn on the lights"
    expected_output = "Turn on the lights"
    assert processor.find_closest_command(user_input) == expected_output

    # Testing with a user input that should match the second command
    user_input = "Turn off the lights"
    expected_output = "Turn off the lights"
    assert processor.find_closest_command(user_input) == expected_output

    # Testing with a user input that should not match any command
    user_input = "Play some music"
    expected_output = "Command not recognized"
    assert processor.find_closest_command(user_input) == expected_output

def test_find_closest_command_with_low_similarity(setup_command_processor):
    processor = setup_command_processor
    
    # Using the actual embedder to create a low similarity embedding
    embedding_dim = processor.embedder.get_embedding_dim()
    def mock_encode(sentences):
        if sentences == ["Play some music"]:
            return np.array([np.zeros(embedding_dim)])
        else:
            return np.array([np.ones(embedding_dim)])
    
    processor.embedder.encode = MagicMock(side_effect=mock_encode)
    
    # Testing with a user input that should not be recognized due to low similarity
    user_input = "Play some music"
    expected_output = "Command not recognized"
    assert processor.find_closest_command(user_input) == expected_output
