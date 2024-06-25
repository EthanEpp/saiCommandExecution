# tests/integration/test_command_processor_integration.py
import pytest
from src.services.command_processor import CommandProcessor
from src.models.embeddings import SentenceEmbedder

@pytest.fixture
def setup_command_processor():
    embedder = SentenceEmbedder()
    
    commands = ["Turn on the lights", "Turn off the lights", "Set the thermostat to 22 degrees"]
    
    processor = CommandProcessor(commands)
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
