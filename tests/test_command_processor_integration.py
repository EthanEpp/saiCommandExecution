# tests/test_command_processor_integration.py
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


def test_similar_phrases(setup_command_processor):
    processor = setup_command_processor
    
    user_input = "Turn on lights"
    expected_output = "Turn on the lights"
    assert processor.find_closest_command(user_input) == expected_output
    
    user_input = "Turn the lights off"
    expected_output = "Turn off the lights"
    assert processor.find_closest_command(user_input) == expected_output

def test_threshold_behavior(setup_command_processor):
    processor = setup_command_processor
    processor.threshold = 0.9  # Adjust the threshold
    
    user_input = "Set thermostat to 22 degrees"
    expected_output = "Set the thermostat to 22 degrees"
    assert processor.find_closest_command(user_input) == expected_output

    user_input = "Play some music"
    expected_output = "Command not recognized"
    assert processor.find_closest_command(user_input) == expected_output

def test_empty_input(setup_command_processor):
    processor = setup_command_processor
    
    user_input = ""
    expected_output = "Command not recognized"
    assert processor.find_closest_command(user_input) == expected_output

def test_non_string_input(setup_command_processor):
    processor = setup_command_processor
    
    user_input = 12345  # Non-string input
    with pytest.raises(Exception):
        processor.find_closest_command(user_input)

def test_varying_thresholds(setup_command_processor):
    processor = setup_command_processor
    
    processor.threshold = 0.0
    user_input = "There is no substance in this statement"
    result = processor.find_closest_command(user_input)
    # Ensure that any command is selected, as the threshold is 0.0
    assert result in processor.commands

    processor.threshold = 1.0
    user_input = "Turn on the lights"
    expected_output = "Command not recognized"
    assert processor.find_closest_command(user_input) == expected_output