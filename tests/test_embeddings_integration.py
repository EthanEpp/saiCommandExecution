import pytest
from src.services.command_processor import CommandProcessor

@pytest.fixture
def commands():
    return [
        {"command": "turn on the light", "requires_extraction": False},
        {"command": "turn off the light", "requires_extraction": False},
        {"command": "set the thermostat to 22 degrees", "requires_extraction": True},
        {"command": "play some music", "requires_extraction": False}
    ]

@pytest.fixture
def command_processor(commands):
    return CommandProcessor(commands)

def test_find_closest_command(command_processor):
    user_input = "please play music"
    expected_command = "play some music"

    result = command_processor.find_closest_command(user_input)
    
    assert result[0] == expected_command
    assert result[1] is None  # No entities expected
    assert result[2] is None  # No clauses expected

def test_find_command_with_extraction(command_processor):
    user_input = "set thermostat to 22 degrees"
    expected_command = "set the thermostat to 22 degrees"

    result = command_processor.find_closest_command(user_input)
    
    assert result[0] == expected_command
    assert result[1] is not None  # Entities expected
    assert result[2] is not None  # Clauses expected

def test_command_not_recognized(command_processor):
    user_input = "open the window"
    expected_response = "Command not recognized"

    result = command_processor.find_closest_command(user_input)
    
    assert result[0] == expected_response
    assert result[1] is None  # No entities expected
    assert result[2] is None  # No clauses expected
