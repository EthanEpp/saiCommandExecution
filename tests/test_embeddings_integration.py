# tests/test_integration.py
import pytest
from src.services.command_processor import CommandProcessor

@pytest.fixture
def commands():
    return [
        "turn on the light",
        "turn off the light",
        "set the thermostat to 22 degrees",
        "play some music"
    ]

@pytest.fixture
def command_processor(commands):
    return CommandProcessor(commands)

def test_find_closest_command(command_processor):
    user_input = "please play music"
    expected_command = "play some music"

    result = command_processor.find_closest_command(user_input)
    
    assert result[0] == expected_command

def test_command_not_recognized(command_processor):
    user_input = "open the window"
    expected_response = "Command not recognized"

    result = command_processor.find_closest_command(user_input)
    
    assert result[0] == expected_response
