import pytest
from src.services.command_processor import CommandProcessor

@pytest.fixture
def commands():
    return [
        {"command": "turn on the light", "requires_extraction": False, "entities": [], "clauses": []},
        {"command": "turn off the light", "requires_extraction": False, "entities": [], "clauses": []},
        {"command": "set the thermostat to 22 degrees", "requires_extraction": True, "entities": ["CARDINAL"], "clauses": []},
        {"command": "play some music", "requires_extraction": False, "entities": [], "clauses": []}
    ]

@pytest.fixture
def command_processor(commands):
    return CommandProcessor(commands)

def test_find_closest_command(command_processor):
    user_input = "please play music"
    expected_command = "play some music"

    result = command_processor.find_closest_command(user_input)
    
    assert result.command_type == expected_command
    assert result.entities == []  # No entities expected
    assert result.clauses == []  # No clauses expected

def test_command_not_recognized(command_processor):
    user_input = "open the window"
    expected_response = "Command not recognized"

    result = command_processor.find_closest_command(user_input)
    
    assert result.command_type == expected_response
    assert result.entities == []  # No entities expected
    assert result.clauses == []  # No clauses expected

# def test_find_command_with_extraction(command_processor):
#     user_input = "set thermostat to 22 degrees"
#     expected_command = "set the thermostat to 22 degrees"

#     result = command_processor.find_closest_command(user_input)
    
#     assert result.command_type == expected_command
#     assert result.entities is not None  # Entities expected
#     assert any(entity_label == "CARDINAL" for _, entity_label in result.entities)  # Check for specific entity
#     assert result.clauses == []  # No clauses expected
