import pytest
from unittest.mock import Mock
from src.services.command_processor import CommandProcessor
from src.models.embeddings import SentenceEmbedder
from src.models.entity_extractor import SpacyEntityExtractor
from src.models.clause_extractor import ClauseExtractor
from src.services.command import Command

@pytest.fixture
def setup_command_processor():
    embedder = SentenceEmbedder()
    
    commands = [
        {"command": "Turn on the lights", "requires_extraction": False, "entities": [], "clauses": []},
        {"command": "Turn off the lights", "requires_extraction": False, "entities": [], "clauses": []},
        {"command": "Set the thermostat to 22 degrees", "requires_extraction": True, "entities": ["QUANTITY"], "clauses": []}
    ]
    
    processor = CommandProcessor(commands, embedder=embedder)
    return processor

def test_find_closest_command(setup_command_processor):
    processor = setup_command_processor
    
    # Testing with a user input that should match the first command
    user_input = "Turn on the lights"
    expected_output = "Turn on the lights"
    assert processor.find_closest_command(user_input).command_type == expected_output

    # Testing with a user input that should match the second command
    user_input = "Turn off the lights"
    expected_output = "Turn off the lights"
    assert processor.find_closest_command(user_input).command_type == expected_output

    # Testing with a user input that should not match any command
    user_input = "Play some music"
    expected_output = "Command not recognized"
    assert processor.find_closest_command(user_input).command_type == expected_output

def test_similar_phrases(setup_command_processor):
    processor = setup_command_processor
    
    user_input = "Turn on lights"
    expected_output = "Turn on the lights"
    assert processor.find_closest_command(user_input).command_type == expected_output
    
    user_input = "Turn the lights off"
    expected_output = "Turn off the lights"
    assert processor.find_closest_command(user_input).command_type == expected_output

def test_threshold_behavior(setup_command_processor):
    processor = setup_command_processor
    processor.threshold = 0.9  # Adjust the threshold
    
    user_input = "Set thermostat to 22 degrees"
    expected_output = "Set the thermostat to 22 degrees"
    assert processor.find_closest_command(user_input).command_type == expected_output

    user_input = "Play some music"
    expected_output = "Command not recognized"
    assert processor.find_closest_command(user_input).command_type == expected_output

def test_empty_input(setup_command_processor):
    processor = setup_command_processor
    
    user_input = ""
    expected_output = "Command not recognized"
    assert processor.find_closest_command(user_input).command_type == expected_output

def test_non_string_input(setup_command_processor):
    processor = setup_command_processor
    
    user_input = 12345  # Non-string input
    with pytest.raises(Exception):
        processor.find_closest_command(user_input)

def test_varying_thresholds(setup_command_processor):
    processor = setup_command_processor
    
    processor.threshold = 0.0
    user_input = "There is no substance in this statement"
    result = processor.find_closest_command(user_input).command_type
    # Ensure that any command is selected, as the threshold is 0.0
    assert result in [cmd['command'] for cmd in processor.commands]

    processor.threshold = 1.0
    user_input = "Turn on the lights"
    expected_output = "Command not recognized"
    assert processor.find_closest_command(user_input).command_type == expected_output

# def test_command_with_extraction(setup_command_processor):
#     processor = setup_command_processor
    
#     processor.entity_extractor = Mock(spec=SpacyEntityExtractor)
#     processor.clause_extractor = Mock(spec=ClauseExtractor)
#     processor.entity_extractor.extract_entities.return_value = [("22 degrees", "QUANTITY")]
#     processor.clause_extractor.extract_clauses.return_value = [("Set the thermostat", "ccomp")]

#     user_input = "Set the thermostat to 22 degrees"
#     expected_output = "Set the thermostat to 22 degrees"
#     result = processor.find_closest_command(user_input)
    
#     assert result.command_type == expected_output
#     assert result.entities == [("22 degrees", "QUANTITY")]
#     assert result.clauses == [("Set the thermostat", "ccomp")]
    
#     processor.entity_extractor.extract_entities.assert_called_once_with(user_input)
#     processor.clause_extractor.extract_clauses.assert_called_once_with(user_input)
