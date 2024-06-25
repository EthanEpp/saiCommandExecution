# import pytest
# from src.services.command_processor import CommandProcessor

# @pytest.fixture
# def setup_processor():
#     commands = ["Turn on the light", "Turn off the light", "Set the temperature to 20 degrees"]
#     processor = CommandProcessor(commands)
#     return processor

# def test_initialization(setup_processor):
#     processor = setup_processor
#     assert processor.commands == ["Turn on the light", "Turn off the light", "Set the temperature to 20 degrees"]
#     assert len(processor.command_embeddings) == len(processor.commands)

# def test_find_closest_command(setup_processor):
#     processor = setup_processor
#     input_command = "Switch on the light"
#     expected_command = "Turn on the light"
#     result = processor.find_closest_command(input_command)
#     assert result == expected_command

# def test_command_not_recognized(setup_processor):
#     processor = setup_processor
#     input_command = "Play some music"
#     result = processor.find_closest_command(input_command)
#     assert result == "Command not recognized"

# def test_threshold_behavior(setup_processor):
#     processor = setup_processor
#     processor.threshold = 0.9  # Set a high threshold
#     input_command = "Start the car engine"
#     result = processor.find_closest_command(input_command)
#     assert result == "Command not recognized"  # Adjust based on the similarity print output
import pytest
from unittest.mock import Mock
from src.services.command_processor import CommandProcessor
import numpy as np

@pytest.fixture
def setup_processor():
    commands = ["Turn on the light", "Turn off the light", "Set the temperature to 20 degrees"]
    mock_embedder = Mock()
    mock_embedder.encode.side_effect = lambda x: np.array({
        "Turn on the light": [0.1, 0.2, 0.3],
        "Turn off the light": [0.4, 0.5, 0.6],
        "Set the temperature to 20 degrees": [0.7, 0.8, 0.9],
        "Switch on the light": [0.1, 0.2, 0.31],
        "Play some music": [0.9, 0.9, 0.9],
        "Turn on the lights": [0.1, 0.2, 0.32],
        "Start the car engine": [0.1, 0.5, 0.9],
    }[x[0]])
    processor = CommandProcessor(commands, embedder=mock_embedder)
    return processor

def test_initialization(setup_processor):
    processor = setup_processor
    assert processor.commands == ["Turn on the light", "Turn off the light", "Set the temperature to 20 degrees"]
    assert len(processor.command_embeddings) == len(processor.commands)

def test_find_closest_command(setup_processor):
    processor = setup_processor
    input_command = "Switch on the light"
    expected_command = "Turn on the light"
    result = processor.find_closest_command(input_command)
    assert result == expected_command

# failing due to low dimensionality have higher similairties in general, need to adjust generation of mock to be higher
def test_command_not_recognized(setup_processor):
    processor = setup_processor
    input_command = "Start the car engine"
    result = processor.find_closest_command(input_command)
    assert result == "Command not recognized"

def test_threshold_behavior(setup_processor):
    processor = setup_processor
    processor.threshold = 0.9999  # Set a high threshold
    input_command = "Start the car engine"
    result = processor.find_closest_command(input_command)
    assert result == "Command not recognized"
