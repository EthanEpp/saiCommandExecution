import pytest
from unittest.mock import Mock
from src.services.command_processor import CommandProcessor
import numpy as np

@pytest.fixture
def setup_processor():
    commands = ["Turn on the light", "Turn off the light", "Set the temperature to 20 degrees"]
    mock_embedder = Mock()

    # Create base embeddings for similar commands
    base_turn_on_light = np.random.rand(384)
    base_turn_off_light = base_turn_on_light + np.random.normal(0, 0.03, 384)
    base_set_temperature = np.random.rand(384)

    # Add small variations to simulate similar commands
    embeddings = {
        "Turn on the light": base_turn_on_light,
        "Turn off the light": base_turn_off_light,
        "Switch on the light": base_turn_on_light + np.random.normal(0, 0.01, 384),
        "Turn on the lights": base_turn_on_light + np.random.normal(0, 0.01, 384),
        "Play some music": np.random.rand(384),
        "Set the temperature to 20 degrees": base_set_temperature,
        "Start the car engine": np.random.rand(384)
    }

    mock_embedder.encode.side_effect = lambda x: embeddings[x[0]]
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
    assert result[0] == expected_command

def test_command_not_recognized(setup_processor):
    processor = setup_processor
    input_command = "Start the car engine"
    result = processor.find_closest_command(input_command)
    assert result[0] == "Command not recognized"

def test_threshold_behavior(setup_processor):
    processor = setup_processor
    processor.threshold = 0.9  # Set a high threshold
    input_command = "Start the car engine"
    result = processor.find_closest_command(input_command)
    assert result[0] == "Command not recognized"

def test_exact_match(setup_processor):
    processor = setup_processor
    input_command = "Turn on the light"
    expected_command = "Turn on the light"
    result = processor.find_closest_command(input_command)
    assert result[0] == expected_command