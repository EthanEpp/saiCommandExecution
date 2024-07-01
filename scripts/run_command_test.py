import json
import os
import sys

# Add the parent directory to the Python path so it can find the utils module
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.models.clause_extractor import ClauseExtractor
from src.models.entity_extractor import SpacyEntityExtractor
from src.services.command_processor import CommandProcessor

# Path to the commands.json file
commands_file_path = os.path.abspath(os.path.join(parent_dir, 'data/commands.json'))
test_commands_file_path = os.path.abspath(os.path.join(parent_dir, 'data/test_data/test_commands.json'))

def load_commands(file_path):
    """
    Loads commands from a JSON file.
    
    Args:
    file_path (str): The path to the JSON file containing commands.
    
    Returns:
    list of dict: A list of command dictionaries.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data.get('commands', [])

def load_test_commands(file_path):
    """
    Loads test commands from a JSON file.
    
    Args:
    file_path (str): The path to the JSON file containing test commands.
    
    Returns:
    list of str: A list of test commands.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data.get('test_commands', [])

def process_and_find_nearest_command(command, command_processor):
    """
    Processes a command to find the nearest matching command and extracts clauses and entities.
    
    Args:
    command (str): The command to process.
    command_processor (CommandProcessor): The command processor instance.
    """
    matched_command = command_processor.find_closest_command(command)
    
    print(f"    Test Command: {command}")
    print(f"Preprocessed Input: {matched_command.preprocessed_input}")
    print(f"Matched Command: {matched_command.command_type}")
    
    # Display matched entities
    if matched_command.entities:
        print(f"  Extracted Entities:")
        for entity, entity_label in matched_command.entities:
            print(f"    {entity} (Entity label: {entity_label})")
    
    # Display matched clauses
    if matched_command.clauses:
        print(f"  Extracted Clauses:")
        for clause, dep_label in matched_command.clauses:
            print(f"    {clause} (Dependency label: {dep_label})")
    print("\n")
def main():
    # Load commands from the JSON file
    print(f"Loading commands from: {commands_file_path}")
    commands = load_commands(commands_file_path)
    
    # Load test commands from the JSON file
    print(f"Loading test commands from: {test_commands_file_path}")
    test_commands = load_test_commands(test_commands_file_path)
    
    # Initialize the command processor
    command_processor = CommandProcessor(commands)
    
    # Process each test command to find the nearest command and extract clauses and entities
    for test_command in test_commands:
        process_and_find_nearest_command(test_command, command_processor)

if __name__ == "__main__":
    main()
