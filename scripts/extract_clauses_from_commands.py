import json
import os
import sys

# Add the parent directory to the Python path so it can find the utils module
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.models.clause_extractor import ClauseExtractor
from src.models.entity_extractor import SpacyEntityExtractor

# Path to the commands.json file
commands_file_path = os.path.abspath(os.path.join(current_dir, '../data/commands.json'))

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

def process_commands(commands, clause_extractor, entity_extractor):
    """
    Processes a list of commands to extract and log clauses and entities.
    a
    Args:
    commands (list of dict): The list of command dictionaries to process.
    clause_extractor (ClauseExtractor): The clause extractor instance.
    entity_extractor (SpacyEntityExtractor): The entity extractor instance.
    """
    for command_dict in commands:
        command = command_dict.get('command', '')
        clauses = clause_extractor.extract_clauses(command)
        entities = entity_extractor.extract_entities(command)
        
        print(f"Command: {command}")
        
        # Display extracted clauses
        for clause, dep_label in clauses:
            print(f"  Extracted clause: {clause} (Dependency label: {dep_label})")
        
        # Display extracted entities
        for entity, entity_label in entities:
            print(f"  Extracted entity: {entity} (Entity label: {entity_label})")

def main():
    # Load commands from the JSON file
    print(f"Loading commands from: {commands_file_path}")
    commands = load_commands(commands_file_path)
    
    # Initialize the extractors
    clause_extractor = ClauseExtractor()
    entity_extractor = SpacyEntityExtractor()
    
    # Process each command to extract and log clauses and entities
    process_commands(commands, clause_extractor, entity_extractor)

if __name__ == "__main__":
    main()
