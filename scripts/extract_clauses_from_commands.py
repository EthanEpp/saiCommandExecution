import json
import os
import sys

# Add the parent directory to the Python path so it can find the utils module
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.models.clause_extractor import ClauseExtractor

# Path to the commands.json file
commands_file_path = os.path.abspath(os.path.join(current_dir, '../data/commands.json'))

def load_commands(file_path):
    """
    Loads commands from a JSON file.
    
    Args:
    file_path (str): The path to the JSON file containing commands.
    
    Returns:
    list of str: A list of commands.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data.get('commands', [])

def process_commands(commands, extractor):
    """
    Processes a list of commands to extract and log clauses along with their dependency labels.
    
    Args:
    commands (list of str): The list of commands to process.
    extractor (ClauseExtractor): The clause extractor instance.
    """
    for command in commands:
        clauses = extractor.extract_clauses(command)
        print(f"Command: {command}")
        for clause, dep_label in clauses:
            print(f"  Extracted clause: {clause} (Dependency label: {dep_label})")

def main():
    # Load commands from the JSON file
    print(f"Loading commands from: {commands_file_path}")
    commands = load_commands(commands_file_path)
    
    # Initialize the ClauseExtractor
    extractor = ClauseExtractor()
    
    # Process each command to extract and log clauses
    process_commands(commands, extractor)

if __name__ == "__main__":
    main()