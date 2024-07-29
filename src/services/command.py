# THIS METHOD OF COMMAND PROCESSING IS NO LONGER USED, IT IS KEPT IN THE REPOSITORY AS LEGACY FOR NOW

class Command:
    def __init__(self, original_input, preprocessed_input, command_type, entities=None, clauses=None):
        self.original_input = original_input
        self.preprocessed_input = preprocessed_input
        self.command_type = command_type
        self.entities = entities if entities else []
        self.clauses = clauses if clauses else []

    def __repr__(self):
        return (f"Command(command_type={self.command_type}, original_input={self.original_input}, "
                f"preprocessed_input={self.preprocessed_input}, entities={self.entities}, clauses={self.clauses})")
# THIS METHOD OF COMMAND PROCESSING IS NO LONGER USED, IT IS KEPT IN THE REPOSITORY AS LEGACY FOR NOW

