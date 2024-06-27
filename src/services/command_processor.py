# from src.models.embeddings import SentenceEmbedder
# from scipy.spatial.distance import cosine
# import numpy as np
# from src.models.entity_extractor import SpacyEntityExtractor
# from src.models.clause_extractor import ClauseExtractor

# class CommandProcessor:
#     def __init__(self, commands, threshold=0.8, embedder=None, entity_extractor=None, clause_extractor=None):
#         self.embedder = embedder if embedder else SentenceEmbedder()
#         self.entity_extractor = entity_extractor if entity_extractor else SpacyEntityExtractor()
#         self.clause_extractor = clause_extractor if clause_extractor else ClauseExtractor()
#         self.commands = commands
#         self.command_embeddings = [self.embedder.encode([cmd["command"]]).squeeze() for cmd in commands]
#         self.threshold = threshold

#     def preprocess_input(self, text):
#         entities = self.entity_extractor.extract_entities(text)
#         clauses = self.clause_extractor.extract_clauses(text)

#         preprocessed_text = text
#         for entity, label in entities:
#             preprocessed_text = preprocessed_text.replace(entity, f"<{label}>")

#         for clause, dep in clauses:
#             preprocessed_text = preprocessed_text.replace(clause, f"<{dep}>")

#         return preprocessed_text, entities, clauses

#     def find_closest_command(self, user_input):
#         preprocessed_input, entities, clauses = self.preprocess_input(user_input)
#         user_embedding = self.embedder.encode([preprocessed_input]).squeeze()  # Ensure it's 1-D
#         similarities = [1 - cosine(user_embedding, cmd_emb) for cmd_emb in self.command_embeddings]
#         max_similarity = max(similarities)
#         best_match = self.commands[similarities.index(max_similarity)]

#         if max_similarity > self.threshold:
#             return best_match["command"], entities, clauses
#         else:
#             return "Command not recognized", entities, clauses

from src.models.embeddings import SentenceEmbedder
from scipy.spatial.distance import cosine
import numpy as np
from src.models.entity_extractor import SpacyEntityExtractor
from src.models.clause_extractor import ClauseExtractor
from src.services.command import Command

class CommandProcessor:
    def __init__(self, commands, threshold=0.8, embedder=None, entity_extractor=None, clause_extractor=None):
        self.embedder = embedder if embedder else SentenceEmbedder()
        self.entity_extractor = entity_extractor if entity_extractor else SpacyEntityExtractor()
        self.clause_extractor = clause_extractor if clause_extractor else ClauseExtractor()
        self.commands = commands
        self.command_embeddings = [self.embedder.encode([cmd["command"]]).squeeze() for cmd in commands]
        self.threshold = threshold

    def preprocess_input(self, text):
        entities = self.entity_extractor.extract_entities(text)
        clauses = self.clause_extractor.extract_clauses(text)

        preprocessed_text = text

        # Replace specific entities with stubs
        for entity, label in entities:
            if label == "PERSON":
                preprocessed_text = preprocessed_text.replace(entity, "John Doe")
            elif label == "TIME":
                preprocessed_text = preprocessed_text.replace(entity, "X length")
            elif label == "DATE":
                preprocessed_text = preprocessed_text.replace(entity, "X date")
            elif label == "CARDINAL":
                preprocessed_text = preprocessed_text.replace(entity, "X")

        # Replace only ccomp clauses with a specific stub
        for clause, dep in clauses:
            if dep == "ccomp":
                preprocessed_text = preprocessed_text.replace(clause, "This is a ccomp clause")

        return preprocessed_text, entities, clauses

    def find_closest_command(self, user_input):
        preprocessed_input, entities, clauses = self.preprocess_input(user_input)
        user_embedding = self.embedder.encode([preprocessed_input]).squeeze()  # Ensure it's 1-D
        similarities = [1 - cosine(user_embedding, cmd_emb) for cmd_emb in self.command_embeddings]
        max_similarity = max(similarities)
        best_match = self.commands[similarities.index(max_similarity)]

        if max_similarity > self.threshold:
            return Command(user_input, preprocessed_input, best_match["command"], entities, clauses)
        else:
            return Command(user_input, preprocessed_input, "Command not recognized", entities, clauses)
