from src.models.embeddings import SentenceEmbedder
from scipy.spatial.distance import cosine
import numpy as np
from src.models.entity_extractor import SpacyEntityExtractor
from src.models.clause_extractor import ClauseExtractor

class CommandProcessor:
    def __init__(self, commands, threshold=0.8, embedder=None, entity_extractor=None, clause_extractor=None):
        self.embedder = embedder if embedder else SentenceEmbedder()
        self.entity_extractor = entity_extractor if entity_extractor else SpacyEntityExtractor()
        self.clause_extractor = clause_extractor if clause_extractor else ClauseExtractor()
        self.commands = commands
        self.command_embeddings = [self.embedder.encode([cmd["command"]]).squeeze() for cmd in commands]
        self.threshold = threshold

    def find_closest_command(self, user_input):
        user_embedding = self.embedder.encode([user_input]).squeeze()  # Ensure it's 1-D
        similarities = [1 - cosine(user_embedding, cmd_emb) for cmd_emb in self.command_embeddings]
        max_similarity = max(similarities)
        best_match = self.commands[similarities.index(max_similarity)]

        if max_similarity > self.threshold:
            if best_match["requires_extraction"]:
                entities = self.entity_extractor.extract_entities(user_input)
                clauses = self.clause_extractor.extract_clauses(user_input)
                return best_match["command"], entities, clauses
            else:
                return best_match["command"], None, None
        else:
            return "Command not recognized", None, None
