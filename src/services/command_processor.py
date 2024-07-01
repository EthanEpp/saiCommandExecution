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

        # Replace only ccomp clauses with a specific stub (this will be replaced with search query/specific message on fine-tuned model)
        for clause, dep in clauses:
            if dep == "ccomp":
                preprocessed_text = preprocessed_text.replace(clause, "This is a ccomp clause")

        # Replace specific entities with stubs Add check for fine not including fine tuned search query/message portions
        for entity, label in entities:
            if label == "PERSON":
                preprocessed_text = preprocessed_text.replace(entity, "John Doe")
            elif label == "TIME":
                preprocessed_text = preprocessed_text.replace(entity, "X length")
            elif label == "DATE":
                preprocessed_text = preprocessed_text.replace(entity, "X date")
            elif label == "CARDINAL":
                preprocessed_text = preprocessed_text.replace(entity, "X")



        return preprocessed_text, entities, clauses

    def find_closest_command(self, user_input):
        preprocessed_input, entities, clauses = self.preprocess_input(user_input)
        user_embedding = self.embedder.encode([preprocessed_input]).squeeze()  # Ensure it's 1-D
        similarities = [1 - cosine(user_embedding, cmd_emb) for cmd_emb in self.command_embeddings]
        max_similarity = max(similarities)
        best_match = self.commands[similarities.index(max_similarity)]

        if max_similarity > self.threshold:
            # Filter entities and clauses based on the best match's specifications
            specified_entities = best_match.get("entities", [])
            specified_clauses = best_match.get("clauses", [])

            filtered_entities = [(entity, label) for entity, label in entities if label in specified_entities]
            filtered_clauses = [(clause, dep) for clause, dep in clauses if dep in specified_clauses]

            return Command(
                user_input,
                preprocessed_input,
                best_match["command"],
                filtered_entities,
                filtered_clauses
            )
        else:
            return self.find_closest_command_raw(user_input)

    def find_closest_command_raw(self, user_input):
        print("Preprocess not found, attempting raw.")
        _, entities, clauses = self.preprocess_input(user_input)
        user_embedding = self.embedder.encode([user_input]).squeeze()  # Ensure it's 1-D
        similarities = [1 - cosine(user_embedding, cmd_emb) for cmd_emb in self.command_embeddings]
        max_similarity = max(similarities)
        best_match = self.commands[similarities.index(max_similarity)]

        if max_similarity > self.threshold:
            # Filter entities and clauses based on the best match's specifications
            specified_entities = best_match.get("entities", [])
            specified_clauses = best_match.get("clauses", [])

            filtered_entities = [(entity, label) for entity, label in entities if label in specified_entities]
            filtered_clauses = [(clause, dep) for clause, dep in clauses if dep in specified_clauses]

            return Command(
                user_input,
                "Raw input taken.",
                best_match["command"],
                filtered_entities,
                filtered_clauses
            )
        else:
            return Command(
                user_input,
                _,
                "Command not recognized",
                entities,
                clauses
            )
