# from src.models.embeddings import SentenceEmbedder
# from scipy.spatial.distance import cosine
# import numpy as np

# class CommandProcessor:
#     def __init__(self, commands, threshold=0.4):
#         self.embedder = SentenceEmbedder()
#         self.commands = commands
#         self.command_embeddings = [self.embedder.encode([cmd]).squeeze() for cmd in commands]
#         self.threshold = threshold

#     def find_closest_command(self, user_input):
#         user_embedding = self.embedder.encode([user_input]).squeeze()  # Ensure it's 1-D
#         similarities = [1 - cosine(user_embedding, cmd_emb) for cmd_emb in self.command_embeddings]
#         max_similarity = max(similarities)
#         best_match = self.commands[similarities.index(max_similarity)]
#         return best_match if max_similarity > self.threshold else "Command not recognized"
from src.models.embeddings import SentenceEmbedder
from scipy.spatial.distance import cosine
import numpy as np

class CommandProcessor:
    def __init__(self, commands, threshold=0.85, embedder=None):
        self.embedder = embedder if embedder else SentenceEmbedder()
        self.commands = commands
        self.command_embeddings = [self.embedder.encode([cmd]).squeeze() for cmd in commands]
        self.threshold = threshold

    def find_closest_command(self, user_input):
        user_embedding = self.embedder.encode([user_input]).squeeze()  # Ensure it's 1-D
        similarities = [1 - cosine(user_embedding, cmd_emb) for cmd_emb in self.command_embeddings]
        max_similarity = max(similarities)
        best_match = self.commands[similarities.index(max_similarity)]
        # if __debug__:
        #     print(f"Input: {user_input}, Similarities: {similarities}, Max Similarity: {max_similarity}")
        return best_match if max_similarity > self.threshold else "Command not recognized"
