from src.models.embeddings import SentenceEmbedder
from scipy.spatial.distance import cosine

class CommandProcessor:
    def __init__(self, commands):
        self.embedder = SentenceEmbedder()
        self.commands = commands
        self.command_embeddings = self.embedder.encode(commands)

    def find_closest_command(self, user_input):
        user_embedding = self.embedder.encode([user_input]).squeeze()  # Ensure it's 1-D
        similarities = [1 - cosine(user_embedding, cmd_emb.squeeze()) for cmd_emb in self.command_embeddings]  # Also ensure command embeddings are 1-D
        max_similarity = max(similarities)
        best_match = self.commands[similarities.index(max_similarity)]
        return best_match if max_similarity > 0.4 else "Command not recognized"
