from sentence_transformers import SentenceTransformer

class SentenceEmbedder:
    def __init__(self):
        # Load the MiniLM model specifically designed for paraphrasing tasks
        self.model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')

    def encode(self, sentences):
        """
        Encode sentences into embeddings using the SentenceTransformer model.
        
        Args:
            sentences (list of str): A list of sentences to encode.

        Returns:
            numpy.ndarray: The encoded embeddings.
        """
        # The encode method of SentenceTransformer handles everything
        embeddings = self.model.encode(sentences)
        return embeddings
