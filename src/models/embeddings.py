# from transformers import AutoModel, AutoTokenizer
# import torch

# class SentenceEmbedder:
#     def __init__(self):
#         self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")
#         self.model = AutoModel.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")

#     def encode(self, sentences):
#         # You can adjust 'max_length' based on what you expect your typical input size to be
#         encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt', max_length=128)
#         with torch.no_grad():
#             model_output = self.model(**encoded_input)
#         embeddings = model_output.last_hidden_state.mean(dim=1)
#         return embeddings

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
