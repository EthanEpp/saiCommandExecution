# tests/test_embeddings_unit.py
import pytest
import numpy as np
from unittest.mock import patch
from src.models.embeddings import SentenceEmbedder

@pytest.fixture
def sentence_embedder():
    return SentenceEmbedder()

def test_encode(sentence_embedder):
    sentences = ["This is a test sentence.", "Another sentence for testing."]
    mock_output = np.random.rand(len(sentences), 384)  # Simulate a 384-dimensional embedding for each sentence

    with patch.object(sentence_embedder.model, 'encode', return_value=mock_output) as mock_encode:
        embeddings = sentence_embedder.encode(sentences)
        
        mock_encode.assert_called_once_with(sentences)
        assert embeddings.shape == (len(sentences), 384)  # Check if the output shape is correct
        assert isinstance(embeddings, np.ndarray)  # Check if the output is a numpy array

def test_get_embedding_dim(sentence_embedder):
    expected_dim = 384  # The known dimension of the embeddings (could make more robust by not hard coding since models can vary)

    with patch.object(sentence_embedder.model, 'get_sentence_embedding_dimension', return_value=expected_dim) as mock_get_dim:
        embedding_dim = sentence_embedder.get_embedding_dim()
        
        mock_get_dim.assert_called_once()
        assert embedding_dim == expected_dim
