import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
from lcr.retrieval.chroma_retriever import ChromaRetriever

class TestChromaRetriever(unittest.TestCase):
    @patch("chromadb.PersistentClient")
    @patch("lcr.retrieval.chroma_retriever.SentenceTransformerEmbeddingFunction")
    def test_retrieve_n_results_safety(self, mock_embed, mock_client):
        # Setup mock collection
        mock_collection = MagicMock()
        mock_collection.count.return_value = 10
        mock_collection.query.return_value = {"documents": [], "metadatas": [], "distances": []}
        
        mock_client.return_value.get_or_create_collection.return_value = mock_collection
        
        retriever = ChromaRetriever(persist_dir=Path("/tmp/fake_chroma"))
        
        # Test with n_results > total count
        retriever.retrieve(question="test", n_results=100)
        
        # Verify n_results was capped to 10
        mock_collection.query.assert_called_once()
        args, kwargs = mock_collection.query.call_args
        self.assertEqual(kwargs["n_results"], 10)

    @patch("chromadb.PersistentClient")
    @patch("lcr.retrieval.chroma_retriever.SentenceTransformerEmbeddingFunction")
    def test_retrieve_n_results_safety_zero(self, mock_embed, mock_client):
        # Setup mock collection with 0 items
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_collection.query.return_value = {"documents": [], "metadatas": [], "distances": []}
        
        mock_client.return_value.get_or_create_collection.return_value = mock_collection
        
        retriever = ChromaRetriever(persist_dir=Path("/tmp/fake_chroma"))
        
        # Test with n_results = 5
        retriever.retrieve(question="test", n_results=5)
        
        # Verify n_results was set to 1 (max(1, total) where total=0)
        mock_collection.query.assert_called_once()
        args, kwargs = mock_collection.query.call_args
        self.assertEqual(kwargs["n_results"], 1)

if __name__ == "__main__":
    unittest.main()
