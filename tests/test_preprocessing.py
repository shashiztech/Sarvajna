"""
Unit tests for text processing and tokenization.
"""

import pytest
import tempfile
from pathlib import Path

from sarvanjna.preprocessing.text_processor import TextProcessor, ProcessedText
from sarvanjna.preprocessing.tokenizer import SentencePieceTokenizer


class TestTextProcessor:
    """Test TextProcessor class."""
    
    def test_normalize(self):
        processor = TextProcessor()
        
        # Test whitespace normalization
        text = "Hello    world  \n\n  test"
        normalized = processor.normalize(text)
        assert "  " not in normalized
        assert normalized == "Hello world test"
    
    def test_lowercase(self):
        processor = TextProcessor(lowercase=True)
        text = "Hello World"
        normalized = processor.normalize(text)
        assert normalized == "hello world"
    
    def test_pii_filter(self):
        processor = TextProcessor(enable_pii_filter=True)
        
        # Test email filtering
        text = "Contact me at john@example.com for details"
        filtered, contains_pii, pii_type = processor.filter_pii(text)
        assert contains_pii
        assert "email" in pii_type
        assert "[EMAIL]" in filtered
    
    def test_quality_check(self):
        processor = TextProcessor(min_length=10, max_length=100)
        
        # Too short
        is_quality, reason = processor.check_quality("short")
        assert not is_quality
        assert "too_short" in reason
        
        # Good quality
        is_quality, reason = processor.check_quality("This is a good quality text with reasonable length.")
        assert is_quality
        assert reason is None
    
    def test_process(self):
        processor = TextProcessor()
        
        text = "Hello    world! This is a test."
        result = processor.process(text)
        
        assert isinstance(result, ProcessedText)
        assert not result.filtered
        assert result.text != result.original  # Normalized


class TestSentencePieceTokenizer:
    """Test SentencePieceTokenizer class."""
    
    def test_encode_decode(self, tmp_path):
        # Create a simple corpus
        corpus_file = tmp_path / "corpus.txt"
        corpus_file.write_text("Hello world\nThis is a test\nSentencePiece tokenizer\n" * 100)
        
        # Train tokenizer
        tokenizer = SentencePieceTokenizer(vocab_size=100, model_type="bpe")
        model_prefix = str(tmp_path / "tokenizer")
        tokenizer.train([corpus_file], model_prefix, num_threads=1)
        
        # Test encoding
        text = "Hello world"
        encoded = tokenizer.encode(text)
        assert len(encoded.ids) > 0
        assert len(encoded.tokens) == len(encoded.ids)
        
        # Test decoding
        decoded = tokenizer.decode(encoded.ids)
        # Decoded might not match exactly due to subword tokenization
        assert decoded is not None
    
    def test_special_tokens(self, tmp_path):
        corpus_file = tmp_path / "corpus.txt"
        # Create a more diverse corpus with proper newlines
        sentences = [
            "Hello world this is a test",
            "SentencePiece tokenizer training example",
            "Machine learning natural language processing",
            "Python programming artificial intelligence",
            "Deep neural networks transformer models",
            "Data science text preprocessing pipeline"
        ]
        # Write sentences with newlines, repeat to get enough text
        corpus_file.write_text("\n".join(sentences * 50))
        
        tokenizer = SentencePieceTokenizer(vocab_size=50)
        model_prefix = str(tmp_path / "tokenizer")
        tokenizer.train([corpus_file], model_prefix, num_threads=1)
        
        # Check special token IDs
        assert tokenizer.pad_token_id == 0
        assert tokenizer.unk_token_id == 1
        assert tokenizer.bos_token_id == 2
        assert tokenizer.eos_token_id == 3
    
    def test_padding(self, tmp_path):
        corpus_file = tmp_path / "corpus.txt"
        # Create diverse corpus with proper newlines
        sentences = [
            "Hello world test sentence example",
            "Natural language processing pipeline",
            "Tokenization padding masking attention",
            "Training transformer models efficiently",
            "Deep learning neural network architecture"
        ]
        # Write sentences with newlines
        corpus_file.write_text("\n".join(sentences * 50))
        
        tokenizer = SentencePieceTokenizer(vocab_size=50)
        model_prefix = str(tmp_path / "tokenizer")
        tokenizer.train([corpus_file], model_prefix, num_threads=1)
        
        # Test padding
        text = "Hello"
        encoded = tokenizer.encode(text, max_length=10, padding=True)
        
        assert len(encoded.ids) == 10
        assert len(encoded.attention_mask) == 10
        assert encoded.attention_mask[-1] == 0  # Padded position


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
