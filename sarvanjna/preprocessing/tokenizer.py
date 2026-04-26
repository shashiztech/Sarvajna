"""
SentencePiece tokenizer implementation.

Based on: https://arxiv.org/abs/1808.06226
Supports both BPE and Unigram Language Model tokenization.
"""

from typing import List, Optional, Union, Dict, Any
from pathlib import Path
import sentencepiece as spm
from dataclasses import dataclass


@dataclass
class TokenizedOutput:
    """Output from tokenization."""
    ids: List[int]
    tokens: List[str]
    attention_mask: List[int]
    

class SentencePieceTokenizer:
    """
    SentencePiece tokenizer wrapper.
    
    Supports:
    - BPE (byte-pair encoding)
    - Unigram language model
    - Language-independent tokenization
    - Direct training from raw text
    """
    
    # Special tokens
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    BOS_TOKEN = "<s>"
    EOS_TOKEN = "</s>"
    MASK_TOKEN = "<mask>"
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        vocab_size: int = 32000,
        model_type: str = "unigram",  # or "bpe"
        normalization: str = "nmt_nfkc",
    ):
        """
        Initialize tokenizer.
        
        Args:
            model_path: Path to trained SentencePiece model
            vocab_size: Vocabulary size (used during training)
            model_type: "unigram" or "bpe"
            normalization: Normalization method
        """
        self.model_path = Path(model_path) if model_path else None
        self.vocab_size = vocab_size
        self.model_type = model_type
        self.normalization = normalization
        
        self.sp_model: Optional[spm.SentencePieceProcessor] = None
        
        if self.model_path and self.model_path.exists():
            self.load(self.model_path)
    
    def train(
        self,
        input_files: List[Path],
        model_prefix: str,
        character_coverage: float = 0.9995,
        num_threads: int = 16,
        **kwargs,
    ):
        """
        Train SentencePiece model from raw text files.
        
        Args:
            input_files: List of text files to train on
            model_prefix: Prefix for output model files
            character_coverage: Character coverage (0.9995 for rich alphabets)
            num_threads: Number of training threads
            **kwargs: Additional sentencepiece training arguments
        """
        input_str = ",".join(str(f) for f in input_files)
        
        # Training arguments
        train_args = {
            "input": input_str,
            "model_prefix": model_prefix,
            "model_type": self.model_type,
            "vocab_size": self.vocab_size,
            "character_coverage": character_coverage,
            "normalization_rule_name": self.normalization,
            "num_threads": num_threads,
            # Special tokens
            "pad_id": 0,
            "unk_id": 1,
            "bos_id": 2,
            "eos_id": 3,
            "pad_piece": self.PAD_TOKEN,
            "unk_piece": self.UNK_TOKEN,
            "bos_piece": self.BOS_TOKEN,
            "eos_piece": self.EOS_TOKEN,
            # Additional control tokens
            "control_symbols": [self.MASK_TOKEN],
        }
        
        # Merge with user-provided kwargs
        train_args.update(kwargs)
        
        # Train model
        spm.SentencePieceTrainer.train(**train_args)
        
        # Load the trained model
        self.model_path = Path(f"{model_prefix}.model")
        self.load(self.model_path)
        
        print(f"✓ Trained SentencePiece model: {self.model_path}")
        print(f"  Model type: {self.model_type}")
        print(f"  Vocab size: {self.get_vocab_size()}")
    
    def load(self, model_path: Path):
        """Load a trained SentencePiece model."""
        self.model_path = Path(model_path)
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(str(self.model_path))
        print(f"✓ Loaded SentencePiece model: {model_path}")
    
    def save(self, save_path: Path):
        """Save the current model to a new location."""
        if not self.sp_model:
            raise RuntimeError("No model loaded")
        
        import shutil
        shutil.copy2(self.model_path, save_path)
        print(f"✓ Saved model to: {save_path}")
    
    def encode(
        self,
        text: Union[str, List[str]],
        max_length: Optional[int] = None,
        padding: bool = False,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> Union[TokenizedOutput, List[TokenizedOutput]]:
        """
        Encode text to token IDs.
        
        Args:
            text: Text or list of texts to encode
            max_length: Maximum sequence length
            padding: Whether to pad to max_length
            add_bos: Add BOS token
            add_eos: Add EOS token
            
        Returns:
            TokenizedOutput or list of TokenizedOutput
        """
        if not self.sp_model:
            raise RuntimeError("No model loaded. Train or load a model first.")
        
        # Handle single string
        is_single = isinstance(text, str)
        texts = [text] if is_single else text
        
        results = []
        for txt in texts:
            # Encode to IDs
            ids = self.sp_model.encode(txt)
            
            # Add special tokens
            if add_bos:
                ids = [self.bos_token_id] + ids
            if add_eos:
                ids = ids + [self.eos_token_id]
            
            # Truncate if needed
            if max_length and len(ids) > max_length:
                ids = ids[:max_length]
                if add_eos:
                    ids[-1] = self.eos_token_id
            
            # Padding
            attention_mask = [1] * len(ids)
            if padding and max_length:
                pad_length = max_length - len(ids)
                ids = ids + [self.pad_token_id] * pad_length
                attention_mask = attention_mask + [0] * pad_length
            
            # Get tokens
            tokens = self.sp_model.id_to_piece(ids)
            
            results.append(TokenizedOutput(
                ids=ids,
                tokens=tokens,
                attention_mask=attention_mask,
            ))
        
        return results[0] if is_single else results
    
    def decode(
        self,
        ids: Union[List[int], List[List[int]]],
        skip_special_tokens: bool = True,
    ) -> Union[str, List[str]]:
        """
        Decode token IDs to text.
        
        Args:
            ids: Token IDs or batch of token IDs
            skip_special_tokens: Skip special tokens in output
            
        Returns:
            Decoded text or list of texts
        """
        if not self.sp_model:
            raise RuntimeError("No model loaded")
        
        # Handle single sequence
        is_single = isinstance(ids[0], int)
        id_sequences = [ids] if is_single else ids
        
        results = []
        for id_seq in id_sequences:
            # Filter special tokens if requested
            if skip_special_tokens:
                id_seq = [
                    id for id in id_seq
                    if id not in [self.pad_token_id, self.bos_token_id, self.eos_token_id]
                ]
            
            # Decode
            text = self.sp_model.decode(id_seq)
            results.append(text)
        
        return results[0] if is_single else results
    
    @property
    def pad_token_id(self) -> int:
        return self.sp_model.pad_id() if self.sp_model else 0
    
    @property
    def unk_token_id(self) -> int:
        return self.sp_model.unk_id() if self.sp_model else 1
    
    @property
    def bos_token_id(self) -> int:
        return self.sp_model.bos_id() if self.sp_model else 2
    
    @property
    def eos_token_id(self) -> int:
        return self.sp_model.eos_id() if self.sp_model else 3
    
    @property
    def mask_token_id(self) -> int:
        """Get ID for mask token."""
        if self.sp_model:
            return self.sp_model.piece_to_id(self.MASK_TOKEN)
        return 4
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.sp_model) if self.sp_model else self.vocab_size
    
    def get_vocab(self) -> Dict[str, int]:
        """Get full vocabulary as dict."""
        if not self.sp_model:
            return {}
        
        vocab = {}
        for i in range(len(self.sp_model)):
            piece = self.sp_model.id_to_piece(i)
            vocab[piece] = i
        
        return vocab
    
    def token_to_id(self, token: str) -> int:
        """Convert token to ID."""
        if not self.sp_model:
            raise RuntimeError("No model loaded")
        return self.sp_model.piece_to_id(token)
    
    def id_to_token(self, id: int) -> str:
        """Convert ID to token."""
        if not self.sp_model:
            raise RuntimeError("No model loaded")
        return self.sp_model.id_to_piece(id)
