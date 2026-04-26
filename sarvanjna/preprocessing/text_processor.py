"""
Text processor with normalization, filtering, and moderation.
"""

from typing import List, Optional, Dict, Any, Callable
from pathlib import Path
import re
import unicodedata
from dataclasses import dataclass


@dataclass
class ProcessedText:
    """Output from text processing."""
    text: str
    original: str
    filtered: bool
    filter_reason: Optional[str] = None
    metadata: Dict[str, Any] = None


class TextProcessor:
    """
    Text processing pipeline with normalization, filtering, and quality checks.
    
    Implements:
    - Unicode normalization
    - Whitespace cleanup
    - Language detection
    - PII filtering
    - Content moderation
    - Quality scoring
    """
    
    def __init__(
        self,
        normalization: str = "NFKC",
        lowercase: bool = False,
        remove_accents: bool = False,
        min_length: int = 10,
        max_length: int = 10000,
        enable_pii_filter: bool = True,
        enable_quality_filter: bool = True,
    ):
        """
        Initialize text processor.
        
        Args:
            normalization: Unicode normalization form (NFC, NFD, NFKC, NFKD)
            lowercase: Convert to lowercase
            remove_accents: Remove accent marks
            min_length: Minimum text length
            max_length: Maximum text length
            enable_pii_filter: Enable PII filtering
            enable_quality_filter: Enable quality filtering
        """
        self.normalization = normalization
        self.lowercase = lowercase
        self.remove_accents = remove_accents
        self.min_length = min_length
        self.max_length = max_length
        self.enable_pii_filter = enable_pii_filter
        self.enable_quality_filter = enable_quality_filter
        
        # Compile regex patterns
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for filtering."""
        # Email pattern
        self.email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        )
        
        # Phone number pattern (simple)
        self.phone_pattern = re.compile(
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        )
        
        # URL pattern
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        
        # Multiple whitespace
        self.whitespace_pattern = re.compile(r'\s+')
        
        # Multiple punctuation
        self.punct_pattern = re.compile(r'([.!?])\1{2,}')
    
    def normalize(self, text: str) -> str:
        """
        Normalize text.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        # Unicode normalization
        text = unicodedata.normalize(self.normalization, text)
        
        # Remove accents if requested
        if self.remove_accents:
            text = ''.join(
                c for c in unicodedata.normalize('NFD', text)
                if unicodedata.category(c) != 'Mn'
            )
        
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        # Clean whitespace
        text = self.whitespace_pattern.sub(' ', text)
        text = text.strip()
        
        # Clean excessive punctuation
        text = self.punct_pattern.sub(r'\1', text)
        
        return text
    
    def filter_pii(self, text: str) -> tuple[str, bool, Optional[str]]:
        """
        Filter or mask PII (Personally Identifiable Information).
        
        Args:
            text: Input text
            
        Returns:
            (filtered_text, contains_pii, pii_type)
        """
        if not self.enable_pii_filter:
            return text, False, None
        
        original_text = text
        contains_pii = False
        pii_types = []
        
        # Mask emails
        if self.email_pattern.search(text):
            text = self.email_pattern.sub('[EMAIL]', text)
            contains_pii = True
            pii_types.append('email')
        
        # Mask phone numbers
        if self.phone_pattern.search(text):
            text = self.phone_pattern.sub('[PHONE]', text)
            contains_pii = True
            pii_types.append('phone')
        
        pii_type = ','.join(pii_types) if pii_types else None
        
        return text, contains_pii, pii_type
    
    def check_quality(self, text: str) -> tuple[bool, Optional[str]]:
        """
        Check text quality.
        
        Args:
            text: Input text
            
        Returns:
            (is_quality, reason_if_not)
        """
        if not self.enable_quality_filter:
            return True, None
        
        # Length check
        if len(text) < self.min_length:
            return False, f"too_short ({len(text)} < {self.min_length})"
        
        if len(text) > self.max_length:
            return False, f"too_long ({len(text)} > {self.max_length})"
        
        # Word count check
        words = text.split()
        if len(words) < 3:
            return False, "too_few_words"
        
        # Alphabet ratio check (must have some alphabetic content)
        alpha_chars = sum(c.isalpha() for c in text)
        alpha_ratio = alpha_chars / len(text) if text else 0
        
        if alpha_ratio < 0.5:
            return False, f"low_alpha_ratio ({alpha_ratio:.2f})"
        
        # Repetition check (detect spam/junk)
        unique_words = set(words)
        if len(words) > 10 and len(unique_words) / len(words) < 0.3:
            return False, "high_repetition"
        
        return True, None
    
    def process(
        self,
        text: str,
        skip_normalization: bool = False,
    ) -> ProcessedText:
        """
        Process text through full pipeline.
        
        Args:
            text: Input text
            skip_normalization: Skip normalization step
            
        Returns:
            ProcessedText with results and metadata
        """
        original = text
        
        # Normalize
        if not skip_normalization:
            text = self.normalize(text)
        
        # Filter PII
        text, contains_pii, pii_type = self.filter_pii(text)
        
        # Quality check
        is_quality, quality_reason = self.check_quality(text)
        
        # Determine if filtered
        filtered = not is_quality
        filter_reason = quality_reason if not is_quality else None
        
        return ProcessedText(
            text=text,
            original=original,
            filtered=filtered,
            filter_reason=filter_reason,
            metadata={
                "contains_pii": contains_pii,
                "pii_type": pii_type,
                "length": len(text),
                "word_count": len(text.split()),
            }
        )
    
    def process_batch(
        self,
        texts: List[str],
        skip_normalization: bool = False,
        return_filtered: bool = False,
    ) -> List[ProcessedText]:
        """
        Process a batch of texts.
        
        Args:
            texts: List of input texts
            skip_normalization: Skip normalization
            return_filtered: Return filtered texts or only valid ones
            
        Returns:
            List of ProcessedText
        """
        results = [self.process(text, skip_normalization) for text in texts]
        
        if not return_filtered:
            results = [r for r in results if not r.filtered]
        
        return results
    
    def filter_dataset(
        self,
        input_path: Path,
        output_path: Path,
        report_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Filter a text dataset file.
        
        Args:
            input_path: Input text file
            output_path: Output filtered file
            report_path: Optional path for filtering report
            
        Returns:
            Statistics dictionary
        """
        stats = {
            "total": 0,
            "kept": 0,
            "filtered": 0,
            "filter_reasons": {},
            "pii_found": 0,
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(input_path, "r", encoding="utf-8") as infile, \
             open(output_path, "w", encoding="utf-8") as outfile:
            
            for line in infile:
                stats["total"] += 1
                line = line.strip()
                
                if not line:
                    continue
                
                result = self.process(line)
                
                if result.metadata.get("contains_pii"):
                    stats["pii_found"] += 1
                
                if not result.filtered:
                    outfile.write(result.text + "\n")
                    stats["kept"] += 1
                else:
                    stats["filtered"] += 1
                    reason = result.filter_reason or "unknown"
                    stats["filter_reasons"][reason] = stats["filter_reasons"].get(reason, 0) + 1
        
        # Write report
        if report_path:
            with open(report_path, "w") as f:
                f.write("Text Filtering Report\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Total texts: {stats['total']}\n")
                f.write(f"Kept: {stats['kept']} ({stats['kept']/stats['total']*100:.1f}%)\n")
                f.write(f"Filtered: {stats['filtered']} ({stats['filtered']/stats['total']*100:.1f}%)\n")
                f.write(f"PII found: {stats['pii_found']}\n\n")
                f.write("Filter reasons:\n")
                for reason, count in stats["filter_reasons"].items():
                    f.write(f"  {reason}: {count}\n")
        
        return stats
