"""
Legal data collection automation for Sarvanjna platform.

Collects data from legal sources with proper license tracking.
"""

import os
import time
import json
import logging
from typing import List, Dict, Optional, Generator
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import hashlib

# Optional imports - install as needed
try:
    import requests
except ImportError:
    print("Install: pip install requests")
    requests = None


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataSource:
    """Track data source with license information."""
    name: str
    url: str
    license: str
    license_url: str
    data_type: str  # text, image, audio, video
    collection_date: str
    item_count: int = 0
    hash: Optional[str] = None
    
    def to_dict(self):
        return asdict(self)


class LegalDataCollector:
    """
    Automated data collection from legal sources.
    
    Sources:
    - Wikipedia (CC-BY-SA 3.0)
    - Common Crawl (public web data)
    - Project Gutenberg (public domain)
    - Wikimedia Commons (various CC licenses)
    - Open Images (CC-BY 4.0)
    """
    
    def __init__(self, output_dir: str = "data/legal_sources"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Manifest tracks all collected data
        self.manifest_path = self.output_dir / "data_manifest.json"
        self.manifest = self.load_manifest()
        
        # Rate limiting
        self.request_delay = 1.0  # seconds between requests
    
    def load_manifest(self) -> Dict:
        """Load or create data manifest."""
        if self.manifest_path.exists():
            with open(self.manifest_path, 'r') as f:
                return json.load(f)
        return {
            'version': '1.0',
            'created': datetime.now().isoformat(),
            'sources': [],
            'total_items': 0,
        }
    
    def save_manifest(self):
        """Save data manifest."""
        with open(self.manifest_path, 'w') as f:
            json.dump(self.manifest, f, indent=2)
    
    def add_source_to_manifest(self, source: DataSource):
        """Add data source to manifest."""
        self.manifest['sources'].append(source.to_dict())
        self.manifest['total_items'] += source.item_count
        self.save_manifest()
    
    # ============== Wikipedia ==============
    
    def collect_wikipedia_articles(
        self,
        categories: List[str],
        max_articles: int = 1000,
        language: str = 'en'
    ) -> Generator[Dict, None, None]:
        """
        Collect Wikipedia articles from specified categories.
        
        Args:
            categories: List of Wikipedia category names
            max_articles: Maximum articles to collect
            language: Wikipedia language code
        
        Yields:
            Dict with article data
        """
        if requests is None:
            logger.error("requests library not installed")
            return
        
        logger.info(f"Collecting Wikipedia articles from {len(categories)} categories")
        
        base_url = f"https://{language}.wikipedia.org/w/api.php"
        collected = 0
        
        for category in categories:
            if collected >= max_articles:
                break
            
            params = {
                'action': 'query',
                'list': 'categorymembers',
                'cmtitle': f'Category:{category}',
                'cmlimit': 'max',
                'format': 'json'
            }
            
            try:
                time.sleep(self.request_delay)
                response = requests.get(base_url, params=params, timeout=30)
                data = response.json()
                
                for page in data.get('query', {}).get('categorymembers', []):
                    if collected >= max_articles:
                        break
                    
                    # Get article content
                    article = self.get_wikipedia_article(page['title'], language)
                    if article:
                        yield article
                        collected += 1
                        
                        if collected % 100 == 0:
                            logger.info(f"Collected {collected} articles")
            
            except Exception as e:
                logger.error(f"Error collecting category {category}: {e}")
        
        # Record source
        source = DataSource(
            name="Wikipedia",
            url=f"https://{language}.wikipedia.org",
            license="CC-BY-SA-3.0",
            license_url="https://creativecommons.org/licenses/by-sa/3.0/",
            data_type="text",
            collection_date=datetime.now().isoformat(),
            item_count=collected
        )
        self.add_source_to_manifest(source)
    
    def get_wikipedia_article(self, title: str, language: str = 'en') -> Optional[Dict]:
        """Get single Wikipedia article content."""
        if requests is None:
            return None
        
        base_url = f"https://{language}.wikipedia.org/w/api.php"
        
        params = {
            'action': 'query',
            'titles': title,
            'prop': 'extracts',
            'explaintext': True,
            'format': 'json'
        }
        
        try:
            response = requests.get(base_url, params=params, timeout=30)
            data = response.json()
            
            pages = data.get('query', {}).get('pages', {})
            for page_id, page_data in pages.items():
                if 'extract' in page_data:
                    return {
                        'title': page_data.get('title'),
                        'text': page_data.get('extract'),
                        'url': f"https://{language}.wikipedia.org/wiki/{title.replace(' ', '_')}",
                        'source': 'wikipedia',
                        'license': 'CC-BY-SA-3.0'
                    }
        except Exception as e:
            logger.error(f"Error fetching article {title}: {e}")
        
        return None
    
    # ============== Project Gutenberg ==============
    
    def collect_gutenberg_books(
        self,
        max_books: int = 100,
        start_id: int = 1
    ) -> Generator[Dict, None, None]:
        """
        Collect public domain books from Project Gutenberg.
        
        Args:
            max_books: Maximum books to collect
            start_id: Starting book ID
        
        Yields:
            Dict with book data
        """
        if requests is None:
            logger.error("requests library not installed")
            return
        
        logger.info(f"Collecting Project Gutenberg books")
        
        collected = 0
        book_id = start_id
        max_attempts = max_books * 10  # Try 10x the target
        attempts = 0
        
        while collected < max_books and attempts < max_attempts:
            # Gutenberg book URL
            url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
            
            try:
                time.sleep(self.request_delay)
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    text = response.text
                    
                    # Extract title (first line after header)
                    lines = text.split('\n')
                    title = lines[0] if lines else f"Book {book_id}"
                    
                    yield {
                        'id': book_id,
                        'title': title.strip(),
                        'text': text,
                        'url': url,
                        'source': 'gutenberg',
                        'license': 'public_domain'
                    }
                    
                    collected += 1
                    
                    if collected % 10 == 0:
                        logger.info(f"Collected {collected} books")
            
            except Exception as e:
                logger.debug(f"Book {book_id} not available: {e}")
            
            book_id += 1
            attempts += 1
        
        # Record source
        source = DataSource(
            name="Project Gutenberg",
            url="https://www.gutenberg.org",
            license="Public Domain",
            license_url="https://www.gutenberg.org/policy/license.html",
            data_type="text",
            collection_date=datetime.now().isoformat(),
            item_count=collected
        )
        self.add_source_to_manifest(source)
    
    # ============== Wikimedia Commons ==============
    
    def collect_wikimedia_images(
        self,
        categories: List[str],
        max_images: int = 1000,
        licenses: List[str] = ['CC-BY-4.0', 'CC-BY-SA-4.0', 'CC0']
    ) -> Generator[Dict, None, None]:
        """
        Collect images from Wikimedia Commons with specified licenses.
        
        Args:
            categories: Wikimedia Commons category names
            max_images: Maximum images to collect
            licenses: Allowed licenses
        
        Yields:
            Dict with image data
        """
        if requests is None:
            logger.error("requests library not installed")
            return
        
        logger.info(f"Collecting Wikimedia Commons images")
        
        base_url = "https://commons.wikimedia.org/w/api.php"
        collected = 0
        
        for category in categories:
            if collected >= max_images:
                break
            
            params = {
                'action': 'query',
                'list': 'categorymembers',
                'cmtitle': f'Category:{category}',
                'cmtype': 'file',
                'cmlimit': 'max',
                'format': 'json'
            }
            
            try:
                time.sleep(self.request_delay)
                response = requests.get(base_url, params=params, timeout=30)
                data = response.json()
                
                for file_data in data.get('query', {}).get('categorymembers', []):
                    if collected >= max_images:
                        break
                    
                    # Get image info including license
                    image_info = self.get_wikimedia_image_info(file_data['title'])
                    
                    if image_info and image_info.get('license') in licenses:
                        yield image_info
                        collected += 1
                        
                        if collected % 100 == 0:
                            logger.info(f"Collected {collected} images")
            
            except Exception as e:
                logger.error(f"Error collecting category {category}: {e}")
        
        # Record source
        source = DataSource(
            name="Wikimedia Commons",
            url="https://commons.wikimedia.org",
            license=", ".join(licenses),
            license_url="https://creativecommons.org/licenses/",
            data_type="image",
            collection_date=datetime.now().isoformat(),
            item_count=collected
        )
        self.add_source_to_manifest(source)
    
    def get_wikimedia_image_info(self, filename: str) -> Optional[Dict]:
        """Get Wikimedia Commons image info including license."""
        if requests is None:
            return None
        
        base_url = "https://commons.wikimedia.org/w/api.php"
        
        params = {
            'action': 'query',
            'titles': filename,
            'prop': 'imageinfo',
            'iiprop': 'url|extmetadata',
            'format': 'json'
        }
        
        try:
            response = requests.get(base_url, params=params, timeout=30)
            data = response.json()
            
            pages = data.get('query', {}).get('pages', {})
            for page_id, page_data in pages.items():
                if 'imageinfo' in page_data:
                    info = page_data['imageinfo'][0]
                    metadata = info.get('extmetadata', {})
                    
                    # Extract license
                    license_info = metadata.get('License', {}).get('value', 'Unknown')
                    
                    return {
                        'title': page_data.get('title'),
                        'url': info.get('url'),
                        'license': license_info,
                        'source': 'wikimedia',
                        'metadata': metadata
                    }
        except Exception as e:
            logger.error(f"Error fetching image {filename}: {e}")
        
        return None
    
    # ============== Save Data ==============
    
    def save_text_data(
        self,
        data_generator: Generator[Dict, None, None],
        output_file: str
    ):
        """Save text data to file."""
        output_path = self.output_dir / output_file
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data_generator:
                # Save as JSONL
                f.write(json.dumps(item) + '\n')
        
        logger.info(f"Saved text data to {output_path}")
    
    def save_image_urls(
        self,
        data_generator: Generator[Dict, None, None],
        output_file: str
    ):
        """Save image URLs and metadata."""
        output_path = self.output_dir / output_file
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data_generator:
                f.write(json.dumps(item) + '\n')
        
        logger.info(f"Saved image URLs to {output_path}")


def main():
    """Example usage."""
    collector = LegalDataCollector()
    
    # Collect Wikipedia articles
    logger.info("=== Collecting Wikipedia Articles ===")
    wiki_categories = [
        'Machine_learning',
        'Artificial_intelligence',
        'Computer_science',
    ]
    
    wiki_data = collector.collect_wikipedia_articles(
        categories=wiki_categories,
        max_articles=100
    )
    collector.save_text_data(wiki_data, 'wikipedia_articles.jsonl')
    
    # Collect Gutenberg books
    logger.info("=== Collecting Gutenberg Books ===")
    books_data = collector.collect_gutenberg_books(max_books=50)
    collector.save_text_data(books_data, 'gutenberg_books.jsonl')
    
    # Collect Wikimedia images
    logger.info("=== Collecting Wikimedia Images ===")
    image_categories = [
        'Featured_pictures',
        'Nature',
        'Architecture',
    ]
    
    images_data = collector.collect_wikimedia_images(
        categories=image_categories,
        max_images=100
    )
    collector.save_image_urls(images_data, 'wikimedia_images.jsonl')
    
    logger.info("=== Data Collection Complete ===")
    logger.info(f"Manifest: {collector.manifest_path}")


if __name__ == '__main__':
    main()
