"""
Script to collect articles from The Guardian API.
"""
import requests
import json
import time
import os
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration constants
GUARDIAN_API_KEY = os.getenv('GUARDIAN_API_KEY')
GUARDIAN_BASE_URL = "https://content.guardianapis.com"
GUARDIAN_SECTION = "business"
GUARDIAN_TAGS = "business/stock-markets"
GUARDIAN_PAGE_SIZE = 50

# Directory paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


class GuardianAPICollector:
    """Collect articles from The Guardian API."""
    
    def __init__(self, api_key):
        if not api_key:
            raise ValueError("Guardian API key is required. Please set GUARDIAN_API_KEY in .env file")
        self.api_key = api_key
        self.base_url = GUARDIAN_BASE_URL
        
    def fetch_articles(self, from_date, to_date, page=1):
        """
        Fetch articles from The Guardian API.
        
        Args:
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            page: Page number
            
        Returns:
            JSON response from API
        """
        endpoint = f"{self.base_url}/search"
        
        params = {
            "api-key": self.api_key,
            "section": GUARDIAN_SECTION,
            "tag": GUARDIAN_TAGS,
            "from-date": from_date,
            "to-date": to_date,
            "page": page,
            "page-size": GUARDIAN_PAGE_SIZE,
            "show-fields": "headline,bodyText,standfirst",
            "order-by": "newest"
        }
        
        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching articles: {e}")
            return None
    
    def collect_articles(self, from_date, to_date, max_articles=1000):
        """
        Collect multiple pages of articles.
        
        Args:
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            max_articles: Maximum number of articles to collect
            
        Returns:
            List of articles
        """
        articles = []
        page = 1
        
        print(f"Collecting articles from {from_date} to {to_date}...")
        
        while len(articles) < max_articles:
            print(f"Fetching page {page}...")
            response = self.fetch_articles(from_date, to_date, page)
            
            if not response or response.get("response", {}).get("status") != "ok":
                print("Failed to fetch articles or no more results")
                break
            
            results = response.get("response", {}).get("results", [])
            
            if not results:
                print("No more articles found")
                break
            
            articles.extend(results)
            
            total_pages = response.get("response", {}).get("pages", 1)
            print(f"Collected {len(articles)} articles (Page {page}/{total_pages})")
            
            if page >= total_pages:
                break
            
            page += 1
            time.sleep(0.5)  # Rate limiting
        
        return articles[:max_articles]
    
    def save_articles(self, articles, filename):
        """Save articles to JSON file."""
        filepath = RAW_DATA_DIR / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(articles, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(articles)} articles to {filepath}")


def main():
    """Main function to collect articles."""
    # Initialize collector
    collector = GuardianAPICollector(GUARDIAN_API_KEY)
    
    # Set date range (last 2 years)
    to_date = datetime.now()
    from_date = to_date - timedelta(days=730)
    
    from_date_str = from_date.strftime("%Y-%m-%d")
    to_date_str = to_date.strftime("%Y-%m-%d")
    
    # Collect articles
    articles = collector.collect_articles(
        from_date=from_date_str,
        to_date=to_date_str,
        max_articles=2000
    )
    
    # Save articles
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"guardian_articles_{timestamp}.json"
    collector.save_articles(articles, filename)
    
    print(f"\nCollection complete! Total articles: {len(articles)}")


if __name__ == "__main__":
    main()
