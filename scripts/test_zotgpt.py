"""
Quick test script to verify ZotGPT API connection and configuration.
Run this before labeling the full dataset.
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import requests

sys.path.append(str(Path(__file__).parent.parent))

load_dotenv()

def test_zotgpt_connection():
    print("ZotGPT API Connection Test")
    print("="*80)
    
    api_key = os.getenv('ZOTGPT_API_KEY')
    base_url = os.getenv('ZOTGPT_BASE_URL')
    
    if not api_key:
        print("ZOTGPT_API_KEY not found in .env file")
        print("\nPlease add your ZotGPT API key to .env:")
        return False
    
    print(f"API Key found")
    print(f"Base URL: {base_url}")
    
    print("\nTesting API connection...")
    
    headers = {
        'Content-Type': 'application/json',
        'api-key': api_key,
        'Cache-Control': 'no-cache'
    }
    
    test_payload = {
        "messages": [
            {
                "role": "user",
                "content": "Say 'Hello' if you can read this."
            }
        ],
        "temperature": 0.0,
        "max_tokens": 10
    }
    
    try:
        response = requests.post(
            base_url,
            headers=headers,
            json=test_payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            message = result['choices'][0]['message']['content']
            print(f"API Response: {message}")
            print("\nConnection successful! You're ready to label articles.")
            return True
        else:
            print(f"API Error: {response.status_code}")
            print(f"Response: {response.text}")
            
            if response.status_code == 401:
                print("\nAuthentication failed. Please check your API key.")
            elif response.status_code == 404:
                print("\nEndpoint not found. Please check ZOTGPT_BASE_URL.")
            
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"Connection error: {str(e)}")
        return False


def test_article_data():
    print("\n" + "="*80)
    print("Article Data Check")
    print("="*80)
    
    base_dir = Path(__file__).parent.parent
    data_file = base_dir / "data" / "raw" / "guardian_articles_20260210_224419.json"
    
    if not data_file.exists():
        print(f"Data file not found: {data_file}")
        return False
    
    print(f"Data file found: {data_file}")
    
    try:
        import json
        with open(data_file, 'r', encoding='utf-8') as f:
            articles = json.load(f)
        
        print(f"Loaded {len(articles)} articles")
        
        if len(articles) > 0:
            sample = articles[0]
            print(f"\nSample article:")
            print(f"  ID: {sample.get('id', 'N/A')}")
            print(f"  Title: {sample.get('webTitle', 'N/A')[:60]}...")
            print(f"  Type: {sample.get('type', 'N/A')}")
            has_body = 'bodyText' in sample.get('fields', {})
            print(f"  Has body text: {has_body}")
        
        return True
        
    except Exception as e:
        print(f"Error reading data file: {str(e)}")
        return False


def main():
    """Run all tests."""
    print("\nTesting ZotGPT Labeling Setup\n")
    
    api_ok = test_zotgpt_connection()
    data_ok = test_article_data()
    
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    
    if api_ok and data_ok:
        print("All checks passed! You can now run:")
    else:
        print("Some checks failed. Please fix the issues above.")


if __name__ == "__main__":
    main()
