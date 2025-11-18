#!/usr/bin/env python3
"""
Data Collection Script - Glassdoor Review Scraper

Production-ready scraper for REIT Glassdoor reviews with defensive error handling,
URL tracking, automatic reviews tab clicking, and flexible configuration.

Author: Konain Niaz (kn4792@rit.edu)
Date: 2025-01-18
Version: 2.1 (Production)

Usage:
    # Start Chrome in debug mode first:
    chrome.exe --remote-debugging-port=9222
    
    # Then run scraper:
    python scripts/scrape_reviews.py
    python scripts/scrape_reviews.py --test
    python scripts/scrape_reviews.py --company PLD
    python scripts/scrape_reviews.py --csv data/reit_companies.csv
"""

import json
import time
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict

import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException


def init_browser(debug_port: int = 9222) -> webdriver.Chrome:
    """
    Initialize Chrome browser connecting to existing debug session.
    
    Args:
        debug_port: Port number for Chrome remote debugging
        
    Returns:
        WebDriver instance connected to Chrome
        
    Raises:
        Exception if connection fails
    """
    print(f"🔧 Connecting to Chrome on port {debug_port}...")
    
    chrome_options = Options()
    chrome_options.add_experimental_option(
        "debuggerAddress", 
        f"127.0.0.1:{debug_port}"
    )
    
    try:
        driver = webdriver.Chrome(options=chrome_options)
        print("✓ Connected to Chrome successfully")
        return driver
    except Exception as e:
        print(f"✗ Failed to connect to Chrome: {e}")
        print("\nPlease start Chrome in debug mode:")
        print("  Windows: chrome.exe --remote-debugging-port=9222")
        print("  macOS: /Applications/Google\\ Chrome.app/Contents/MacOS/Google\\ Chrome --remote-debugging-port=9222")
        print("  Linux: google-chrome --remote-debugging-port=9222")
        raise


def manual_login_wait(driver: webdriver.Chrome):
    """
    Open Glassdoor and wait for user to login manually.
    
    Args:
        driver: WebDriver instance
    """
    print("\n" + "="*60)
    print("⏸️  MANUAL LOGIN REQUIRED")
    print("="*60)
    print("\n📋 Instructions:")
    print("1. A Chrome window should now be open")
    print("2. Login to Glassdoor in that window")
    print("3. Once logged in, return here and press ENTER")
    print("4. Scraping will start automatically")
    print("\n" + "="*60)
    
    driver.get('https://www.glassdoor.com')
    time.sleep(2)
    
    input("\n⏸️  Press ENTER after logging in to Glassdoor... ")
    print("\n✓ Starting scraper...")


def wait_for_reviews_to_load(driver: webdriver.Chrome, timeout: int = 10) -> bool:
    """
    Wait for reviews list to load on page.
    
    Uses the correct CSS selector: ol.ReviewsList_reviewsList__Qfw6M
    Scrolls to trigger lazy loading.
    
    Args:
        driver: WebDriver instance
        timeout: Maximum wait time in seconds
        
    Returns:
        True if reviews loaded, False otherwise
    """
    print("    ⏳ Waiting for reviews to load...")
    
    # Scroll to trigger lazy loading
    for i in range(3):
        scroll_position = (i + 1) / 3
        driver.execute_script(
            f"window.scrollTo(0, document.body.scrollHeight * {scroll_position});"
        )
        time.sleep(1)
    
    # Wait for reviews list
    wait = WebDriverWait(driver, timeout)
    
    try:
        reviews_list = wait.until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, 'ol.ReviewsList_reviewsList__Qfw6M')
            )
        )
        print("    ✓ Reviews loaded")
        return True
    except TimeoutException:
        print("    ⚠️ Timeout waiting for reviews")
        return False


def find_review_elements(driver: webdriver.Chrome) -> List:
    """
    Find all review list items on current page.
    
    Args:
        driver: WebDriver instance
        
    Returns:
        List of WebElements representing reviews
    """
    # Try primary selector
    try:
        reviews_list = driver.find_element(
            By.CSS_SELECTOR, 
            'ol.ReviewsList_reviewsList__Qfw6M'
        )
        li_elements = reviews_list.find_elements(By.TAG_NAME, 'li')
        
        if li_elements:
            print(f"    ✓ Found {len(li_elements)} review elements")
            return li_elements
    except NoSuchElementException:
        pass
    
    # Fallback: direct selector
    try:
        li_elements = driver.find_elements(
            By.CSS_SELECTOR,
            'ol.ReviewsList_reviewsList__Qfw6M > li'
        )
        if li_elements:
            print(f"    ✓ Found {len(li_elements)} reviews (fallback)")
            return li_elements
    except Exception:
        pass
    
    print("    ✗ No review elements found")
    return []


def extract_review_data(review_element) -> Dict[str, str]:
    """
    Extract data from a single review element.
    
    Extracts:
    - Title
    - Rating
    - Date
    - Employee info (job title, status)
    - Pros
    - Cons
    
    Args:
        review_element: Selenium WebElement containing review HTML
        
    Returns:
        Dictionary with review data
    """
    data = {}
    
    # Extract title
    title_selectors = [
        'h2',
        'a[href*="Reviews"]',
        'span[class*="ReviewTitle"]',
        'div[class*="ReviewTitle"]'
    ]
    
    for selector in title_selectors:
        try:
            element = review_element.find_element(By.CSS_SELECTOR, selector)
            title_text = element.text.strip()
            if title_text and len(title_text) > 3:
                data['title'] = title_text
                break
        except NoSuchElementException:
            continue
    
    # Extract rating
    rating_selectors = [
    '[data-test="review-rating-label"]',   # <-- add this line as FIRST selector!
    'span[class*="RatingNumber"]',
    'div[class*="RatingNumber"]',
    '[data-test="rating"]'
    ]

    
    for selector in rating_selectors:
        try:
            element = review_element.find_element(By.CSS_SELECTOR, selector)
            rating_text = element.text.strip()
            if rating_text and any(c.isdigit() for c in rating_text):
                data['rating'] = rating_text
                break
        except NoSuchElementException:
            continue
    
    # Extract date
    date_selectors = [
        'span[class*="date" i]',
        'time',
        'span.minor',
        '[data-test="review-date"]'
    ]
    
    for selector in date_selectors:
        try:
            element = review_element.find_element(By.CSS_SELECTOR, selector)
            date_text = element.text.strip()
            if date_text:
                data['date'] = date_text
                break
        except NoSuchElementException:
            continue
    
    # Extract employee info
    employee_selectors = [
        'span[class*="JobTitle" i]',
        'span[class*="employee" i]',
        'div[class*="EmployeeInfo"]'
    ]
    
    for selector in employee_selectors:
        try:
            element = review_element.find_element(By.CSS_SELECTOR, selector)
            info_text = element.text.strip()
            if info_text:
                data['employee_info'] = info_text
                break
        except NoSuchElementException:
            continue
    
    # Extract pros (XPath method)
    try:
        pros_candidates = review_element.find_elements(
            By.XPATH,
            ".//*[contains(text(), 'Pros') or contains(text(), 'pros')]"
        )
        
        for candidate in pros_candidates:
            parent = candidate.find_element(By.XPATH, '..')
            parent_text = parent.text
            
            if 'Pros' in parent_text:
                parts = parent_text.split('Pros', 1)
                if len(parts) > 1:
                    pros_text = parts[1].split('Cons')[0].strip()
                    if pros_text and len(pros_text) > 5:
                        data['pros'] = pros_text
                        break
    except Exception:
        pass
    
    # Extract cons (XPath method)
    try:
        cons_candidates = review_element.find_elements(
            By.XPATH,
            ".//*[contains(text(), 'Cons') or contains(text(), 'cons')]"
        )
        
        for candidate in cons_candidates:
            parent = candidate.find_element(By.XPATH, '..')
            parent_text = parent.text
            
            if 'Cons' in parent_text:
                parts = parent_text.split('Cons', 1)
                if len(parts) > 1:
                    cons_text = parts[1].split('Advice')[0].split('Helpful')[0].strip()
                    if cons_text and len(cons_text) > 5:
                        data['cons'] = cons_text
                        break
    except Exception:
        pass
    
    return data


def try_next_page(driver: webdriver.Chrome) -> bool:
    """
    Attempt to navigate to next page of reviews.
    
    Args:
        driver: WebDriver instance
        
    Returns:
        True if navigation successful, False if on last page
    """
    print("    🔄 Attempting pagination...")
    
    try:
        # Find pagination container
        pagination = driver.find_element(
            By.CSS_SELECTOR,
            'ol[data-test="page-list"]'
        )
        
        # Find enabled next button
        next_btn = pagination.find_element(
            By.CSS_SELECTOR,
            'button[data-test="next-page"]:not([disabled])'
        )
        
        # Scroll into view and click
        driver.execute_script(
            "arguments[0].scrollIntoView({block:'center'});",
            next_btn
        )
        time.sleep(1)
        next_btn.click()
        
        print("    ✓ Navigated to next page")
        time.sleep(4)  # Wait for new reviews to load
        return True
        
    except NoSuchElementException:
        print("    ✗ Next button not found (last page)")
        return False
    except Exception as e:
        print(f"    ⚠️ Pagination error: {e}")
        return False


def scrape_company(
    driver: webdriver.Chrome,
    company_url: str,
    max_reviews: int = 4000
) -> pd.DataFrame:
    """
    Scrape all reviews for a single company.
    
    PRODUCTION FEATURES:
    - URL change tracking to detect pagination issues
    - Automatic reviews tab clicking
    - Zero-review detection
    - Duplicate detection
    - Error recovery
    
    Args:
        driver: WebDriver instance
        company_url: Full URL to company's Glassdoor reviews page
        max_reviews: Maximum number of reviews to scrape
        
    Returns:
        DataFrame with scraped reviews
    """
    print(f"\n📄 Navigating to: {company_url}")
    driver.get(company_url)
    time.sleep(4)
    
    reviews_data = []
    page_num = 1
    consecutive_failures = 0
    max_failures = 2
    last_url = None  # Track URL changes for debugging
    
    while len(reviews_data) < max_reviews and consecutive_failures < max_failures:
        print(f"\n  📄 Page {page_num}...")
        
        # Check if URL changed (indicates successful pagination)
        current_url = driver.current_url
        if last_url and current_url == last_url:
            print(f"    ⚠️ URL hasn't changed (still: {current_url})")
        last_url = current_url
        
        # Wait for reviews to load
        if not wait_for_reviews_to_load(driver):
            consecutive_failures += 1
            
            # DEFENSIVE: Try clicking reviews tab on first failure
            if consecutive_failures == 1:
                print("    🔍 DEBUG: Checking why reviews aren't loading...")
                
                # Try clicking reviews tab
                try:
                    reviews_tab = driver.find_elements(
                        By.CSS_SELECTOR,
                        '[data-test="ei-nav-reviews-link"]'
                    )
                    if reviews_tab and reviews_tab[0].get_attribute('data-ui-selected') != 'true':
                        print("    Clicking reviews tab...")
                        reviews_tab[0].click()
                        time.sleep(3)
                        
                        # Try loading again
                        if wait_for_reviews_to_load(driver):
                            consecutive_failures = 0
                            continue
                except Exception:
                    pass
                
                # DEFENSIVE: Check if company has no reviews
                try:
                    count_elements = driver.find_elements(
                        By.CSS_SELECTOR,
                        '[data-test="ei-nav-reviews-link"] [class*="HeaderWrapper"]'
                    )
                    if count_elements:
                        review_count = count_elements[0].text
                        print(f"    Review count shown: {review_count}")
                        if review_count in ['--', '0']:
                            print("    ⚠️ Company has no reviews!")
                            return pd.DataFrame(reviews_data)
                except Exception:
                    pass
            
            if not try_next_page(driver):
                break
            page_num += 1
            continue
        
        # Find review elements
        review_elements = find_review_elements(driver)
        
        if not review_elements:
            consecutive_failures += 1
            if not try_next_page(driver):
                break
            page_num += 1
            continue
        
        # Reset failure counter
        consecutive_failures = 0
        
        # Extract data from each review
        new_reviews = 0
        for review_elem in review_elements:
            if len(reviews_data) >= max_reviews:
                break
            
            try:
                data = extract_review_data(review_elem)
                
                # Skip empty reviews
                if not any([
                    data.get('title'),
                    data.get('pros'),
                    data.get('cons'),
                    data.get('rating')
                ]):
                    continue
                
                # PRODUCTION: Better duplicate detection using composite signature
                review_signature = (
                    f"{data.get('title', '')}|"
                    f"{data.get('date', '')}|"
                    f"{data.get('rating', '')}|"
                    f"{(data.get('pros', '') or '')[:50]}"
                )
                
                is_duplicate = False
                for existing in reviews_data:
                    existing_signature = (
                        f"{existing.get('title', '')}|"
                        f"{existing.get('date', '')}|"
                        f"{existing.get('rating', '')}|"
                        f"{(existing.get('pros', '') or '')[:50]}"
                    )
                    if review_signature == existing_signature:
                        is_duplicate = True
                        break
                
                if is_duplicate:
                    continue
                
                reviews_data.append(data)
                new_reviews += 1
                
            except Exception:
                # Silently skip problematic reviews
                continue
        
        print(f"    ✓ Extracted {new_reviews} new reviews (total: {len(reviews_data)})")
        
        # Try next page if not at limit
        if len(reviews_data) < max_reviews:
            if not try_next_page(driver):
                print("    ✓ No more pages available")
                break
            page_num += 1
        else:
            print(f"\n  ✓ Reached max_reviews limit ({max_reviews})")
            break
    
    return pd.DataFrame(reviews_data)


def load_companies(config_path: str = None, csv_path: str = None) -> List[Dict]:
    """
    Load company list from JSON config or CSV file.
    
    FLEXIBLE: Supports both JSON and CSV formats
    
    Args:
        config_path: Path to JSON config file
        csv_path: Path to CSV file (alternative to JSON)
        
    Returns:
        List of company dictionaries with ticker, name, glassdoor_url
    """
    companies = []
    
    if csv_path:
        print(f"📂 Loading companies from CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        
        for _, row in df.iterrows():
            if pd.notna(row.get('glassdoor_url')):
                companies.append({
                    'ticker': row['ticker'],
                    'name': row['name'],
                    'glassdoor_url': row['glassdoor_url'],
                    'property_type': row.get('property_type', '')
                })
    else:
        print(f"📂 Loading companies from JSON: {config_path}")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        companies = [c for c in config['companies'] if c.get('glassdoor_url')]
    
    print(f"✓ Loaded {len(companies)} companies with Glassdoor URLs")
    return companies


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Scrape Glassdoor reviews for REIT companies (PRODUCTION VERSION)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scrape all companies (default: 4000 reviews each)
  python scripts/scrape_reviews.py
  
  # Test mode (1 company, 50 reviews)
  python scripts/scrape_reviews.py --test
  
  # Scrape specific company
  python scripts/scrape_reviews.py --company PLD
  
  # Resume from company #50
  python scripts/scrape_reviews.py --start-from 50
  
  # Use CSV file instead of JSON config
  python scripts/scrape_reviews.py --csv data/reit_companies.csv
  
  # Custom max reviews
  python scripts/scrape_reviews.py --max-reviews 1000

Before running:
  1. Start Chrome in debug mode:
     Windows: chrome.exe --remote-debugging-port=9222
     macOS: /Applications/Google\\ Chrome.app/Contents/MacOS/Google\\ Chrome --remote-debugging-port=9222
     Linux: google-chrome --remote-debugging-port=9222
  
  2. Run this script
  3. Login to Glassdoor when prompted

PRODUCTION FEATURES:
  ✓ URL change tracking
  ✓ Automatic reviews tab clicking
  ✓ Zero-review detection
  ✓ CSV/JSON config support
  ✓ Defensive error handling
        """
    )
    
    parser.add_argument(
        '--config',
        default='config/reit_companies.json',
        help='JSON config file with company list'
    )
    parser.add_argument(
        '--csv',
        help='CSV file with company list (alternative to --config)'
    )
    parser.add_argument(
        '--max-reviews',
        type=int,
        default=4000,
        help='Maximum reviews per company (default: 4000)'
    )
    parser.add_argument(
        '--output-dir',
        default='data/raw',
        help='Output directory for CSV files (default: data/raw)'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test mode: scrape 1 company, 50 reviews'
    )
    parser.add_argument(
        '--start-from',
        type=int,
        default=0,
        help='Resume from company number N (default: 0)'
    )
    parser.add_argument(
        '--company',
        help='Scrape specific company by ticker symbol'
    )
    parser.add_argument(
        '--debug-port',
        type=int,
        default=9222,
        help='Chrome debug port (default: 9222)'
    )
    
    args = parser.parse_args()
    
    # Test mode overrides
    if args.test:
        args.max_reviews = 50
        print("🧪 TEST MODE: 1 company, 50 reviews")
    
    # Load companies (CSV or JSON)
    companies = load_companies(
        config_path=args.config if not args.csv else None,
        csv_path=args.csv
    )
    
    # Filter companies
    if args.company:
        companies = [c for c in companies if c['ticker'] == args.company.upper()]
        if not companies:
            print(f"✗ Company {args.company} not found")
            return
    elif args.test:
        companies = companies[:1]
    elif args.start_from > 0:
        print(f"\n⏭️  Resuming from company #{args.start_from}")
        companies = companies[args.start_from:]
    
    print(f"\n📊 Will scrape {len(companies)} companies")
    print("="*60)
    
    # Initialize browser
    try:
        driver = init_browser(debug_port=args.debug_port)
    except Exception:
        return
    
    try:
        # Manual login
        manual_login_wait(driver)
        
        # Create output directory
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Track results
        all_reviews = []
        successful = []
        failed = []
        
        # Scrape each company
        for i, company in enumerate(companies, 1):
            actual_index = args.start_from + i if args.start_from > 0 else i
            total = args.start_from + len(companies) if args.start_from > 0 else len(companies)
            
            print(f"\n{'='*60}")
            print(f"[{actual_index}/{total}] {company['ticker']} - {company['name']}")
            print('='*60)
            
            try:
                # Scrape company
                df = scrape_company(
                    driver,
                    company['glassdoor_url'],
                    max_reviews=args.max_reviews
                )
                
                if len(df) == 0:
                    print("  ✗ No reviews scraped")
                    failed.append((company['ticker'], 'No reviews found'))
                    continue
                
                # Add metadata
                df['company'] = company['name']
                df['ticker'] = company['ticker']
                df['property_type'] = company.get('property_type', '')
                df['scrape_date'] = datetime.now().strftime('%Y-%m-%d')
                df['glassdoor_url'] = company['glassdoor_url']
                
                # Save individual file
                output_file = output_path / f"{company['ticker']}_reviews.csv"
                df.to_csv(output_file, index=False)
                print(f"  ✓ Saved {len(df)} reviews → {output_file}")
                
                all_reviews.append(df)
                successful.append((company['ticker'], len(df), output_file))
                
                # Brief pause between companies
                time.sleep(2)
                
            except KeyboardInterrupt:
                print("\n\n⚠️  Interrupted by user")
                break
            except Exception as e:
                print(f"  ✗ Error: {e}")
                failed.append((company['ticker'], str(e)[:50]))
                continue
        
        # Save combined file
        if all_reviews:
            combined = pd.concat(all_reviews, ignore_index=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            combined_file = output_path / f'all_reit_reviews_{timestamp}.csv'
            combined.to_csv(combined_file, index=False)
            
            # Print summary
            print(f"\n{'='*60}")
            print("📊 SCRAPING SUMMARY")
            print('='*60)
            print(f"✓ Total reviews scraped: {len(combined):,}")
            print(f"✓ Companies successful: {len(successful)}")
            print(f"✗ Companies failed: {len(failed)}")
            
            if successful:
                print(f"\n✅ Successful companies:")
                for ticker, count, filepath in successful:
                    print(f"  • {ticker}: {count:,} reviews → {filepath}")
            
            if failed:
                print(f"\n❌ Failed companies:")
                for ticker, reason in failed:
                    print(f"  • {ticker}: {reason}")
            
            print(f"\n💾 Combined file: {combined_file}")
            print('='*60)
        else:
            print("\n✗ No reviews scraped from any company")
    
    finally:
        driver.quit()
        print("\n✅ Scraping complete!")


if __name__ == '__main__':
    main()