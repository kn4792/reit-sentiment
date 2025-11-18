"""
src/data_collection/glassdoor_scraper.py
Glassdoor Employee Review Scraper for REITs - Updated 2024
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time
import json
from datetime import datetime


class GlassdoorScraper:
    """
    Scraper for Glassdoor employee reviews.
    Updated for 2024 Glassdoor structure
    """
    
    def __init__(self, username=None, password=None, headless=False, test_mode=False, credentials_path=None):
        """
        Initialize the scraper.
        
        Args:
            username: Glassdoor account email
            password: Glassdoor account password
            headless: Run browser in headless mode (default: False for debugging)
            test_mode: If True, skip actual scraping (for testing)
            credentials_path: Path to JSON file with credentials
        """
        self.test_mode = test_mode
        
        if test_mode:
            self.driver = None
            self.username = "test@example.com"
            self.password = "test_password"
            return
        
        # Load credentials from file if provided
        if credentials_path:
            with open(credentials_path, 'r') as f:
                creds = json.load(f)
                self.username = creds.get('glassdoor_username', username)
                self.password = creds.get('glassdoor_password', password)
        else:
            self.username = username
            self.password = password
        
        if not self.username or not self.password:
            raise ValueError("Username and password required (or provide credentials_path)")
        
        # Setup Chrome options with anti-detection measures
        options = webdriver.ChromeOptions()
        if headless:
            options.add_argument('--headless=new')  # New headless mode
        
        # Anti-detection options
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        
        try:
            # Use webdriver-manager to automatically handle ChromeDriver
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=options)
            
            # Hide webdriver property
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            self.wait = WebDriverWait(self.driver, 15)
            
            print("✓ Chrome browser initialized")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Chrome driver: {e}")
    
    def login(self):
        """Login to Glassdoor - Updated for 2024 structure"""
        if self.test_mode:
            print("Test mode: Skipping login")
            return True
        
        print("Navigating to Glassdoor login page...")
        
        try:
            # Go directly to login page
            self.driver.get('https://www.glassdoor.com/profile/login_input.htm')
            time.sleep(3)
            
            print("Looking for email field...")
            
            # Try multiple selectors for email field
            email_field = None
            email_selectors = [
                (By.ID, 'inlineUserEmail'),
                (By.NAME, 'username'),
                (By.CSS_SELECTOR, 'input[type="email"]'),
                (By.CSS_SELECTOR, 'input[name="username"]'),
                (By.XPATH, '//input[@type="email"]'),
            ]
            
            for by, selector in email_selectors:
                try:
                    email_field = self.wait.until(
                        EC.presence_of_element_located((by, selector))
                    )
                    print(f"✓ Found email field using: {by}={selector}")
                    break
                except TimeoutException:
                    continue
            
            if not email_field:
                print("✗ Could not find email field")
                print("Page source preview:")
                print(self.driver.page_source[:500])
                raise Exception("Email field not found - Glassdoor structure may have changed")
            
            # Enter email
            email_field.clear()
            email_field.send_keys(self.username)
            time.sleep(1)
            
            print("Looking for password field...")
            
            # Try multiple selectors for password field
            password_field = None
            password_selectors = [
                (By.ID, 'inlineUserPassword'),
                (By.NAME, 'password'),
                (By.CSS_SELECTOR, 'input[type="password"]'),
                (By.XPATH, '//input[@type="password"]'),
            ]
            
            for by, selector in password_selectors:
                try:
                    password_field = self.driver.find_element(by, selector)
                    print(f"✓ Found password field using: {by}={selector}")
                    break
                except NoSuchElementException:
                    continue
            
            if not password_field:
                raise Exception("Password field not found")
            
            # Enter password
            password_field.clear()
            password_field.send_keys(self.password)
            time.sleep(1)
            
            print("Looking for submit button...")
            
            # Try multiple selectors for submit button
            submit_btn = None
            submit_selectors = [
                (By.CSS_SELECTOR, 'button[type="submit"]'),
                (By.CSS_SELECTOR, 'button[name="submit"]'),
                (By.XPATH, '//button[contains(text(), "Sign In")]'),
                (By.XPATH, '//button[contains(text(), "Log In")]'),
                (By.CSS_SELECTOR, '.gd-ui-button'),
            ]
            
            for by, selector in submit_selectors:
                try:
                    submit_btn = self.driver.find_element(by, selector)
                    print(f"✓ Found submit button using: {by}={selector}")
                    break
                except NoSuchElementException:
                    continue
            
            if not submit_btn:
                raise Exception("Submit button not found")
            
            # Click submit
            submit_btn.click()
            print("✓ Clicked submit button")
            
            time.sleep(5)  # Wait for login to complete
            
            # Check if login was successful
            current_url = self.driver.current_url
            if 'login' not in current_url.lower():
                print("✓ Login successful!")
                return True
            else:
                print("⚠ Still on login page - check credentials or CAPTCHA")
                print(f"Current URL: {current_url}")
                return False
            
        except Exception as e:
            print(f"✗ Login failed: {e}")
            print(f"\nCurrent URL: {self.driver.current_url}")
            print(f"\nPage title: {self.driver.title}")
            
            # Save screenshot for debugging
            try:
                self.driver.save_screenshot('data/login_error.png')
                print("✓ Screenshot saved to data/login_error.png")
            except:
                pass
            
            raise
    
    def scrape_reviews(self, company_url, max_reviews=1000, output_file=None):
        """
        Scrape employee reviews from a company's Glassdoor page.
        
        Args:
            company_url: Full Glassdoor company review URL
            max_reviews: Maximum number of reviews to scrape
            output_file: CSV filename for output (optional)
        
        Returns:
            DataFrame of scraped reviews
        """
        if self.test_mode:
            print(f"Test mode: Would scrape {max_reviews} reviews from {company_url}")
            return pd.DataFrame({
                'title': ['Test Review'],
                'rating': ['4.0'],
                'pros': ['Good company'],
                'cons': ['Some issues'],
                'date': ['Jan 1, 2023']
            })
        
        print(f"\nNavigating to: {company_url}")
        self.driver.get(company_url)
        time.sleep(5)  # Wait for page to load
        
        reviews_data = []
        pages_scraped = 0
        
        while len(reviews_data) < max_reviews:
            # Wait for reviews to load
            try:
                self.wait.until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "[data-test='employerReview'], .review, .empReview"))
                )
            except TimeoutException:
                print("✗ No reviews found on this page")
                break
            
            # Scroll to load dynamic content
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            
            # Get all review elements - try multiple selectors
            review_elements = []
            review_selectors = [
                "[data-test='employerReview']",
                ".review",
                ".empReview",
                "[class*='review']"
            ]
            
            for selector in review_selectors:
                review_elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                if review_elements:
                    print(f"✓ Found {len(review_elements)} reviews using selector: {selector}")
                    break
            
            if not review_elements:
                print("✗ No review elements found")
                break
            
            for review in review_elements:
                if len(reviews_data) >= max_reviews:
                    break
                    
                try:
                    review_data = self._extract_review_data(review)
                    reviews_data.append(review_data)
                    
                    if len(reviews_data) % 10 == 0:
                        print(f"  Scraped {len(reviews_data)} reviews...")
                        
                except Exception as e:
                    print(f"  ✗ Error extracting review: {e}")
                    continue
            
            pages_scraped += 1
            print(f"✓ Page {pages_scraped} complete ({len(reviews_data)} total reviews)")
            
            # Try to go to next page
            if len(reviews_data) < max_reviews:
                if not self._go_to_next_page():
                    break
                time.sleep(3)
        
        print(f"\n✓ Total reviews scraped: {len(reviews_data)}")
        
        # Convert to DataFrame
        df = pd.DataFrame(reviews_data)
        
        # Save to CSV if filename provided
        if output_file and len(df) > 0:
            df.to_csv(output_file, index=False)
            print(f"✓ Saved to {output_file}")
        
        return df
    
    def _extract_review_data(self, review_element):
        """Extract data from a single review element"""
        data = {}
        
        # Review title
        title_selectors = ["h2[class*='title']", ".reviewLink", "[data-test='title']"]
        for selector in title_selectors:
            try:
                data['title'] = review_element.find_element(By.CSS_SELECTOR, selector).text
                break
            except:
                data['title'] = None
        
        # Overall rating
        rating_selectors = ["span[class*='ratingNumber']", ".ratingNumber", "[data-test='rating']"]
        for selector in rating_selectors:
            try:
                data['rating'] = review_element.find_element(By.CSS_SELECTOR, selector).text
                break
            except:
                data['rating'] = None
        
        # Employee info
        try:
            data['employee_info'] = review_element.find_element(By.CSS_SELECTOR, "[class*='employee']").text
        except:
            data['employee_info'] = None
        
        # Review date
        try:
            data['date'] = review_element.find_element(By.CSS_SELECTOR, "[class*='reviewDate'], .date").text
        except:
            data['date'] = None
        
        # Pros
        pros_selectors = ["span[data-test='pros']", ".pros", "[class*='pros']"]
        for selector in pros_selectors:
            try:
                data['pros'] = review_element.find_element(By.CSS_SELECTOR, selector).text
                break
            except:
                data['pros'] = None
        
        # Cons
        cons_selectors = ["span[data-test='cons']", ".cons", "[class*='cons']"]
        for selector in cons_selectors:
            try:
                data['cons'] = review_element.find_element(By.CSS_SELECTOR, selector).text
                break
            except:
                data['cons'] = None
        
        # Advice to management (optional)
        try:
            data['advice'] = review_element.find_element(By.CSS_SELECTOR, "span[data-test='advice-management']").text
        except:
            data['advice'] = None
        
        return data
    
    def _go_to_next_page(self):
        """Navigate to the next page of reviews"""
        try:
            # Try multiple selectors for next button
            next_selectors = [
                "button[data-test='pagination-next']",
                ".nextButton",
                "[aria-label='Next']",
                "a[data-test='pagination-link']:last-child"
            ]
            
            for selector in next_selectors:
                try:
                    next_button = self.driver.find_element(By.CSS_SELECTOR, selector)
                    if next_button.is_enabled():
                        next_button.click()
                        return True
                except:
                    continue
            
            print("  ✗ No next button found - reached last page")
            return False
                
        except Exception as e:
            print(f"  ✗ Error navigating to next page: {e}")
            return False
    
    def close(self):
        """Close the browser"""
        if self.driver:
            self.driver.quit()
            print("\n✓ Browser closed")


# For backwards compatibility
GlassdoorREITScraper = GlassdoorScraper