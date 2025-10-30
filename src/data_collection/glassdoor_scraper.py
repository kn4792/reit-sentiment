from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import pandas as pd
import time
import json
from datetime import datetime


class GlassdoorScraper:
    """
    Scraper for Glassdoor employee reviews.
    
    Example:
        >>> scraper = GlassdoorScraper(username="email@example.com", password="pass")
        >>> scraper.login()
        >>> reviews = scraper.scrape_reviews(company_url, max_reviews=100)
    """
    
    def __init__(self, username=None, password=None, headless=True, test_mode=False, credentials_path=None):
        """
        Initialize the scraper.
        
        Args:
            username: Glassdoor account email
            password: Glassdoor account password
            headless: Run browser in headless mode (default: True)
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
        
        # Setup Chrome options
        options = webdriver.ChromeOptions()
        if headless:
            options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        
        try:
            self.driver = webdriver.Chrome(options=options)
            self.wait = WebDriverWait(self.driver, 10)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Chrome driver: {e}")
    
    def login(self):
        """Login to Glassdoor"""
        if self.test_mode:
            print("Test mode: Skipping login")
            return True
        
        print("Logging into Glassdoor...")
        try:
            self.driver.get('https://www.glassdoor.com/profile/login_input.htm')
            time.sleep(2)
            
            # Enter email
            email_field = self.wait.until(
                EC.presence_of_element_located((By.NAME, "username"))
            )
            email_field.send_keys(self.username)
            
            # Enter password
            password_field = self.driver.find_element(By.NAME, "password")
            password_field.send_keys(self.password)
            
            # Click submit
            submit_btn = self.driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
            submit_btn.click()
            
            time.sleep(3)
            print("Login successful!")
            return True
            
        except Exception as e:
            print(f"Login failed: {e}")
            if self.driver:
                self.driver.quit()
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
        
        print(f"\nScraping reviews from: {company_url}")
        self.driver.get(company_url)
        time.sleep(3)
        
        reviews_data = []
        pages_scraped = 0
        
        while len(reviews_data) < max_reviews:
            # Wait for reviews to load
            try:
                self.wait.until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "[data-test='employerReview']"))
                )
            except TimeoutException:
                print("No reviews found on this page")
                break
            
            # Get all review elements on current page
            review_elements = self.driver.find_elements(By.CSS_SELECTOR, "[data-test='employerReview']")
            
            for review in review_elements:
                if len(reviews_data) >= max_reviews:
                    break
                    
                try:
                    review_data = self._extract_review_data(review)
                    reviews_data.append(review_data)
                    
                    if len(reviews_data) % 10 == 0:
                        print(f"Scraped {len(reviews_data)} reviews...")
                        
                except Exception as e:
                    print(f"Error extracting review: {e}")
                    continue
            
            pages_scraped += 1
            
            # Try to go to next page
            if len(reviews_data) < max_reviews:
                if not self._go_to_next_page():
                    break
        
        print(f"\nTotal reviews scraped: {len(reviews_data)}")
        
        # Convert to DataFrame
        df = pd.DataFrame(reviews_data)
        
        # Save to CSV if filename provided
        if output_file:
            df.to_csv(output_file, index=False)
            print(f"Saved to {output_file}")
        
        return df
    
    def _extract_review_data(self, review_element):
        """Extract data from a single review element"""
        data = {}
        
        try:
            data['title'] = review_element.find_element(
                By.CSS_SELECTOR, "h2[class*='title']"
            ).text
        except NoSuchElementException:
            data['title'] = None
        
        try:
            data['rating'] = review_element.find_element(
                By.CSS_SELECTOR, "span[class*='ratingNumber']"
            ).text
        except NoSuchElementException:
            data['rating'] = None
        
        try:
            data['employee_info'] = review_element.find_element(
                By.CSS_SELECTOR, "[class*='employee']"
            ).text
        except NoSuchElementException:
            data['employee_info'] = None
        
        try:
            data['date'] = review_element.find_element(
                By.CSS_SELECTOR, "[class*='reviewDate']"
            ).text
        except NoSuchElementException:
            data['date'] = None
        
        try:
            data['pros'] = review_element.find_element(
                By.CSS_SELECTOR, "span[data-test='pros']"
            ).text
        except NoSuchElementException:
            data['pros'] = None
        
        try:
            data['cons'] = review_element.find_element(
                By.CSS_SELECTOR, "span[data-test='cons']"
            ).text
        except NoSuchElementException:
            data['cons'] = None
        
        try:
            data['advice'] = review_element.find_element(
                By.CSS_SELECTOR, "span[data-test='advice-management']"
            ).text
        except NoSuchElementException:
            data['advice'] = None
        
        return data
    
    def _go_to_next_page(self):
        """Navigate to the next page of reviews"""
        try:
            next_button = self.driver.find_element(
                By.CSS_SELECTOR, "button[data-test='pagination-next']"
            )
            
            if next_button.is_enabled():
                next_button.click()
                time.sleep(3)
                return True
            else:
                print("Next button disabled - reached last page")
                return False
                
        except NoSuchElementException:
            print("No next button found - reached last page")
            return False
    
    def close(self):
        """Close the browser"""
        if self.driver:
            self.driver.quit()
            print("\nBrowser closed")



# For backwards compatibility
GlassdoorREITScraper = GlassdoorScraper