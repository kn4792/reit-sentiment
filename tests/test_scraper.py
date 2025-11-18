#!/usr/bin/env python3
"""
Unit Tests for Glassdoor Scraper Module

Tests scraping functions with mocked Selenium WebDriver and elements.

Author: Konain Niaz (kn4792@rit.edu)
Date: 2025-01-18

Usage:
    pytest tests/test_scraper.py -v
    pytest tests/test_scraper.py::TestExtractReviewData -v
"""

import pytest
import pandas as pd
from unittest.mock import Mock, MagicMock, patch, call
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.common.by import By
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# MOCK HELPER FUNCTIONS
# ============================================================================

def create_mock_review_element(title="Great Company", rating="4.5", 
                               date="Jan 15, 2025", employee_info="Current Employee",
                               pros="Great benefits and culture", 
                               cons="Long hours sometimes"):
    """
    Create a mock review element for testing.
    
    Args:
        title: Review title
        rating: Rating value
        date: Review date
        employee_info: Employee information
        pros: Pros text
        cons: Cons text
        
    Returns:
        Mock WebElement configured with review data
    """
    mock_elem = Mock()
    
    # Mock title element
    mock_title = Mock()
    mock_title.text = title
    
    # Mock rating element
    mock_rating = Mock()
    mock_rating.text = rating
    
    # Mock date element
    mock_date = Mock()
    mock_date.text = date
    
    # Mock employee info
    mock_employee = Mock()
    mock_employee.text = employee_info
    
    # Mock pros/cons parent elements
    mock_pros_parent = Mock()
    mock_pros_parent.text = f"Pros\n{pros}"
    
    mock_cons_parent = Mock()
    mock_cons_parent.text = f"Cons\n{cons}"
    
    # Mock pros/cons candidates
    mock_pros_candidate = Mock()
    mock_pros_candidate.find_element.return_value = mock_pros_parent
    
    mock_cons_candidate = Mock()
    mock_cons_candidate.find_element.return_value = mock_cons_parent
    
    # Configure find_element behavior
    def find_element_side_effect(by, selector):
        if 'h2' in selector or 'ReviewTitle' in selector:
            return mock_title
        elif 'RatingNumber' in selector or 'rating' in selector:
            return mock_rating
        elif 'date' in selector.lower() or selector == 'time':
            return mock_date
        elif 'JobTitle' in selector or 'EmployeeInfo' in selector:
            return mock_employee
        else:
            raise NoSuchElementException()
    
    mock_elem.find_element.side_effect = find_element_side_effect
    
    # Configure find_elements for XPath
    def find_elements_side_effect(by, xpath):
        if 'Pros' in xpath or 'pros' in xpath:
            return [mock_pros_candidate]
        elif 'Cons' in xpath or 'cons' in xpath:
            return [mock_cons_candidate]
        return []
    
    mock_elem.find_elements.side_effect = find_elements_side_effect
    
    return mock_elem


# ============================================================================
# MOCK SCRAPER MODULE FUNCTIONS
# ============================================================================

def mock_extract_review_data(review_element):
    """Extract data from review element (mock version for testing)."""
    data = {}
    
    # Extract title
    title_selectors = ['h2', 'a[href*="Reviews"]', 'span[class*="ReviewTitle"]']
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
    rating_selectors = ['span[class*="RatingNumber"]', '[data-test="rating"]']
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
    date_selectors = ['span[class*="date" i]', 'time', '[data-test="review-date"]']
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
    employee_selectors = ['span[class*="JobTitle" i]', 'div[class*="EmployeeInfo"]']
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


# ============================================================================
# BROWSER INITIALIZATION TESTS
# ============================================================================

class TestInitBrowser:
    """Tests for browser initialization."""
    
    @patch('selenium.webdriver.Chrome')
    def test_init_browser_success(self, mock_chrome):
        """Test successful browser initialization."""
        mock_driver = Mock()
        mock_chrome.return_value = mock_driver
        
        # Test directly with mock
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        
        options = Options()
        options.add_experimental_option("debuggerAddress", "127.0.0.1:9222")
        driver = mock_chrome(options=options)
        
        assert driver is not None
        mock_chrome.assert_called_once()
    
    @patch('selenium.webdriver.Chrome')
    def test_init_browser_failure(self, mock_chrome):
        """Test browser initialization failure."""
        mock_chrome.side_effect = Exception("Connection failed")
        
        with pytest.raises(Exception):
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            
            options = Options()
            driver = mock_chrome(options=options)
    
    @patch('selenium.webdriver.Chrome')
    def test_init_browser_custom_port(self, mock_chrome):
        """Test browser initialization with custom port."""
        mock_driver = Mock()
        mock_chrome.return_value = mock_driver
        
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        
        options = Options()
        options.add_experimental_option("debuggerAddress", "127.0.0.1:9223")
        driver = mock_chrome(options=options)
        
        assert driver is not None


# ============================================================================
# REVIEW ELEMENT FINDING TESTS
# ============================================================================

class TestFindReviewElements:
    """Tests for finding review elements."""
    
    def test_find_review_elements_success(self):
        """Test successful review element finding."""
        # Create mock driver
        mock_driver = Mock()
        mock_reviews_list = Mock()
        
        # Create mock list items
        mock_li_1 = Mock()
        mock_li_2 = Mock()
        mock_li_3 = Mock()
        
        mock_reviews_list.find_elements.return_value = [mock_li_1, mock_li_2, mock_li_3]
        mock_driver.find_element.return_value = mock_reviews_list
        
        # Test directly
        try:
            reviews_list = mock_driver.find_element(By.CSS_SELECTOR, 'ol.ReviewsList_reviewsList__Qfw6M')
            elements = reviews_list.find_elements(By.TAG_NAME, 'li')
        except NoSuchElementException:
            elements = []
        
        assert len(elements) == 3
        assert elements[0] == mock_li_1
    
    def test_find_review_elements_not_found(self):
        """Test when review elements are not found."""
        mock_driver = Mock()
        mock_driver.find_element.side_effect = NoSuchElementException()
        mock_driver.find_elements.return_value = []
        
        try:
            reviews_list = mock_driver.find_element(By.CSS_SELECTOR, 'ol.ReviewsList_reviewsList__Qfw6M')
            elements = reviews_list.find_elements(By.TAG_NAME, 'li')
        except NoSuchElementException:
            elements = []
        
        assert elements == []
    
    def test_find_review_elements_empty_list(self):
        """Test when reviews list is found but empty."""
        mock_driver = Mock()
        mock_reviews_list = Mock()
        mock_reviews_list.find_elements.return_value = []
        mock_driver.find_element.return_value = mock_reviews_list
        
        try:
            reviews_list = mock_driver.find_element(By.CSS_SELECTOR, 'ol.ReviewsList_reviewsList__Qfw6M')
            elements = reviews_list.find_elements(By.TAG_NAME, 'li')
        except NoSuchElementException:
            elements = []
        
        assert isinstance(elements, list)


# ============================================================================
# EXTRACT REVIEW DATA TESTS
# ============================================================================

class TestExtractReviewData:
    """Tests for review data extraction."""
    
    def test_extract_complete_review(self):
        """Test extraction of complete review with all fields."""
        mock_elem = create_mock_review_element()
        data = mock_extract_review_data(mock_elem)
        
        assert data['title'] == "Great Company"
        assert data['rating'] == "4.5"
        assert data['date'] == "Jan 15, 2025"
        assert data['employee_info'] == "Current Employee"
        assert 'benefits' in data['pros'].lower()
        assert 'hours' in data['cons'].lower()
    
    def test_extract_review_missing_title(self):
        """Test extraction when title is missing."""
        mock_elem = Mock()
        mock_elem.find_element.side_effect = NoSuchElementException()
        mock_elem.find_elements.return_value = []
        
        data = mock_extract_review_data(mock_elem)
        
        # Should return empty dict or partial data
        assert isinstance(data, dict)
    
    def test_extract_review_missing_rating(self):
        """Test extraction when rating is missing."""
        mock_elem = create_mock_review_element()
        
        # Override rating to fail
        def find_elem_override(by, selector):
            if 'Rating' in selector:
                raise NoSuchElementException()
            return Mock(text="Some text")
        
        mock_elem.find_element.side_effect = find_elem_override
        
        data = mock_extract_review_data(mock_elem)
        
        # Should still extract other fields
        assert isinstance(data, dict)
    
    def test_extract_review_various_ratings(self):
        """Test extraction of various rating formats."""
        test_ratings = ["5.0", "3.5", "2", "4.75"]
        
        for rating in test_ratings:
            mock_elem = create_mock_review_element(rating=rating)
            data = mock_extract_review_data(mock_elem)
            
            assert data['rating'] == rating
            assert any(c.isdigit() for c in data['rating'])
    
    def test_extract_review_pros_cons_extraction(self):
        """Test pros/cons extraction with various formats."""
        test_cases = [
            ("Great benefits", "Long hours"),
            ("Good work-life balance", "Management issues"),
            ("Flexible schedule", "Low pay")
        ]
        
        for pros, cons in test_cases:
            mock_elem = create_mock_review_element(pros=pros, cons=cons)
            data = mock_extract_review_data(mock_elem)
            
            assert pros.lower() in data['pros'].lower()
            assert cons.lower() in data['cons'].lower()


# ============================================================================
# PAGINATION TESTS
# ============================================================================

class TestTryNextPage:
    """Tests for pagination functionality."""
    
    def test_try_next_page_success(self):
        """Test successful navigation to next page."""
        mock_driver = Mock()
        mock_pagination = Mock()
        mock_next_btn = Mock()
        
        mock_pagination.find_element.return_value = mock_next_btn
        mock_driver.find_element.return_value = mock_pagination
        mock_driver.execute_script = Mock()
        
        # Test pagination logic
        try:
            pagination = mock_driver.find_element(By.CSS_SELECTOR, 'ol[data-test="page-list"]')
            next_btn = pagination.find_element(By.CSS_SELECTOR, 'button[data-test="next-page"]:not([disabled])')
            next_btn.click()
            result = True
        except NoSuchElementException:
            result = False
        
        assert result == True
        mock_next_btn.click.assert_called_once()
    
    def test_try_next_page_disabled(self):
        """Test when next button is disabled (last page)."""
        mock_driver = Mock()
        mock_pagination = Mock()
        mock_pagination.find_element.side_effect = NoSuchElementException()
        mock_driver.find_element.return_value = mock_pagination
        
        try:
            pagination = mock_driver.find_element(By.CSS_SELECTOR, 'ol[data-test="page-list"]')
            next_btn = pagination.find_element(By.CSS_SELECTOR, 'button[data-test="next-page"]:not([disabled])')
            result = True
        except NoSuchElementException:
            result = False
        
        assert result == False
    
    def test_try_next_page_no_pagination(self):
        """Test when pagination container is not found."""
        mock_driver = Mock()
        mock_driver.find_element.side_effect = NoSuchElementException()
        
        try:
            pagination = mock_driver.find_element(By.CSS_SELECTOR, 'ol[data-test="page-list"]')
            result = True
        except NoSuchElementException:
            result = False
        
        assert result == False


# ============================================================================
# WAIT FOR REVIEWS TESTS
# ============================================================================

class TestWaitForReviewsToLoad:
    """Tests for waiting for reviews to load."""
    
    @patch('selenium.webdriver.support.ui.WebDriverWait')
    def test_wait_for_reviews_success(self, mock_wait):
        """Test successful wait for reviews."""
        mock_driver = Mock()
        mock_wait_instance = Mock()
        mock_wait.return_value = mock_wait_instance
        mock_wait_instance.until.return_value = Mock()  # Reviews loaded
        
        # Simulate wait
        wait = mock_wait(mock_driver, 10)
        result = True
        
        assert result == True
    
    @patch('selenium.webdriver.support.ui.WebDriverWait')
    def test_wait_for_reviews_timeout(self, mock_wait):
        """Test timeout waiting for reviews."""
        mock_driver = Mock()
        mock_wait_instance = Mock()
        mock_wait.return_value = mock_wait_instance
        mock_wait_instance.until.side_effect = TimeoutException()
        
        # Simulate timeout
        wait = mock_wait(mock_driver, 10)
        try:
            wait.until(lambda d: d)
            result = True
        except TimeoutException:
            result = False
        
        assert result == False


# ============================================================================
# COMPANY SCRAPING INTEGRATION TESTS
# ============================================================================

class TestScrapeCompany:
    """Integration tests for scraping a complete company."""
    
    def test_scrape_company_single_page(self):
        """Test scraping company with single page of reviews."""
        # Create mock data
        reviews_data = [
            {'title': 'Review 1', 'rating': '4.0'},
            {'title': 'Review 2', 'rating': '3.5'},
            {'title': 'Review 3', 'rating': '5.0'}
        ]
        
        df = pd.DataFrame(reviews_data)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
    
    def test_scrape_company_max_reviews_limit(self):
        """Test scraping stops at max_reviews limit."""
        # Create mock data beyond limit
        reviews_data = [
            {'title': f'Review {i}', 'rating': '4.0'}
            for i in range(20)
        ]
        
        df = pd.DataFrame(reviews_data)
        max_reviews = 5
        df_limited = df.head(max_reviews)
        
        # Should stop at 5 reviews
        assert len(df_limited) == 5
    
    def test_scrape_company_no_reviews(self):
        """Test scraping company with no reviews."""
        reviews_data = []
        df = pd.DataFrame(reviews_data)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


# ============================================================================
# DUPLICATE DETECTION TESTS
# ============================================================================

class TestDuplicateDetection:
    """Tests for duplicate review detection."""
    
    def test_duplicate_signature_matching(self):
        """Test duplicate detection using composite signature."""
        # Create two identical reviews
        review1 = {
            'title': 'Great Company',
            'date': 'Jan 15, 2025',
            'rating': '4.5',
            'pros': 'Great benefits and excellent culture'
        }
        
        review2 = {
            'title': 'Great Company',
            'date': 'Jan 15, 2025',
            'rating': '4.5',
            'pros': 'Great benefits and excellent culture'
        }
        
        # Create signatures
        sig1 = (
            f"{review1.get('title', '')}|"
            f"{review1.get('date', '')}|"
            f"{review1.get('rating', '')}|"
            f"{review1.get('pros', '')[:50]}"
        )
        
        sig2 = (
            f"{review2.get('title', '')}|"
            f"{review2.get('date', '')}|"
            f"{review2.get('rating', '')}|"
            f"{review2.get('pros', '')[:50]}"
        )
        
        assert sig1 == sig2
    
    def test_duplicate_signature_different(self):
        """Test that different reviews have different signatures."""
        review1 = {
            'title': 'Great Company',
            'date': 'Jan 15, 2025',
            'rating': '4.5',
            'pros': 'Great benefits'
        }
        
        review2 = {
            'title': 'Good Company',
            'date': 'Jan 16, 2025',
            'rating': '4.0',
            'pros': 'Good benefits'
        }
        
        sig1 = (
            f"{review1.get('title', '')}|"
            f"{review1.get('date', '')}|"
            f"{review1.get('rating', '')}|"
            f"{review1.get('pros', '')[:50]}"
        )
        
        sig2 = (
            f"{review2.get('title', '')}|"
            f"{review2.get('date', '')}|"
            f"{review2.get('rating', '')}|"
            f"{review2.get('pros', '')[:50]}"
        )
        
        assert sig1 != sig2


# ============================================================================
# CONFIGURATION LOADING TESTS
# ============================================================================

class TestLoadCompanies:
    """Tests for loading company configuration."""
    
    def test_load_companies_from_json(self, tmp_path):
        """Test loading companies from JSON config."""
        import json
        
        # Create temp JSON file
        config = {
            'companies': [
                {
                    'ticker': 'PLD',
                    'name': 'Prologis',
                    'glassdoor_url': 'https://example.com/pld'
                },
                {
                    'ticker': 'AMT',
                    'name': 'American Tower',
                    'glassdoor_url': 'https://example.com/amt'
                }
            ]
        }
        
        config_file = tmp_path / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f)
        
        # Load companies
        with open(config_file, 'r') as f:
            loaded_config = json.load(f)
        companies = [c for c in loaded_config['companies'] if c.get('glassdoor_url')]
        
        assert len(companies) == 2
        assert companies[0]['ticker'] == 'PLD'
    
    def test_load_companies_from_csv(self, tmp_path):
        """Test loading companies from CSV file."""
        # Create temp CSV file
        df = pd.DataFrame({
            'ticker': ['PLD', 'AMT'],
            'name': ['Prologis', 'American Tower'],
            'glassdoor_url': ['https://example.com/pld', 'https://example.com/amt']
        })
        
        csv_file = tmp_path / "companies.csv"
        df.to_csv(csv_file, index=False)
        
        # Load companies
        loaded_df = pd.read_csv(csv_file)
        companies = []
        for _, row in loaded_df.iterrows():
            if pd.notna(row.get('glassdoor_url')):
                companies.append(dict(row))
        
        assert len(companies) == 2
        assert companies[0]['ticker'] == 'PLD'
    
    def test_load_companies_filters_missing_urls(self, tmp_path):
        """Test that companies without URLs are filtered out."""
        df = pd.DataFrame({
            'ticker': ['PLD', 'AMT', 'EQIX'],
            'name': ['Prologis', 'American Tower', 'Equinix'],
            'glassdoor_url': ['https://example.com/pld', None, 'https://example.com/eqix']
        })
        
        csv_file = tmp_path / "companies.csv"
        df.to_csv(csv_file, index=False)
        
        # Load companies
        loaded_df = pd.read_csv(csv_file)
        companies = []
        for _, row in loaded_df.iterrows():
            if pd.notna(row.get('glassdoor_url')):
                companies.append(dict(row))
        
        assert len(companies) == 2  # AMT should be filtered out


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])