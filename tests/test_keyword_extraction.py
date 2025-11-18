import pytest
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.keyword_extraction import KeywordExtractor

@pytest.fixture
def sample_texts():
    return [
        "Great company culture and amazing team collaboration",
        "Excellent benefits package and competitive salary",
        "Good work-life balance and flexible schedule",
        "Strong leadership and clear communication",
        "Innovative AI technology and machine learning projects",
        "Poor management and lack of direction",
        "Low compensation compared to market rates",
        "Limited career growth opportunities",
        "Outdated technology and legacy systems",
        "High pressure environment with long hours"
    ]

def test_initialization():
    extractor = KeywordExtractor(max_features=100)
    assert extractor is not None
    assert hasattr(extractor, 'fit')
    assert hasattr(extractor, 'extract_top_keywords')

def test_fit_and_keywords(sample_texts):
    extractor = KeywordExtractor(max_features=10, min_df=1)
    extractor.fit(sample_texts)
    keywords = extractor.extract_top_keywords(sample_texts, top_k=5)
    assert isinstance(keywords, list)
    assert len(keywords) <= 5
    # Check that keyword is a tuple of (word, score)
    assert isinstance(keywords[0], tuple)
    assert isinstance(keywords[0][0], str)

def test_extract_by_company():
    df = pd.DataFrame({
        "ticker": ["AAA", "AAA", "BBB", "BBB"],
        "pros_cleaned": [
            "AI machine learning automation",
            "Strong management team",
            "work life balance innovation",
            "leadership culture"
        ]
    })
    extractor = KeywordExtractor(max_features=10, min_df=1)
    texts = df["pros_cleaned"].tolist()
    extractor.fit(texts)
    company_keywords = extractor.extract_by_company(df, text_column="pros_cleaned", top_k=3)
    assert isinstance(company_keywords, pd.DataFrame)
    assert "keyword" in company_keywords.columns
    assert "ticker" in company_keywords.columns
    assert len(company_keywords) > 0

def test_handles_small_empty_corpus():
    extractor = KeywordExtractor(max_features=10, min_df=1)
    with pytest.raises(ValueError):
        extractor.fit(["", ""])

def test_top_keywords_are_relevant(sample_texts):
    extractor = KeywordExtractor(max_features=20, min_df=1)
    extractor.fit(sample_texts)
    keywords = extractor.extract_top_keywords(sample_texts, top_k=10)
    word_list = [k[0].lower() for k in keywords]
    found = any(word in word_list for word in ['company', 'culture', 'technology'])
    assert found

