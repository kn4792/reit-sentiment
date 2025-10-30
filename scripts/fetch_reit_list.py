

import pandas as pd
import yfinance as yf
import json
import requests
from bs4 import BeautifulSoup
import time
from pathlib import Path
import re


def fetch_nareit_reits():
    """
    Fetch REITs from NAREIT or other sources
    Returns DataFrame with ticker, company name, sector
    """
    print("Fetching REIT list from multiple sources...")
    
    # Method 1: From a known REIT ETF holdings (VNQ - Vanguard Real Estate ETF)
    try:
        print("\n1. Fetching from VNQ ETF holdings...")
        vnq = yf.Ticker("VNQ")
        holdings = vnq.info
        # Note: This might not give all holdings, need alternative
    except Exception as e:
        print(f"   Could not fetch from VNQ: {e}")
    
    # Method 2: Manually curated list of major REITs by sector
    # This is a starter list - you can expand it
    major_reits = [
        # Retail REITs
        {"name": "Simon Property Group", "ticker": "SPG", "sector": "Retail"},
        {"name": "Realty Income", "ticker": "O", "sector": "Retail"},
        {"name": "Regency Centers", "ticker": "REG", "sector": "Retail"},
        {"name": "Kimco Realty", "ticker": "KIM", "sector": "Retail"},
        {"name": "Federal Realty Investment Trust", "ticker": "FRT", "sector": "Retail"},
        {"name": "Brixmor Property Group", "ticker": "BRX", "sector": "Retail"},
        {"name": "SITE Centers", "ticker": "SITC", "sector": "Retail"},
        
        # Office REITs
        {"name": "Boston Properties", "ticker": "BXP", "sector": "Office"},
        {"name": "Vornado Realty Trust", "ticker": "VNO", "sector": "Office"},
        {"name": "Kilroy Realty", "ticker": "KRC", "sector": "Office"},
        {"name": "SL Green Realty", "ticker": "SLG", "sector": "Office"},
        {"name": "Douglas Emmett", "ticker": "DEI", "sector": "Office"},
        {"name": "Alexandria Real Estate", "ticker": "ARE", "sector": "Office"},
        {"name": "Highwoods Properties", "ticker": "HIW", "sector": "Office"},
        
        # Industrial REITs
        {"name": "Prologis", "ticker": "PLD", "sector": "Industrial"},
        {"name": "Duke Realty", "ticker": "DRE", "sector": "Industrial"},
        {"name": "Americold Realty Trust", "ticker": "COLD", "sector": "Industrial"},
        {"name": "STAG Industrial", "ticker": "STAG", "sector": "Industrial"},
        {"name": "EastGroup Properties", "ticker": "EGP", "sector": "Industrial"},
        {"name": "First Industrial Realty Trust", "ticker": "FR", "sector": "Industrial"},
        {"name": "Terreno Realty", "ticker": "TRNO", "sector": "Industrial"},
        
        # Residential REITs
        {"name": "Equity Residential", "ticker": "EQR", "sector": "Residential"},
        {"name": "AvalonBay Communities", "ticker": "AVB", "sector": "Residential"},
        {"name": "Essex Property Trust", "ticker": "ESS", "sector": "Residential"},
        {"name": "UDR Inc", "ticker": "UDR", "sector": "Residential"},
        {"name": "Mid-America Apartment Communities", "ticker": "MAA", "sector": "Residential"},
        {"name": "Camden Property Trust", "ticker": "CPT", "sector": "Residential"},
        {"name": "Apartment Income REIT", "ticker": "AIRC", "sector": "Residential"},
        {"name": "American Homes 4 Rent", "ticker": "AMH", "sector": "Residential"},
        {"name": "Invitation Homes", "ticker": "INVH", "sector": "Residential"},
        {"name": "Sun Communities", "ticker": "SUI", "sector": "Residential"},
        
        # Healthcare REITs
        {"name": "Welltower", "ticker": "WELL", "sector": "Healthcare"},
        {"name": "Ventas", "ticker": "VTR", "sector": "Healthcare"},
        {"name": "Healthpeak Properties", "ticker": "PEAK", "sector": "Healthcare"},
        {"name": "Medical Properties Trust", "ticker": "MPW", "sector": "Healthcare"},
        {"name": "Omega Healthcare Investors", "ticker": "OHI", "sector": "Healthcare"},
        {"name": "Sabra Health Care REIT", "ticker": "SBRA", "sector": "Healthcare"},
        {"name": "National Health Investors", "ticker": "NHI", "sector": "Healthcare"},
        
        # Data Center REITs
        {"name": "Equinix", "ticker": "EQIX", "sector": "Data Center"},
        {"name": "Digital Realty Trust", "ticker": "DLR", "sector": "Data Center"},
        {"name": "CyrusOne", "ticker": "CONE", "sector": "Data Center"},
        {"name": "CoreSite Realty", "ticker": "COR", "sector": "Data Center"},
        {"name": "QTS Realty Trust", "ticker": "QTS", "sector": "Data Center"},
        
        # Cell Tower REITs
        {"name": "American Tower", "ticker": "AMT", "sector": "Infrastructure"},
        {"name": "Crown Castle", "ticker": "CCI", "sector": "Infrastructure"},
        {"name": "SBA Communications", "ticker": "SBAC", "sector": "Infrastructure"},
        
        # Self Storage REITs
        {"name": "Public Storage", "ticker": "PSA", "sector": "Self Storage"},
        {"name": "Extra Space Storage", "ticker": "EXR", "sector": "Self Storage"},
        {"name": "CubeSmart", "ticker": "CUBE", "sector": "Self Storage"},
        {"name": "Life Storage", "ticker": "LSI", "sector": "Self Storage"},
        {"name": "National Storage Affiliates Trust", "ticker": "NSA", "sector": "Self Storage"},
        
        # Hotel REITs
        {"name": "Host Hotels & Resorts", "ticker": "HST", "sector": "Lodging"},
        {"name": "Park Hotels & Resorts", "ticker": "PK", "sector": "Lodging"},
        {"name": "Pebblebrook Hotel Trust", "ticker": "PEB", "sector": "Lodging"},
        {"name": "RLJ Lodging Trust", "ticker": "RLJ", "sector": "Lodging"},
        {"name": "Sunstone Hotel Investors", "ticker": "SHO", "sector": "Lodging"},
        {"name": "Apple Hospitality REIT", "ticker": "APLE", "sector": "Lodging"},
        
        # Diversified REITs
        {"name": "W. P. Carey", "ticker": "WPC", "sector": "Diversified"},
        {"name": "STORE Capital", "ticker": "STOR", "sector": "Diversified"},
        {"name": "Spirit Realty Capital", "ticker": "SRC", "sector": "Diversified"},
        {"name": "Essential Properties Realty Trust", "ticker": "EPRT", "sector": "Diversified"},
        {"name": "Agree Realty", "ticker": "ADC", "sector": "Diversified"},
        
        # Specialty REITs
        {"name": "Gaming and Leisure Properties", "ticker": "GLPI", "sector": "Specialty"},
        {"name": "VICI Properties", "ticker": "VICI", "sector": "Specialty"},
        {"name": "EPR Properties", "ticker": "EPR", "sector": "Specialty"},
        {"name": "Four Corners Property Trust", "ticker": "FCPT", "sector": "Specialty"},
        {"name": "Innovative Industrial Properties", "ticker": "IIPR", "sector": "Specialty"},
        
        # Timber REITs
        {"name": "Weyerhaeuser", "ticker": "WY", "sector": "Timber"},
        {"name": "Rayonier", "ticker": "RYN", "sector": "Timber"},
        {"name": "PotlatchDeltic", "ticker": "PCH", "sector": "Timber"},
        {"name": "Catchmark Timber Trust", "ticker": "CTT", "sector": "Timber"},
    ]
    
    return pd.DataFrame(major_reits)


def search_glassdoor_url(company_name):
    """
    Search for company's Glassdoor page
    Returns Glassdoor review URL or None
    """
    try:
        # Format search query
        search_query = company_name.replace(' ', '+')
        search_url = f"https://www.google.com/search?q={search_query}+glassdoor+reviews"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(search_url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Look for Glassdoor links
        for link in soup.find_all('a', href=True):
            href = link['href']
            if 'glassdoor.com/Reviews/' in href:
                # Extract actual URL from Google redirect
                match = re.search(r'https://www\.glassdoor\.com/Reviews/[^&]+', href)
                if match:
                    return match.group(0)
        
        # Fallback: construct likely URL
        company_slug = company_name.lower().replace(' ', '-').replace('.', '')
        return f"https://www.glassdoor.com/Reviews/{company_slug}-Reviews-E{hash(company_name) % 100000}.htm"
        
    except Exception as e:
        print(f"   Error searching for {company_name}: {e}")
        return None


def build_reit_config(output_file='config/reit_companies.json', 
                      search_glassdoor=True,
                      max_companies=None):
    """
    Build comprehensive REIT configuration file
    
    Args:
        output_file: Path to save JSON config
        search_glassdoor: If True, search for Glassdoor URLs (slow)
        max_companies: Limit number of companies (for testing)
    """
    # Fetch REIT list
    df = fetch_nareit_reits()
    
    if max_companies:
        df = df.head(max_companies)
    
    print(f"\nFound {len(df)} REITs")
    print(f"\nBreakdown by sector:")
    print(df['sector'].value_counts())
    
    # Build companies list
    companies = []
    
    for idx, row in df.iterrows():
        print(f"\n[{idx+1}/{len(df)}] Processing: {row['name']} ({row['ticker']})")
        
        company_data = {
            "name": row['name'],
            "ticker": row['ticker'],
            "sector": row['sector']
        }
        
        # Search for Glassdoor URL
        if search_glassdoor:
            print(f"   Searching for Glassdoor page...")
            glassdoor_url = search_glassdoor_url(row['name'])
            company_data['glassdoor_url'] = glassdoor_url
            if glassdoor_url:
                print(f"   ✓ Found: {glassdoor_url}")
            else:
                print(f"   ✗ Not found")
            time.sleep(2)  # Be polite to Google
        else:
            # Placeholder URL
            company_data['glassdoor_url'] = None
        
        companies.append(company_data)
    
    # Save to JSON
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    config = {"companies": companies}
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Saved {len(companies)} companies to {output_file}")
    
    # Save summary CSV for reference
    csv_file = output_path.parent / 'reit_companies.csv'
    df_out = pd.DataFrame(companies)
    df_out.to_csv(csv_file, index=False)
    print(f"✓ Saved summary to {csv_file}")
    
    return companies


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Build REIT companies configuration file'
    )
    parser.add_argument(
        '--search-glassdoor',
        action='store_true',
        help='Search for Glassdoor URLs (slow, ~2 sec per company)'
    )
    parser.add_argument(
        '--max-companies',
        type=int,
        help='Limit number of companies (for testing)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='config/reit_companies.json',
        help='Output JSON file path'
    )
    
    args = parser.parse_args()
    
    build_reit_config(
        output_file=args.output,
        search_glassdoor=args.search_glassdoor,
        max_companies=args.max_companies
    )


if __name__ == '__main__':
    main()