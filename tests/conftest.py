import pytest
import os
import sys
from pathlib import Path

# Add project root to Python path (backup if pip install -e . not used)
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Define test markers
def pytest_configure(config):
    config.addinivalue_line("markers", "smoke: quick smoke tests")
    config.addinivalue_line("markers", "slow: slow running tests")
    config.addinivalue_line("markers", "integration: integration tests")

# Common fixtures
@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Create temporary test data directory"""
    return tmp_path_factory.mktemp("test_data")

@pytest.fixture
def mock_credentials():
    """Mock credentials for testing"""
    return {
        'username': 'test@example.com',
        'password': 'test_password'
    }
