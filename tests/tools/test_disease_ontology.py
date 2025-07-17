import os
import json
import shutil
import pytest
import tempfile
import networkx as nx
from unittest.mock import patch, MagicMock, mock_open
import appdirs

from SRAgent.tools.disease_ontology import (
    get_neighbors,
    query_vector_db,
    query_mondo_ols,
    get_mondo_ontology_graph
)
from SRAgent.tools.vector_db import load_vector_store

# Fixture to mock appdirs.user_cache_dir to return a temp directory
@pytest.fixture
def mock_cache_dir():
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch('appdirs.user_cache_dir', return_value=temp_dir):
            yield temp_dir


# Fixture to create a mock OBO graph
@pytest.fixture
def mock_obo_graph():
    # Create a simple mock graph with a few MONDO nodes
    g = nx.MultiDiGraph()
    
    # Add some test nodes
    test_nodes = {
        "MONDO:0005267": {
            "name": "heart disease",
            "def": "A disease that affects the heart."
        },
        "MONDO:0005180": {
            "name": "Parkinson disease", 
            "def": "A progressive neurodegenerative disorder characterized by tremor."
        },
        "MONDO:0005015": {
            "name": "diabetes mellitus",
            "def": "A group of metabolic disorders characterized by high blood sugar."
        },
        "PATO:0000001": {
            "name": "quality",
            "def": "A dependent entity that inheres in a bearer."
        }
    }
    
    # Add nodes to the graph
    for node_id, attrs in test_nodes.items():
        g.add_node(node_id, **attrs)
    
    # Add edges between nodes
    g.add_edge("MONDO:0005267", "MONDO:0005015")
    g.add_edge("MONDO:0005180", "PATO:0000001")
    
    return g


# Mock for get_mondo_ontology_graph function
@pytest.fixture
def mock_get_ontology_graph(mock_obo_graph):
    with patch('SRAgent.tools.disease_ontology.get_mondo_ontology_graph', return_value=mock_obo_graph):
        yield


# Mock for requests.get
@pytest.fixture
def mock_requests_get():
    with patch('requests.get') as mock_get:
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.content = b"Mock OBO file content"
        mock_response.json.return_value = {
            "response": {
                "docs": [
                    {
                        "obo_id": "MONDO:0005267",
                        "label": "heart disease",
                        "description": ["A disease that affects the heart."],
                        "synonym": ["cardiac disease", "cardiovascular disease"]
                    }
                ]
            }
        }
        mock_get.return_value = mock_response
        yield mock_get


# Mock for load_vector_store function
@pytest.fixture
def mock_chroma():
    with patch('SRAgent.tools.disease_ontology.load_vector_store') as mock_load_vector_store:
        # Configure mock vector store and search results
        mock_result = MagicMock()
        mock_result.metadata = {"id": "MONDO:0005267", "name": "heart disease"}
        mock_result.page_content = "A disease that affects the heart."
        
        mock_vector_store = MagicMock()
        mock_vector_store.similarity_search.return_value = [mock_result]
        mock_load_vector_store.return_value = mock_vector_store
        
        yield mock_load_vector_store


# Mock for tarfile
@pytest.fixture
def mock_tarfile():
    with patch('tarfile.open') as mock_tar:
        mock_tar_instance = MagicMock()
        mock_tar.return_value.__enter__.return_value = mock_tar_instance
        yield mock_tar


# Test get_neighbors function
def test_get_neighbors_invalid_id():
    """Test get_neighbors with invalid MONDO ID format"""
    result = get_neighbors.invoke({"mondo_id": "invalid"})
    assert "Invalid MONDO ID format" in result


def test_get_neighbors_invalid_id_wrong_format():
    """Test get_neighbors with wrong ID format (too short)"""
    result = get_neighbors.invoke({"mondo_id": "MONDO:123"})
    assert "Invalid MONDO ID format" in result


@patch('os.path.exists', return_value=True)
def test_get_neighbors_valid_mondo_id(mock_exists, mock_get_ontology_graph):
    """Test get_neighbors with valid MONDO ID"""
    result = get_neighbors.invoke({"mondo_id": "MONDO:0005267"})
    assert "Neighbors in the ontology for: \"MONDO:0005267\"" in result
    assert "MONDO:0005015" in result
    assert "diabetes mellitus" in result


@patch('os.path.exists', return_value=True)
def test_get_neighbors_valid_pato_id(mock_exists, mock_get_ontology_graph):
    """Test get_neighbors with valid PATO ID"""
    result = get_neighbors.invoke({"mondo_id": "PATO:0000001"})
    assert "Neighbors in the ontology for: \"PATO:0000001\"" in result
    assert "MONDO:0005180" in result
    assert "Parkinson disease" in result


@patch('os.path.exists', return_value=False)
def test_get_neighbors_download_obo(mock_exists, mock_requests_get, mock_cache_dir):
    """Test get_neighbors when OBO file needs to be downloaded"""
    with patch('os.makedirs'):
        with patch('builtins.open', mock_open()):
            with patch('SRAgent.tools.disease_ontology.get_mondo_ontology_graph') as mock_graph:
                # Configure the mock graph to return an empty graph (to avoid processing neighbors)
                mock_graph.return_value = nx.MultiDiGraph()
                
                result = get_neighbors.invoke({"mondo_id": "MONDO:0005267"})
                
                # Verify the download was attempted
                mock_requests_get.assert_called_once()
                assert "https://purl.obolibrary.org/obo/mondo.obo" in mock_requests_get.call_args[0][0]


# Test query_vector_db function
@patch('os.path.exists', return_value=True)
@patch('os.listdir', return_value=['some_file'])
def test_query_vector_db_with_existing_db(mock_listdir, mock_exists, mock_chroma):
    """Test query_vector_db when the Chroma DB already exists"""
    result = query_vector_db.invoke({"query": "heart disease", "k": 3})
    
    assert "Results for query: \"heart disease\"" in result
    assert "MONDO:0005267" in result
    assert "heart disease" in result


def test_query_vector_db_download_db(mock_requests_get, mock_tarfile, mock_cache_dir, mock_chroma):
    """Test query_vector_db when the Chroma DB needs to be downloaded"""
    # Set up a more comprehensive patching strategy
    with patch('os.makedirs'):
        with patch('os.path.exists') as mock_exists:
            # First it checks cache dir, then DB dir, then again for load_vector_store
            mock_exists.side_effect = [True, False, True]
            
            with patch('os.listdir') as mock_listdir:
                # First check is empty, then after extraction it has content
                mock_listdir.side_effect = [[], ['mondo_chroma']]
                
                with patch('tempfile.TemporaryDirectory') as mock_temp_dir:
                    mock_temp_dir.return_value.__enter__.return_value = '/tmp/mocktemp'
                    
                    with patch('os.path.isdir', return_value=True):
                        with patch('shutil.move'):
                            with patch('os.remove'):
                                # Call the function
                                result = query_vector_db.invoke({"query": "heart disease", "k": 3})
                                
                                # Verify the download was attempted
                                mock_requests_get.assert_called_once()
                                assert "storage.googleapis.com" in mock_requests_get.call_args[0][0]
                                assert "mondo_chroma.tar.gz" in mock_requests_get.call_args[0][0]
                                
                                # Check the result contains expected text
                                assert "Results for query: \"heart disease\"" in result


# Test query_mondo_ols function
def test_query_mondo_ols(mock_requests_get):
    """Test query_mondo_ols function"""
    result = query_mondo_ols.invoke({"search_term": "heart disease"})
    
    # Check that the API was called with the correct URL
    mock_requests_get.assert_called_once()
    assert "ebi.ac.uk/ols/api/search" in mock_requests_get.call_args[0][0]
    assert "heart%20disease" in mock_requests_get.call_args[0][0]
    assert "ontology=mondo" in mock_requests_get.call_args[0][0]
    
    # Check the result
    assert "Results from OLS for 'heart disease'" in result
    assert "MONDO:0005267" in result
    assert "heart disease" in result
    assert "cardiac disease" in result  # Check synonyms are included


def test_query_mondo_ols_error_handling():
    """Test query_mondo_ols error handling"""
    with patch('requests.get') as mock_get:
        mock_get.side_effect = Exception("API error")
        
        result = query_mondo_ols.invoke({"search_term": "heart disease"})
        assert "Error querying OLS API after 2 attempts" in result


def test_query_mondo_ols_no_results(mock_requests_get):
    """Test query_mondo_ols when no results are found"""
    mock_requests_get.return_value.json.return_value = {"response": {"docs": []}}
    
    result = query_mondo_ols.invoke({"search_term": "nonexistent_disease"})
    assert "No results found for search term: 'nonexistent_disease'" in result


def test_query_mondo_ols_non_mondo_results(mock_requests_get):
    """Test query_mondo_ols filtering out non-MONDO results"""
    mock_requests_get.return_value.json.return_value = {
        "response": {
            "docs": [
                {
                    "obo_id": "DOID:1234",  # This should be filtered out
                    "label": "some disease",
                    "description": ["A disease from DOID"]
                },
                {
                    "obo_id": "MONDO:0005267",
                    "label": "heart disease", 
                    "description": ["A disease that affects the heart."]
                }
            ]
        }
    }
    
    result = query_mondo_ols.invoke({"search_term": "disease"})
    assert "DOID:1234" not in result  # Should be filtered out
    assert "MONDO:0005267" in result  # Should be included


def test_query_mondo_ols_with_retries():
    """Test query_mondo_ols retry mechanism"""
    with patch('requests.get') as mock_get:
        # First call fails, second succeeds
        mock_get.side_effect = [
            Exception("Network error"),
            MagicMock(json=lambda: {"response": {"docs": []}}, raise_for_status=lambda: None)
        ]
        
        with patch('time.sleep'):  # Mock sleep to speed up test
            result = query_mondo_ols.invoke({"search_term": "test"})
            
            # Should have been called twice
            assert mock_get.call_count == 2
            assert "No results found" in result


def test_query_vector_db_search_error(mock_chroma):
    """Test query_vector_db when search fails"""
    # Mock the vector store to raise an exception during search
    mock_vector_store = MagicMock()
    mock_vector_store.similarity_search.side_effect = Exception("Search failed")
    mock_chroma.return_value = mock_vector_store
    
    with patch('os.path.exists', return_value=True):
        with patch('os.listdir', return_value=['some_file']):
            result = query_vector_db.invoke({"query": "test", "k": 3})
            assert "Error performing search: Search failed" in result 