import os
import sys
import asyncio
import pytest
from unittest.mock import patch, MagicMock, AsyncMock, Mock
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai.chat_models.base import OpenAIRefusalError
from pydantic import BaseModel, Field

# Import the modules we're testing
from SRAgent.workflows.tissue_ontology import create_tissue_ontology_workflow, UBERON_ID, UBERON_IDS
from SRAgent.workflows.metadata import create_get_metadata_node, AllMetadataEnum
from SRAgent.workflows.convert import create_get_accessions_node, Acessions
from SRAgent.workflows.find_datasets import create_get_entrez_ids_node, EntrezInfo


class TestTissueOntologyRefusalError:
    """Test OpenAIRefusalError handling in tissue ontology workflow"""
    
    @pytest.mark.asyncio
    async def test_tissue_ontology_handles_refusal(self):
        """Test that tissue ontology workflow handles OpenAIRefusalError gracefully"""
        # Create a mock agent that raises OpenAIRefusalError
        mock_agent = AsyncMock()
        mock_agent.ainvoke.side_effect = OpenAIRefusalError(
            "I'm afraid 'tumor' by itself isn't mappable to a specific Uberon term"
        )
        
        # Patch the create_react_agent to return our mock
        with patch('SRAgent.workflows.tissue_ontology.create_react_agent', return_value=mock_agent):
            with patch('SRAgent.workflows.tissue_ontology.set_model'):
                with patch('SRAgent.workflows.tissue_ontology.create_tissue_ontology_agent'):
                    workflow = create_tissue_ontology_workflow()
                    
                    # Test invoking with "tumor" which should cause refusal
                    messages = [HumanMessage(content="Tissues: tumor")]
                    result = await workflow.ainvoke({"messages": messages}, config={})
                    
                    # Should return empty list instead of raising error
                    assert result == []
                    
    @pytest.mark.asyncio
    async def test_tissue_ontology_normal_operation(self):
        """Test that tissue ontology workflow works normally when no refusal"""
        # Create a mock successful response
        mock_response = {
            'structured_response': Mock(
                ids=[
                    Mock(id="UBERON:0000955"),  # brain
                    Mock(id="UBERON:0002048"),  # lung
                ]
            )
        }
        
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = mock_response
        
        with patch('SRAgent.workflows.tissue_ontology.create_react_agent', return_value=mock_agent):
            with patch('SRAgent.workflows.tissue_ontology.set_model'):
                with patch('SRAgent.workflows.tissue_ontology.create_tissue_ontology_agent'):
                    workflow = create_tissue_ontology_workflow()
                    
                    messages = [HumanMessage(content="Tissues: brain, lung")]
                    result = await workflow.ainvoke({"messages": messages}, config={})
                    
                    # Should return the UBERON IDs
                    assert result == ["UBERON:0000955", "UBERON:0002048"]


class TestMetadataExtractionRefusalError:
    """Test OpenAIRefusalError handling in metadata extraction"""
    
    @pytest.mark.asyncio
    async def test_metadata_extraction_handles_refusal(self):
        """Test that metadata extraction handles OpenAIRefusalError with retries"""
        # Create a proper mock response object that mimics AllMetadataEnum
        mock_response = Mock()
        mock_response.model_fields = AllMetadataEnum.model_fields
        
        # Set up attributes to match what get_extracted_fields expects
        mock_response.is_illumina = "yes"
        mock_response.is_single_cell = "yes"  
        mock_response.is_paired_end = "yes"
        mock_response.lib_prep = "10x_Genomics"
        mock_response.tech_10x = "3_prime_gex"
        mock_response.cell_prep = "single_cell"
        mock_response.organism = "Homo sapiens"
        mock_response.tissue = "brain"
        mock_response.disease = "none"
        mock_response.perturbation = "none"
        mock_response.cell_line = "none"
        
        # Create a mock that first raises errors, then succeeds
        def side_effect_func(*args, **kwargs):
            call_count = getattr(side_effect_func, 'call_count', 0)
            side_effect_func.call_count = call_count + 1
            
            if call_count < 2:  # First two calls fail
                raise OpenAIRefusalError(f"Cannot determine metadata (attempt {call_count + 1})")
            else:  # Third call succeeds
                return mock_response
        
        # Mock the complete model structure
        mock_structured_output = AsyncMock()
        mock_structured_output.ainvoke.side_effect = side_effect_func
        
        mock_model = Mock()
        mock_model.with_structured_output.return_value = mock_structured_output
        
        with patch('SRAgent.workflows.metadata.set_model', return_value=mock_model):
            node_func = create_get_metadata_node()
            
            state = {
                "messages": [
                    HumanMessage(content="Some metadata about a dataset")
                ]
            }
            
            result = await node_func(state, config={})
            
            # Should succeed after retries
            assert result["is_illumina"] == "yes"
            assert result["organism"] == "Homo sapiens"
            assert mock_structured_output.ainvoke.call_count == 3
    
    @pytest.mark.asyncio
    async def test_metadata_extraction_all_retries_fail(self):
        """Test metadata extraction when all retries fail"""
        # Mock that always raises OpenAIRefusalError
        mock_structured_output = AsyncMock()
        mock_structured_output.ainvoke.side_effect = OpenAIRefusalError("Persistent refusal")
        
        mock_model = Mock()
        mock_model.with_structured_output.return_value = mock_structured_output
        
        with patch('SRAgent.workflows.metadata.set_model', return_value=mock_model):
            with patch('sys.stderr'):  # Suppress error prints during test
                node_func = create_get_metadata_node()
                
                state = {
                    "messages": [
                        HumanMessage(content="Some metadata")
                    ]
                }
                
                result = await node_func(state, config={})
                
                # Should return default values
                assert result["is_illumina"] == "unsure"
                assert result["is_single_cell"] == "unsure"
                assert result["lib_prep"] == "other"
                assert result["organism"] == "other"
                assert result["tissue"] == "unknown"


class TestAccessionExtractionRefusalError:
    """Test OpenAIRefusalError handling in accession extraction"""
    
    @pytest.mark.asyncio
    async def test_accession_extraction_handles_refusal(self):
        """Test that accession extraction handles OpenAIRefusalError"""
        # Mock extract_accessions to return empty list (no regex match)
        with patch('SRAgent.workflows.convert.extract_accessions', return_value=[]):
            # Create a mock response for successful third attempt
            mock_response = Mock()
            mock_response.srx = ["SRX123456", "ERX789012"]
            
            # Create a mock that first raises errors, then succeeds
            def side_effect_func(*args, **kwargs):
                call_count = getattr(side_effect_func, 'call_count', 0)
                side_effect_func.call_count = call_count + 1
                
                if call_count < 2:  # First two calls fail
                    raise OpenAIRefusalError(f"No accessions found (attempt {call_count + 1})")
                else:  # Third call succeeds
                    return mock_response
            
            mock_structured_output = AsyncMock()
            mock_structured_output.ainvoke.side_effect = side_effect_func
            
            mock_model = Mock()
            mock_model.with_structured_output.return_value = mock_structured_output
            
            with patch('SRAgent.workflows.convert.set_model', return_value=mock_model):
                with patch('sys.stderr'):  # Suppress error prints
                    node_func = create_get_accessions_node()
                    
                    state = {
                        "messages": [
                            AIMessage(content="Here are accessions SRX123456 and ERX789012")
                        ]
                    }
                    
                    result = await node_func(state)
                    
                    # Should succeed after retries
                    assert result["SRX"] == ["SRX123456", "ERX789012"]
                    assert mock_structured_output.ainvoke.call_count == 3
    
    @pytest.mark.asyncio
    async def test_accession_extraction_returns_empty_on_persistent_failure(self):
        """Test accession extraction returns empty list when all attempts fail"""
        with patch('SRAgent.workflows.convert.extract_accessions', return_value=[]):
            mock_structured_output = AsyncMock()
            mock_structured_output.ainvoke.side_effect = OpenAIRefusalError("No valid accessions")
            
            mock_model = Mock()
            mock_model.with_structured_output.return_value = mock_structured_output
            
            with patch('SRAgent.workflows.convert.set_model', return_value=mock_model):
                with patch('sys.stderr'):
                    node_func = create_get_accessions_node()
                    
                    state = {
                        "messages": [
                            AIMessage(content="No valid accessions here")
                        ]
                    }
                    
                    result = await node_func(state)
                    
                    # Should return empty list
                    assert result["SRX"] == []


class TestEntrezIDExtractionRefusalError:
    """Test OpenAIRefusalError handling in Entrez ID extraction"""
    
    @pytest.mark.asyncio
    async def test_entrez_id_extraction_handles_refusal(self):
        """Test that Entrez ID extraction handles OpenAIRefusalError"""
        # Create a mock response for successful attempt
        mock_response = Mock()
        mock_response.entrez_ids = [12345, 67890]
        mock_response.database = "sra"
        
        # Create a mock that first raises errors, then succeeds
        def side_effect_func(*args, **kwargs):
            call_count = getattr(side_effect_func, 'call_count', 0)
            side_effect_func.call_count = call_count + 1
            
            if call_count < 2:  # First two calls fail
                raise OpenAIRefusalError(f"Cannot extract IDs (attempt {call_count + 1})")
            else:  # Third call succeeds
                return mock_response
        
        mock_structured_output = AsyncMock()
        mock_structured_output.ainvoke.side_effect = side_effect_func
        
        mock_model = Mock()
        mock_model.with_structured_output.return_value = mock_structured_output
        
        with patch('SRAgent.workflows.find_datasets.set_model', return_value=mock_model):
            with patch('sys.stderr'):
                node_func = create_get_entrez_ids_node()
                
                state = {
                    "messages": [
                        AIMessage(content="Found datasets 12345 and 67890 in SRA database")
                    ]
                }
                
                config = {
                    "configurable": {
                        "use_database": False,
                        "max_datasets": 10
                    }
                }
                
                result = await node_func(state, config)
                
                # Should succeed after retries
                assert result["entrez_ids"] == [12345, 67890]
                assert result["database"] == "sra"
                assert mock_structured_output.ainvoke.call_count == 3
    
    @pytest.mark.asyncio
    async def test_entrez_id_extraction_returns_empty_on_failure(self):
        """Test Entrez ID extraction returns empty when all attempts fail"""
        mock_structured_output = AsyncMock()
        mock_structured_output.ainvoke.side_effect = OpenAIRefusalError("No IDs found")
        
        mock_model = Mock()
        mock_model.with_structured_output.return_value = mock_structured_output
        
        with patch('SRAgent.workflows.find_datasets.set_model', return_value=mock_model):
            with patch('sys.stderr'):
                node_func = create_get_entrez_ids_node()
                
                state = {
                    "messages": [
                        AIMessage(content="No clear dataset information")
                    ]
                }
                
                config = {
                    "configurable": {
                        "use_database": False
                    }
                }
                
                result = await node_func(state, config)
                
                # Should return empty values
                assert result["entrez_ids"] == []
                assert result["database"] == ""


# Integration test showing the full error in context
class TestOpenAIRefusalErrorIntegration:
    """Integration test showing the original error scenario"""
    
    @pytest.mark.asyncio
    async def test_tumor_without_context_workflow(self):
        """Test the specific case of 'tumor' without tissue context that caused the original error"""
        # This simulates the exact scenario from the error message
        mock_agent = AsyncMock()
        mock_agent.ainvoke.side_effect = OpenAIRefusalError(
            "I'm afraid \"tumor\" by itself isn't mappable to a specific Uberon term—"
            "you need to specify the tissue or organ of origin"
        )
        
        with patch('SRAgent.workflows.tissue_ontology.create_react_agent', return_value=mock_agent):
            with patch('SRAgent.workflows.tissue_ontology.set_model'):
                with patch('SRAgent.workflows.tissue_ontology.create_tissue_ontology_agent'):
                    workflow = create_tissue_ontology_workflow()
                    
                    # This is what would have caused the original error
                    messages = [HumanMessage(content="Tissues: tumor")]
                    
                    # Should not raise an exception
                    result = await workflow.ainvoke({"messages": messages}, config={})
                    
                    # Should return empty list, allowing the workflow to continue
                    assert result == []


