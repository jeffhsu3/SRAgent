import sys
import asyncio
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dynaconf.base import LazySettings
from SRAgent.agents.display import (
    create_step_summary_chain,
    create_agent_stream
)


class TestCreateStepSummaryChain:
    """Tests for create_step_summary_chain function"""
    
    def test_create_step_summary_chain_output(self):
        """Test that create_step_summary_chain returns the expected object"""
        # Mock load_settings to provide necessary config for step_summary agent
        mock_settings_data = {
            "models": {"default": "gpt-4.1-mini", "step_summary": "gpt-4.1-mini"},
            "temperature": {"default": 0.1, "step_summary": 0},
            "reasoning_effort": {"default": "low", "step_summary": None},
            "max_tokens": {"default": None, "step_summary": 45},
            "service_tier": {},  # Empty dict so KeyError is raised
            "flex_timeout": {}   # Empty dict so KeyError is raised
        }
        # Use a helper to mock dictionary access
        def mock_getitem(key):
            if key in mock_settings_data:
                return mock_settings_data[key]
            raise KeyError(f"{key} does not exist")

        mock_settings = MagicMock()
        mock_settings.__getitem__.side_effect = mock_getitem
        mock_settings.get.side_effect = lambda k, d=None: mock_settings_data.get(k, d) # Mock .get() too

        with patch("SRAgent.agents.utils.load_settings", return_value=mock_settings):
             with patch("SRAgent.agents.utils.ChatOpenAI") as mock_chat:
                chain = create_step_summary_chain()
                # Check that ChatOpenAI was created with expected parameters
                mock_chat.assert_called_once_with(
                    model_name="gpt-4.1-mini",
                    temperature=0,
                    reasoning_effort=None,
                    max_tokens=45,
                    service_tier="default"
                )

                # Check the structure of the returned chain
                assert "RunnableSequence" in str(type(chain))
    
    def test_create_step_summary_chain_with_custom_params(self):
        """Test create_step_summary_chain with custom parameters"""
        # Mock load_settings similar to the previous test
        mock_settings_data = {
            "models": {"default": "gpt-4.1-mini", "step_summary": "gpt-4.1-mini"},
            "temperature": {"default": 0.1, "step_summary": 0},
            "reasoning_effort": {"default": "low", "step_summary": None},
            "max_tokens": {"default": None, "step_summary": 100}, # Will be overridden by argument
            "service_tier": {},  # Empty dict so KeyError is raised
            "flex_timeout": {}   # Empty dict so KeyError is raised
        }
        def mock_getitem(key):
            if key in mock_settings_data:
                return mock_settings_data[key]
            raise KeyError(f"{key} does not exist")

        mock_settings = MagicMock()
        mock_settings.__getitem__.side_effect = mock_getitem
        mock_settings.get.side_effect = lambda k, d=None: mock_settings_data.get(k, d)

        with patch("SRAgent.agents.utils.load_settings", return_value=mock_settings):
             with patch("SRAgent.agents.utils.ChatOpenAI") as mock_chat:
                # Pass the custom max_tokens argument
                chain = create_step_summary_chain(max_tokens=100)
                # Check that ChatOpenAI was created with the custom max_tokens
                mock_chat.assert_called_once_with(
                    model_name="gpt-4.1-mini",
                    temperature=0,
                    reasoning_effort=None,
                    max_tokens=100,
                    service_tier="default"
                )

                # Check the structure of the returned chain
                assert "RunnableSequence" in str(type(chain))


class TestCreateAgentStream:
    """Tests for create_agent_stream function"""
    
    @pytest.mark.asyncio
    async def test_create_agent_stream_basic(self):
        """Test basic functionality of create_agent_stream"""
        # Mock agent with astream
        mock_agent = MagicMock()
        mock_step = {
            "messages": [MagicMock(content="Test message")]
        }
        
        # Create a proper async iterator for astream
        async def mock_astream(*args, **kwargs):
            yield mock_step
        
        mock_agent.astream = mock_astream
        
        # Mock create_agent_func
        mock_create_agent_func = MagicMock(return_value=mock_agent)
        
        # Create proper input format that create_agent_stream expects
        test_input = {
            "messages": [MagicMock(content="Test query")]
        }
        
        # Call create_agent_stream
        with patch("sys.stderr"):  # Redirect stderr to avoid printing during test
            result = await create_agent_stream(test_input, mock_create_agent_func)
        
        # Assert results
        assert result == "Test message"
        mock_create_agent_func.assert_called_once_with(return_tool=False)
    
    @pytest.mark.asyncio
    async def test_create_agent_stream_with_summarization(self):
        """Test create_agent_stream with step summarization"""
        # Mock agent with astream
        mock_agent = MagicMock()
        mock_step = {
            "messages": [MagicMock(content="Test message")]
        }
        
        # Create a proper async iterator for astream
        async def mock_astream(*args, **kwargs):
            yield mock_step
        
        mock_agent.astream = mock_astream
        
        # Mock create_agent_func
        mock_create_agent_func = MagicMock(return_value=mock_agent)
        
        # Mock step summary chain
        mock_summary_chain = MagicMock()
        mock_summary_chain.ainvoke = AsyncMock(return_value=MagicMock(content="Summary"))
        
        # Create proper input format
        test_input = {
            "messages": [MagicMock(content="Test query")]
        }
        
        # Call create_agent_stream with summarize_steps=True
        with patch("SRAgent.agents.display.create_step_summary_chain", 
                   return_value=mock_summary_chain):
            with patch("sys.stderr"):  # Redirect stderr to avoid printing during test
                result = await create_agent_stream(
                    test_input, 
                    mock_create_agent_func, 
                    summarize_steps=True
                )
        
        # Assert results
        assert result == "Test message"
        mock_summary_chain.ainvoke.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_agent_stream_dict_formats(self):
        """Test dictionary formats of final step in create_agent_stream"""
        # Test with agent.messages format
        mock_agent1 = MagicMock()
        mock_step1 = {
            "agent": {
                "messages": [MagicMock(content="Final agent message")]
            }
        }
        
        async def mock_astream1(*args, **kwargs):
            yield mock_step1
        
        mock_agent1.astream = mock_astream1
        mock_create_agent_func1 = MagicMock(return_value=mock_agent1)
        
        test_input1 = {
            "messages": [MagicMock(content="Test query")]
        }
        
        with patch("sys.stderr"):
            result1 = await create_agent_stream(test_input1, mock_create_agent_func1)
        assert result1 == "Final agent message"
        
        # Test with messages format
        mock_agent2 = MagicMock()
        mock_step2 = {
            "messages": [MagicMock(content="Regular message")]
        }
        
        async def mock_astream2(*args, **kwargs):
            yield mock_step2
        
        mock_agent2.astream = mock_astream2
        mock_create_agent_func2 = MagicMock(return_value=mock_agent2)
        
        test_input2 = {
            "messages": [MagicMock(content="Test query")]
        }
        
        with patch("sys.stderr"):
            result2 = await create_agent_stream(test_input2, mock_create_agent_func2)
        assert result2 == "Regular message"
    
    @pytest.mark.asyncio
    async def test_create_agent_stream_string_format(self):
        """Test string format of final step in create_agent_stream"""
        # This test aims to check if the function handles a final step being a simple string
        # Modify the mock agent's astream to yield a string directly
        mock_agent = MagicMock()

        async def mock_astream_string(*args, **kwargs):
            yield "Plain string final step"

        mock_agent.astream = mock_astream_string
        mock_create_agent_func = MagicMock(return_value=mock_agent)

        test_input = {
            "messages": [MagicMock(content="Test query")]
        }

        # Call create_agent_stream
        with patch("sys.stderr"):  # Redirect stderr
            result = await create_agent_stream(test_input, mock_create_agent_func)

        # Assert the final string is returned
        # NOTE: The original implementation likely expects a dict-like structure.
        # This test might need adjustment based on how string steps are *actually* handled.
        # For now, assuming it should return the string if that's the final yield.
        assert result == "Plain string final step" 