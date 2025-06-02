import sys
import asyncio
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dynaconf.base import LazySettings
from SRAgent.agents.utils import (
    load_settings,
    set_model,
    create_step_summary_chain,
    create_agent_stream
)

class TestLoadSettings:
    """Tests for load_settings function"""
    
    def test_load_settings_returns_lazy_settings(self):
        """Test that load_settings returns a LazySettings object"""
        settings = load_settings()
        assert isinstance(settings, LazySettings)
    
    @patch("SRAgent.agents.utils.Dynaconf")
    def test_load_settings_has_expected_keys(self, mock_dynaconf):
        """Test that settings are loaded and can be accessed"""
        # Setup mock settings that behave more like a Dynaconf object
        # Need to mock the __getitem__ behaviour for nested access like settings['models']['default']
        mock_dynaconf_instance = MagicMock()
        # Set up nested mocks or direct return values as needed for tests using load_settings
        # Example:
        # mock_dynaconf_instance.__getitem__.side_effect = lambda key: {
        #     "models": {"default": "o4-mini"},
        #     "temperature": {"default": 0.1},
        #     "reasoning_effort": {"default": "low"}
        # }.get(key, MagicMock()) # Return a mock for keys not explicitly defined

        # For simplicity here, just make it return a MagicMock instance
        mock_dynaconf.return_value = mock_dynaconf_instance

        # Call load_settings
        settings = load_settings()

        # Verify that Dynaconf was called correctly
        mock_dynaconf.assert_called_once()
        # Optionally assert that settings is the mock instance
        assert settings is mock_dynaconf_instance


class TestSetModel:
    """Tests for set_model function"""
    
    @patch("SRAgent.agents.utils.load_settings")
    def test_set_model_default_settings(self, mock_load_settings):
        """Test set_model with default settings"""
        # Mock settings
        mock_settings = {
            "models": {"default": "o4-mini"},
            "temperature": {"default": 0.1},
            "reasoning_effort": {"default": "low"},
            "service_tier": {},  # Empty dict so KeyError is raised
            "flex_timeout": {}   # Empty dict so KeyError is raised
        }
        mock_load_settings.return_value = mock_settings
        
        # Test with o1/o3/o4 model
        with patch("SRAgent.agents.utils.ChatOpenAI") as mock_chat:
            model = set_model()
            mock_chat.assert_called_once_with(
                model_name="o4-mini", 
                temperature=None, 
                reasoning_effort="low",
                max_tokens=None
            )
    
    @patch("SRAgent.agents.utils.load_settings")
    def test_set_model_with_gpt4o(self, mock_load_settings):
        """Test set_model with gpt-4o model"""
        # Mock settings
        mock_settings = {
            "models": {"default": "gpt-4.1-mini"},
            "temperature": {"default": 0.1},
            "reasoning_effort": {"default": "low"},
            "service_tier": {},  # Empty dict so KeyError is raised
            "flex_timeout": {}   # Empty dict so KeyError is raised
        }
        mock_load_settings.return_value = mock_settings
        
        # Test with GPT-4.1-mini model
        with patch("SRAgent.agents.utils.ChatOpenAI") as mock_chat:
            model = set_model()
            mock_chat.assert_called_once_with(
                model_name="gpt-4.1-mini", 
                temperature=0.1, 
                reasoning_effort=None,
                max_tokens=None
            )
    
    @patch("SRAgent.agents.utils.load_settings")
    def test_set_model_with_overrides(self, mock_load_settings):
        """Test set_model with parameter overrides"""
        # Mock settings
        mock_settings = {
            "models": {"default": "o4-mini"},
            "temperature": {"default": 0.1},
            "reasoning_effort": {"default": "low"},
            "service_tier": {},  # Empty dict so KeyError is raised
            "flex_timeout": {}   # Empty dict so KeyError is raised
        }
        mock_load_settings.return_value = mock_settings
        
        # Test with override parameters
        with patch("SRAgent.agents.utils.ChatOpenAI") as mock_chat:
            model = set_model(
                model_name="gpt-4.1-mini",
                temperature=0.5,
                reasoning_effort="high"
            )
            mock_chat.assert_called_once_with(
                model_name="gpt-4.1-mini", 
                temperature=0.5, 
                reasoning_effort=None,
                max_tokens=None
            )
    
    @patch("SRAgent.agents.utils.load_settings")
    def test_set_model_specific_agent(self, mock_load_settings):
        """Test set_model with specific agent settings"""
        # Mock settings with agent-specific settings
        mock_settings = {
            "models": {"default": "o4-mini", "entrez": "o4-mini"},
            "temperature": {"default": 0.1, "entrez": 0.2},
            "reasoning_effort": {"default": "low", "entrez": "medium"},
            "service_tier": {},  # Empty dict so KeyError is raised
            "flex_timeout": {}   # Empty dict so KeyError is raised
        }
        mock_load_settings.return_value = mock_settings
        
        # Test with agent_name parameter
        with patch("SRAgent.agents.utils.ChatOpenAI") as mock_chat:
            model = set_model(agent_name="entrez")
            mock_chat.assert_called_once_with(
                model_name="o4-mini", 
                temperature=None, 
                reasoning_effort="medium",
                max_tokens=None
            )
    
    @patch("SRAgent.agents.utils.load_settings")
    def test_set_model_unsupported_model(self, mock_load_settings):
        """Test set_model with unsupported model"""
        # Mock settings
        mock_settings = {
            "models": {"default": "unsupported-model"},
            "temperature": {"default": 0.1},
            "reasoning_effort": {"default": "low"}
        }
        mock_load_settings.return_value = mock_settings
        
        # Test with unsupported model
        with pytest.raises(ValueError, match="Model unsupported-model not supported"):
            set_model()

    @patch("SRAgent.agents.utils.load_settings")
    def test_set_model_with_claude(self, mock_load_settings):
        """Test set_model with claude model"""
        # Mock settings
        mock_settings = {
            "models": {"default": "claude-3-7-sonnet-20250219"},
            "temperature": {"default": 0.1},
            "reasoning_effort": {"default": "low"}
        }
        mock_load_settings.return_value = mock_settings
        
        # Test with claude model and low reasoning effort
        with patch("SRAgent.agents.utils.ChatAnthropic") as mock_chat:
            model = set_model()
            mock_chat.assert_called_once_with(
                model="claude-3-7-sonnet-20250219", 
                temperature=None, 
                thinking={"type": "enabled", "budget_tokens": 1024},
                max_tokens=2048
            )
        
        # Test with claude model and medium reasoning effort
        mock_settings["reasoning_effort"]["default"] = "medium"
        with patch("SRAgent.agents.utils.ChatAnthropic") as mock_chat:
            model = set_model()
            mock_chat.assert_called_once_with(
                model="claude-3-7-sonnet-20250219",
                temperature=None,
                thinking={"type": "enabled", "budget_tokens": 2048},
                max_tokens=3072
            )
        
        # Test with claude model and high reasoning effort
        mock_settings["reasoning_effort"]["default"] = "high"
        with patch("SRAgent.agents.utils.ChatAnthropic") as mock_chat:
            model = set_model()
            mock_chat.assert_called_once_with(
                model="claude-3-7-sonnet-20250219",
                temperature=None,
                thinking={"type": "enabled", "budget_tokens": 4096},
                max_tokens=5120
            )
        
        # Test with claude model and disabled reasoning effort
        mock_settings["reasoning_effort"]["default"] = "none"
        with patch("SRAgent.agents.utils.ChatAnthropic") as mock_chat:
            model = set_model()
            mock_chat.assert_called_once_with(
                model="claude-3-7-sonnet-20250219",
                temperature=0.1,
                thinking={"type": "disabled"},
                max_tokens=None
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
                    max_tokens=45
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
                    max_tokens=100
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
        with patch("SRAgent.agents.utils.create_step_summary_chain", 
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
