import os
import re
import sys
import asyncio
from functools import wraps
from importlib import resources
from typing import Dict, Any, Optional, Union
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.base import BaseLanguageModel
from dynaconf import Dynaconf
import openai

def load_settings() -> Dict[str, Any]:
    """
    Load settings from settings.yml file
    
    Args:
        env: Environment to load settings for ('test' or 'prod')
        
    Returns:
        Dictionary containing settings for the specified environment
    """
    # get path to settings
    if os.getenv("DYNACONF_SETTINGS_PATH"):
        s_path = os.getenv("DYNACONF_SETTINGS_PATH")
    else:
        s_path = str(resources.files("SRAgent").joinpath("settings.yml"))
    if not os.path.exists(s_path):
        raise FileNotFoundError(f"Settings file not found: {s_path}")
    settings = Dynaconf(
        settings_files=[s_path], 
        environments=True, 
        env_switcher="DYNACONF"
    )
    return settings

def create_openai_model_with_retry(
    model_name: str,
    temperature: Optional[float],
    reasoning_effort: Optional[str],
    max_tokens: Optional[int],
    service_tier: Optional[str],
    timeout: Optional[float] = 60.0,
) -> BaseLanguageModel:
    """
    Create an OpenAI model with retry logic for flex tier fallback.
    
    Args:
        model_name: The model name
        temperature: Temperature setting
        reasoning_effort: Reasoning effort level
        max_tokens: Maximum tokens
        service_tier: Service tier ("flex" or "default")
        timeout: Timeout in seconds for flex tier requests
        
    Returns:
        Configured ChatOpenAI model instance
    """
    # Check if flex tier is allowed for this model
    flex_allowed_models = ["o3-mini", "o4-mini"]
    model_supports_flex = any(model_name.startswith(prefix) for prefix in flex_allowed_models)
    
    # If flex requested but not supported, fall back to default
    if service_tier == "flex" and not model_supports_flex:
        print(f"Warning: Flex tier not supported for model {model_name}, using default tier", file=sys.stderr)
        service_tier = "default"
    
    # Determine if this is an o[0-9] model
    is_o_model = re.match(r"^o[0-9]-", model_name) is not None
    
    # Create kwargs for ChatOpenAI
    kwargs = {
        "model_name": model_name,
        "max_tokens": max_tokens,
    }
    
    # Add temperature or reasoning_effort based on model type
    if is_o_model:
        kwargs["temperature"] = None
        kwargs["reasoning_effort"] = reasoning_effort
    else:
        kwargs["temperature"] = temperature
        kwargs["reasoning_effort"] = None
    
    # Add service tier if specified
    if service_tier and service_tier != "default":
        kwargs["service_tier"] = service_tier
        if timeout:
            kwargs["timeout"] = timeout
    
    return ChatOpenAI(**kwargs)

def async_retry_on_flex_timeout(func):
    """
    Async decorator to retry with default tier if flex tier times out.
    """
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        # Check if we're using flex tier
        service_tier = getattr(self, '_service_tier', None)
        model_name = getattr(self, 'model_name', None)
        
        if service_tier != "flex":
            # Not using flex tier, just call the function normally
            return await func(self, *args, **kwargs)
        
        try:
            # Try with flex tier first
            return await func(self, *args, **kwargs)
        except (asyncio.TimeoutError, openai.APITimeoutError) as e:
            print(f"Flex tier timeout for model {model_name}, retrying with default tier...", file=sys.stderr)
            
            # Create a new instance with default tier
            if hasattr(self, '_fallback_model'):
                # Use pre-created fallback model if available
                fallback_model = self._fallback_model
            else:
                # Create fallback model on the fly
                fallback_kwargs = {
                    "model_name": self.model_name,
                    "temperature": getattr(self, 'temperature', None),
                    "max_tokens": getattr(self, 'max_tokens', None),
                }
                # Add reasoning_effort if it's an o-model
                if hasattr(self, 'reasoning_effort'):
                    fallback_kwargs["reasoning_effort"] = self.reasoning_effort
                fallback_model = ChatOpenAI(**fallback_kwargs)
            
            # Retry with default tier
            return await fallback_model.ainvoke(*args, **kwargs)
        except Exception as e:
            # For other exceptions, just raise them
            raise
    
    return wrapper

def sync_retry_on_flex_timeout(func):
    """
    Sync decorator to retry with default tier if flex tier times out.
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # Check if we're using flex tier
        service_tier = getattr(self, '_service_tier', None)
        model_name = getattr(self, 'model_name', None)
        
        if service_tier != "flex":
            # Not using flex tier, just call the function normally
            return func(self, *args, **kwargs)
        
        try:
            # Try with flex tier first
            return func(self, *args, **kwargs)
        except (openai.APITimeoutError,) as e:
            print(f"Flex tier timeout for model {model_name}, retrying with default tier...", file=sys.stderr)
            
            # Create a new instance with default tier
            if hasattr(self, '_fallback_model'):
                # Use pre-created fallback model if available
                fallback_model = self._fallback_model
            else:
                # Create fallback model on the fly
                fallback_kwargs = {
                    "model_name": self.model_name,
                    "temperature": getattr(self, 'temperature', None),
                    "max_tokens": getattr(self, 'max_tokens', None),
                }
                # Add reasoning_effort if it's an o-model
                if hasattr(self, 'reasoning_effort'):
                    fallback_kwargs["reasoning_effort"] = self.reasoning_effort
                fallback_model = ChatOpenAI(**fallback_kwargs)
            
            # Retry with default tier
            return fallback_model.invoke(*args, **kwargs)
        except Exception as e:
            # For other exceptions, just raise them
            raise
    
    return wrapper

class FlexTierChatOpenAI(ChatOpenAI):
    """
    Extended ChatOpenAI that supports automatic fallback from flex to default tier.
    """
    def __init__(self, *args, service_tier: Optional[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._service_tier = service_tier
        
        # Create fallback model if using flex tier
        if service_tier == "flex":
            fallback_kwargs = kwargs.copy()
            fallback_kwargs.pop('service_tier', None)
            fallback_kwargs.pop('timeout', None)
            self._fallback_model = ChatOpenAI(**fallback_kwargs)
    
    @async_retry_on_flex_timeout
    async def ainvoke(self, *args, **kwargs):
        return await super().ainvoke(*args, **kwargs)
    
    @sync_retry_on_flex_timeout
    def invoke(self, *args, **kwargs):
        return super().invoke(*args, **kwargs)

def set_model(
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    reasoning_effort: Optional[str] = None,
    agent_name: str = "default",
    max_tokens: Optional[int] = None,
    service_tier: Optional[str] = None,
) -> Any:
    """
    Create a model instance with settings from configuration
    Args:
        model_name: Override model name from settings
        temperature: Override temperature from settings
        reasoning_effort: Override reasoning effort from settings
        agent_name: Name of the agent to get settings for
        max_tokens: Maximum number of tokens to use for the model
        service_tier: Override service tier from settings ("flex" or "default")
    Returns:
        Configured model instance
    """
    # Load settings
    settings = load_settings()
    
    # Use provided params or get from settings
    if model_name is None:
        try:
            model_name = settings["models"][agent_name]
        except KeyError:
            # try default
            try:
                model_name = settings["models"]["default"]
            except KeyError:
                raise ValueError(f"No model name was provided for agent '{agent_name}'")
    if temperature is None:
        try:
            temperature = settings["temperature"][agent_name]
        except KeyError:
            try:
                temperature = settings["temperature"]["default"]
            except KeyError:
                raise ValueError(f"No temperature was provided for agent '{agent_name}'")
    if reasoning_effort is None:
        try:
            reasoning_effort = settings["reasoning_effort"][agent_name]
        except KeyError:
            try:
                reasoning_effort = settings["reasoning_effort"]["default"]
            except KeyError:
                if temperature is None:
                    raise ValueError(f"No reasoning effort or temperature was provided for agent '{agent_name}'")
    
    # Get service tier from settings if not provided
    if service_tier is None:
        try:
            service_tier = settings["service_tier"][agent_name]
        except KeyError:
            try:
                service_tier = settings["service_tier"]["default"]
            except KeyError:
                service_tier = "default"  # Default to standard tier
    
    # Get timeout from settings
    timeout = None
    try:
        timeout = settings["flex_timeout"][agent_name]
    except KeyError:
        try:
            timeout = settings["flex_timeout"]["default"]
        except KeyError:
            timeout = 60.0  # Default 60 seconds timeout

    # Check model provider and initialize appropriate model
    if model_name.startswith("claude"): # e.g.,  "claude-3-7-sonnet-20250219"
        if reasoning_effort == "low":
            think_tokens = 1024
        elif reasoning_effort == "medium":
            think_tokens = 1024 * 2
        elif reasoning_effort == "high":
            think_tokens = 1024 * 4
        else:
            think_tokens = 0
        if think_tokens > 0:
            if not max_tokens:
                max_tokens = 1024
            max_tokens += think_tokens
            thinking = {"type": "enabled", "budget_tokens": think_tokens}
            temperature = None
        else:
            thinking = {"type": "disabled"}
            if temperature is None:
                raise ValueError(f"Temperature is required for Claude models if reasoning_effort is not set")
        model = ChatAnthropic(model=model_name, temperature=temperature, thinking=thinking, max_tokens=max_tokens)
    elif model_name.startswith("gpt-4") or re.search(r"^o[0-9]-", model_name):
        # OpenAI models (including o-models)
        model = FlexTierChatOpenAI(
            model_name=model_name,
            temperature=temperature if not re.search(r"^o[0-9]-", model_name) else None,
            reasoning_effort=reasoning_effort if re.search(r"^o[0-9]-", model_name) else None,
            max_tokens=max_tokens,
            service_tier=service_tier,
            timeout=timeout if service_tier == "flex" else None,
        )
    else:
        raise ValueError(f"Model {model_name} not supported")

    return model

def create_step_summary_chain(model: Optional[str]=None, max_tokens: int=45):
    """
    Create a chain of tools to summarize each step in a workflow.
    Args:
        model: The OpenAI model to use for the language model.
        max_tokens: The maximum number of tokens to use for the summary.
    Returns:
        A chain of tools to summarize each step in a workflow.
    """
    # Create the prompt template
    template = "\n".join([
        "Concisely summarize the provided step in the langgraph workflow.",
        f"The summary must be {max_tokens} tokens or less.",
        "Do not use introductory words such as \"The workflow step involves\"",
        "Write your output as plain text instead of markdown.",
        "#-- The workflow step --#",
        "{step}"
    ])
    prompt = PromptTemplate(input_variables=["step"], template=template)

    # Initialize the language model
    model = set_model(agent_name="step_summary", max_tokens=max_tokens)

    # Return the LLM chain
    return prompt | model


async def create_agent_stream(
    input,  
    create_agent_func,
    config: dict={}, 
    summarize_steps: bool=False
) -> str:
    """
    Create an Entrez agent and stream the steps.
    Args:
        input: Input message to the agent.
        create_agent_func: Function to create the agent.
        config: Configuration for the agent.
        summarize_steps: Whether to summarize the steps.
    Returns:
        The final step message.
    """
    # create entrez agent
    agent = create_agent_func(return_tool=False)

    # create step summary chain
    step_summary_chain = create_step_summary_chain() if summarize_steps else None
    
    # invoke agent
    step_cnt = 0
    final_step = ""
    async for step in agent.astream(input, stream_mode="values", config=config):
        step_cnt += 1
        final_step = step
        # summarize step
        if step_summary_chain:
            msg = step_summary_chain.invoke({"step": step.get("messages")})
            print(f"Step {step_cnt}: {msg.content}", file=sys.stderr)
        else:
            try:
                if "messages" in step and step["messages"]:
                    last_msg = step["messages"][-1].content
                    if last_msg != "":
                        print(f"Step {step_cnt}: {last_msg}", file=sys.stderr)
                    else:
                        step_cnt -= 1
            except (KeyError, IndexError, AttributeError):
                print(f"Step {step_cnt}: {step}", file=sys.stderr)
    # get final step, and handle different types
    try:
        final_step = final_step["agent"]["messages"][-1].content
    except KeyError:
        try:
            final_step = final_step["messages"][-1].content
        except (KeyError, IndexError, AttributeError):
            if isinstance(final_step, str):
                return final_step
            return str(final_step)
    except TypeError:
        return str(final_step)
    return final_step

# main
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(override=True)

    # load settings
    settings = load_settings()
    print(settings)

    # set model
    model = set_model(model_name="claude-3-7-sonnet-20250219", agent_name="default")
    print(model)
