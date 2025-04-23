import json
from litellm import model_cost
import logging
import os
import re
from datetime import datetime
import tiktoken

logger = logging.getLogger("RagaAICatalyst")
logging_level = (
    logger.setLevel(logging.DEBUG) if os.getenv("DEBUG") == "1" else logging.INFO
)

def rag_trace_json_converter(input_trace, custom_model_cost, trace_id, user_details, tracer_type,user_context):
    trace_aggregate = {}
    def get_prompt(input_trace):
        try:
            if tracer_type == "langchain":
                for span in input_trace:
                    try:
                        # First check if there's a user message in any of the input messages
                        attributes = span.get("attributes", {})
                        
                        # Look for user role in any of the input messages
                        if attributes:
                            for key, value in attributes.items():
                                try:
                                    if key.startswith("llm.input_messages.") and key.endswith(".message.role") and value == "user":
                                        # Extract the message number
                                        message_num = key.split(".")[2]
                                        # Construct the content key
                                        content_key = f"llm.input_messages.{message_num}.message.content"
                                        if content_key in attributes:
                                            return attributes.get(content_key)
                                except Exception as e:
                                    logger.warning(f"Error processing attribute key-value pair: {str(e)}")
                                    continue

                            for key, value in attributes.items():
                                try:
                                    if key.startswith("llm.prompts") and isinstance(value, list):
                                        human_message = None
                                        for message in value:
                                            if isinstance(message, str):
                                                human_index = message.find("Human:")
                                                if human_index != -1:
                                                    human_message = message[human_index:].replace("Human:", "")
                                                    break
                                        return human_message if human_message else value
                                except Exception as e:
                                    logger.warning(f"Error processing attribute key-value pair for prompt: {str(e)}")
                                    continue
                    except Exception as e:
                        logger.warning(f"Error processing span for prompt extraction: {str(e)}")
                        continue
                
                for span in input_trace:
                    try:
                        # If no user message found, check for specific span types
                        if span["name"] == "LLMChain":
                            try:
                                input_value = span["attributes"].get("input.value", "{}")
                                return json.loads(input_value).get("question", "")
                            except json.JSONDecodeError:
                                logger.warning(f"Invalid JSON in LLMChain input.value: {input_value}")
                                continue
                        elif span["name"] == "RetrievalQA":
                            return span["attributes"].get("input.value", "")
                        elif span["name"] == "VectorStoreRetriever":
                            return span["attributes"].get("input.value", "")
                    except Exception as e:
                        logger.warning(f"Error processing span for fallback prompt extraction: {str(e)}")
                        continue
                
                # If we've gone through all spans and found nothing
                logger.warning("No user message found in any span")
                logger.warning("Returning empty string for prompt.")
                return ""
            
            logger.error("Prompt not found in the trace")
            return None
        except Exception as e:
            logger.error(f"Error while extracting prompt from trace: {str(e)}")
            return None
    
    def get_response(input_trace):
        try:
            if tracer_type == "langchain":
                for span in input_trace:
                    try:
                        attributes = span.get("attributes", {})
                        if attributes:
                            for key, value in attributes.items():
                                try:
                                    if key.startswith("llm.output_messages.") and key.endswith(".message.content"):
                                        return value
                                except Exception as e:
                                    logger.warning(f"Error processing attribute key-value pair for response: {str(e)}")
                                    continue
                            
                            for key, value in attributes.items():
                                try:
                                    if key.startswith("output.value"):
                                        try:
                                            output_json = json.loads(value)
                                            if "generations" in output_json and isinstance(output_json.get("generations"), list) and len(output_json.get("generations")) > 0:
                                                if isinstance(output_json.get("generations")[0], list) and len(output_json.get("generations")[0]) > 0:
                                                    first_generation = output_json.get("generations")[0][0]
                                                    if "text" in first_generation:
                                                        return first_generation["text"]
                                        except json.JSONDecodeError:
                                            logger.warning(f"Invalid JSON in output.value: {value}")
                                            continue
                                except Exception as e:
                                    logger.warning(f"Error processing attribute key-value pair for response: {str(e)}")
                                    continue
                    except Exception as e:
                        logger.warning(f"Error processing span for response extraction: {str(e)}")
                        continue
                
                for span in input_trace:
                    try:
                        if span["name"] == "LLMChain":
                            try:
                                output_value = span["attributes"].get("output.value", "")
                                if output_value:
                                    return json.loads(output_value)
                                return ""
                            except json.JSONDecodeError:
                                logger.warning(f"Invalid JSON in LLMChain output.value: {output_value}")
                                continue
                        elif span["name"] == "RetrievalQA":
                            return span["attributes"].get("output.value", "")
                        elif span["name"] == "VectorStoreRetriever":
                            return span["attributes"].get("output.value", "")
                    except Exception as e:
                        logger.warning(f"Error processing span for fallback response extraction: {str(e)}")
                        continue
                
                logger.warning("No response found in any span")
                return ""
            
            logger.error("Response not found in the trace")
            return None
        except Exception as e:
            logger.error(f"Error while extracting response from trace: {str(e)}")
            return None
    
    def get_context(input_trace):
        try:
            if user_context and user_context.strip():
                return user_context
            elif tracer_type == "langchain":
                for span in input_trace:
                    try:
                        if span["name"] == "VectorStoreRetriever":
                            return span["attributes"].get("retrieval.documents.1.document.content", "")
                    except Exception as e:
                        logger.warning(f"Error processing span for context extraction: {str(e)}")
                        continue
            
            logger.warning("Context not found in the trace")
            return ""
        except Exception as e:
            logger.error(f"Error while extracting context from trace: {str(e)}")
            return ""
        
    prompt = get_prompt(input_trace)
    response = get_response(input_trace)
    context = get_context(input_trace)
    
    if tracer_type == "langchain":
        trace_aggregate["tracer_type"] = "langchain"
    else:
        trace_aggregate["tracer_type"] = "llamaindex"

    trace_aggregate['trace_id'] = trace_id
    trace_aggregate['session_id'] = None
    trace_aggregate["metadata"] = user_details.get("trace_user_detail", {}).get("metadata")
    trace_aggregate["pipeline"] = user_details.get("trace_user_detail", {}).get("pipeline")

    trace_aggregate["data"] = {}
    trace_aggregate["data"]["prompt"] = prompt
    trace_aggregate["data"]["response"] = response
    trace_aggregate["data"]["context"] = context
    
    if tracer_type == "langchain":
        additional_metadata = get_additional_metadata(input_trace, custom_model_cost, model_cost, prompt, response)
    else:
        additional_metadata = get_additional_metadata(input_trace, custom_model_cost, model_cost)
    
    trace_aggregate["metadata"] = user_details.get("trace_user_detail", {}).get("metadata")
    trace_aggregate["metadata"].update(additional_metadata)
    additional_metadata.pop("total_cost")
    additional_metadata.pop("total_latency")
    return trace_aggregate, additional_metadata

def get_additional_metadata(spans, custom_model_cost, model_cost_dict, prompt="", response=""):
    additional_metadata = {}
    additional_metadata["cost"] = 0.0
    additional_metadata["tokens"] = {}
    try:
        for span in spans:
            if span["name"] in ["ChatOpenAI", "ChatAnthropic", "ChatGoogleGenerativeAI", "OpenAI", "ChatOpenAI_LangchainOpenAI", "ChatOpenAI_ChatModels",
                                "ChatVertexAI", "VertexAI", "ChatLiteLLM", "ChatBedrock", "AzureChatOpenAI", "ChatAnthropicVertex"]:
                start_time = datetime.fromisoformat(span.get("start_time", "")[:-1])  # Remove 'Z' and parse
                end_time = datetime.fromisoformat(span.get("end_time", "")[:-1])    # Remove 'Z' and parse
                additional_metadata["latency"] = (end_time - start_time).total_seconds()
                additional_metadata["model_name"] = span["attributes"].get("llm.model_name", "").replace("models/", "")
                additional_metadata["model"] = additional_metadata["model_name"]
                try:
                    additional_metadata["tokens"]["prompt"] = span["attributes"]["llm.token_count.prompt"]

                except:
                    logger.warning("Warning: prompt token not found. using fallback strategies to get tokens.")
                    try:
                        additional_metadata["tokens"]["prompt"] = num_tokens_from_messages(
                            model=additional_metadata["model_name"],
                            message=prompt
                        )
                    except Exception as e:
                        logger.warning(f"Failed to count prompt tokens: {str(e)}. Using 'gpt-4o-mini' model count as fallback.")
                        additional_metadata["tokens"]["prompt"] = num_tokens_from_messages(
                            model="gpt-4o-mini",
                            message=prompt
                        )
                
                try:
                    additional_metadata["tokens"]["completion"] = span["attributes"]["llm.token_count.completion"]
                except:
                    logger.warning("Warning: completion token not found. using fallback strategies to get tokens.")
                    try:
                        additional_metadata["tokens"]["completion"] = num_tokens_from_messages(
                            model=additional_metadata["model_name"],
                            message=response
                        )
                    except Exception as e:
                        logger.warning(f"Failed to count completion tokens: {str(e)}. Using 'gpt-4o-mini' model count as fallback.")
                        additional_metadata["tokens"]["completion"] = num_tokens_from_messages(
                            model="gpt-4o-mini",
                            message=response
                        )
                
                # Ensure both values are not None before adding
                prompt_tokens = additional_metadata["tokens"].get("prompt", 0) or 0
                completion_tokens = additional_metadata["tokens"].get("completion", 0) or 0
                additional_metadata["tokens"]["total"] = prompt_tokens + completion_tokens

    except Exception as e:
        logger.error(f"Error getting additional metadata: {str(e)}")
    
    try:
        if custom_model_cost.get(additional_metadata.get('model_name')):
            model_cost_data = custom_model_cost[additional_metadata.get('model_name')]
        else:
            model_cost_data = model_cost_dict.get(additional_metadata.get('model_name'))
        
        # Check if model_cost_data is None
        if model_cost_data is None:
            logger.warning(f"No cost data found for model: {additional_metadata.get('model_name')}")
            # Set default values
            additional_metadata["cost"] = 0.0
            additional_metadata["total_cost"] = 0.0
            additional_metadata["total_latency"] = additional_metadata.get("latency", 0)
            additional_metadata["prompt_tokens"] = additional_metadata["tokens"].get("prompt", 0) or 0
            additional_metadata["completion_tokens"] = additional_metadata["tokens"].get("completion", 0) or 0
        elif 'tokens' in additional_metadata and all(k in additional_metadata['tokens'] for k in ['prompt', 'completion']):
            # Get input and output costs, defaulting to 0 if not found
            input_cost_per_token = model_cost_data.get("input_cost_per_token", 0) or 0
            output_cost_per_token = model_cost_data.get("output_cost_per_token", 0) or 0
            
            # Get token counts, defaulting to 0 if not found
            prompt_tokens = additional_metadata["tokens"].get("prompt", 0) or 0
            completion_tokens = additional_metadata["tokens"].get("completion", 0) or 0
            
            # Calculate costs
            prompt_cost = prompt_tokens * input_cost_per_token
            completion_cost = completion_tokens * output_cost_per_token
            
            additional_metadata["cost"] = prompt_cost + completion_cost 
            additional_metadata["total_cost"] = additional_metadata["cost"]
            additional_metadata["total_latency"] = additional_metadata.get("latency", 0)
            additional_metadata["prompt_tokens"] = prompt_tokens
            additional_metadata["completion_tokens"] = completion_tokens
    except Exception as e:
        logger.warning(f"Error getting model cost data: {str(e)}")
        # Set default values in case of error
        additional_metadata["cost"] = 0.0
        additional_metadata["total_cost"] = 0.0
        additional_metadata["total_latency"] = additional_metadata.get("latency", 0)
        additional_metadata["prompt_tokens"] = additional_metadata["tokens"].get("prompt", 0) or 0
        additional_metadata["completion_tokens"] = additional_metadata["tokens"].get("completion", 0) or 0
    try:
        additional_metadata.pop("tokens", None)
    except Exception as e:
        logger.error(f"Error removing tokens from additional metadata: {str(e)}")

    return additional_metadata

def num_tokens_from_messages(model, message):
    try:
        # Handle None or empty message
        if not message:
            logger.warning("Empty or None message provided to token counter")
            return 0
            
        # GPT models
        if re.match(r'^gpt-', model):
            """Check if the model is any GPT model (pattern: ^gpt-)
            This matches any model name that starts with 'gpt-'
            """
            def num_tokens_from_string(string: str, encoding_name: str) -> int:
                """Returns the number of tokens in a text string."""
                try:
                    encoding = tiktoken.get_encoding(encoding_name)
                    num_tokens = len(encoding.encode(string))
                    return num_tokens
                except Exception as e:
                    logger.warning(f"Error encoding with {encoding_name}: {str(e)}")
                    # Fallback to a different encoding if the requested one fails
                    try:
                        fallback_encoding = tiktoken.get_encoding("cl100k_base")
                        return len(fallback_encoding.encode(string))
                    except:
                        logger.error("Failed to use fallback encoding")
                        return 0
            
            if re.match(r'^gpt-4o.*', model):
                """Check for GPT-4 Optimized models (pattern: ^gpt-4o.*)
                Examples that match:
                - gpt-4o
                - gpt-4o-mini
                - gpt-4o-2024-08-06
                The .* allows for any characters after 'gpt-4o'
                """
                encoding_name = "o200k_base"
                return num_tokens_from_string(message, encoding_name)
            
            elif re.match(r'^gpt-(4|3\.5).*', model):
                """Check for GPT-4 and GPT-3.5 models (pattern: ^gpt-(4|3\.5).*)
                Uses cl100k_base encoding for GPT-4 and GPT-3.5 models
                Examples that match:
                - gpt-4
                - gpt-4-turbo
                - gpt-4-2024-08-06
                - gpt-3.5-turbo
                - gpt-3.5-turbo-16k
                """
                encoding_name = "cl100k_base"
                return num_tokens_from_string(message, encoding_name)
            
            else:
                """Default case for any other GPT models
                Uses o200k_base encoding as the default tokenizer
                """
                return num_tokens_from_string(message, encoding_name="o200k_base")
            

        # Gemini models 
        elif re.match(r'^gemini-', model):
            try:
                GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
                if not GOOGLE_API_KEY:
                    logger.warning("GOOGLE_API_KEY not found in environment variables")
                    return 0
                    
                import google.generativeai as genai
                client = genai.Client(api_key=GOOGLE_API_KEY)

                response = client.models.count_tokens(
                        model=model,
                        contents=message,
                    )
                return response.total_tokens
            except ImportError:
                logger.warning("google.generativeai module not found. Install with pip install google-generativeai")
                return 0
            except Exception as e:
                logger.warning(f"Error counting tokens for Gemini model: {str(e)}")
                return 0
        
        # Default case for unknown models
        else:
            logger.warning(f"Unknown model type: {model}. Using default token counter.")
            try:
                # Use cl100k_base as a fallback for unknown models
                encoding = tiktoken.get_encoding("cl100k_base")
                return len(encoding.encode(message))
            except:
                logger.error("Failed to use fallback encoding for unknown model")
                return 0
    except Exception as e:
        logger.error(f"Unexpected error in token counting: {str(e)}")
        return 0