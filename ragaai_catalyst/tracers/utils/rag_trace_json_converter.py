import json
from litellm import model_cost
import logging
import os
from datetime import datetime

logger = logging.getLogger("RagaAICatalyst")
logging_level = (
    logger.setLevel(logging.DEBUG) if os.getenv("DEBUG") == "1" else logging.INFO
)

def rag_trace_json_converter(input_trace, custom_model_cost, trace_id, user_details, tracer_type):
    trace_aggregate = {}
    
    def get_prompt(input_trace):
        if tracer_type == "rag/langchain":
            for span in input_trace:
                if span["name"] == "ChatOpenAI":
                    return span["attributes"]["llm.input_messages.1.message.content"]

                elif span["name"] == "LLMChain":
                    return json.loads(span["attributes"]["input.value"])["question"]

                elif span["name"] == "RetrievalQA":
                    return span["attributes"]["input.value"]
                
                elif span["name"] == "VectorStoreRetriever":
                    return span["attributes"]["input.value"]
                
                elif span["name"] == "ChatAnthropic":
                    return span["attributes"]["llm.input_messages.1.message.content"]
        return None
    
    def get_response(input_trace):
        if tracer_type == "rag/langchain":
            for span in input_trace:
                if span["name"] == "ChatOpenAI":
                    return span["attributes"]["llm.output_messages.0.message.content"]

                elif span["name"] == "LLMChain":
                    return json.loads(span["attributes"]["output.value"])

                elif span["name"] == "RetrievalQA":
                    return span["attributes"]["output.value"]

                elif span["name"] == "ChatAnthropic":
                    return span["attributes"]["llm.output_messages.0.message.content"]
        return None
    
    def get_context(input_trace):
        if tracer_type == "rag/langchain":
            for span in input_trace:
                if span["name"] == "VectorStoreRetriever":
                    return span["attributes"]["retrieval.documents.1.document.content"]
        return None
        
    prompt = get_prompt(input_trace)
    response = get_response(input_trace)
    context = get_context(input_trace)
    
    if tracer_type == "rag/langchain":
        trace_aggregate["tracer_type"] = "langchain"
    else:
        trace_aggregate["tracer_type"] = "llamaindex"

    trace_aggregate['trace_id'] = trace_id
    trace_aggregate['session_id'] = None
    trace_aggregate["metadata"] = user_details["trace_user_detail"]["metadata"]

    #dummy data need to be fetched
    trace_aggregate["pipeline"] = {
        'llm_model': 'gpt-4o-mini', 
        'vector_store': 'faiss',
        'embed_model': 'text-embedding-ada-002'
        }
    
    trace_aggregate["data"] = {}
    trace_aggregate["data"]["prompt"] = prompt
    trace_aggregate["data"]["response"] = response
    trace_aggregate["data"]["context"] = context
    
    if tracer_type == "rag/langchain":
        additional_metadata = get_additional_metadata(input_trace, custom_model_cost, model_cost)
    else:
        additional_metadata = get_additional_metadata(input_trace, custom_model_cost, model_cost)
    
    trace_aggregate["metadata"].update(additional_metadata)
    return trace_aggregate, additional_metadata

def get_additional_metadata(spans, custom_model_cost, model_cost_dict):
    additional_metadata = {}
    additional_metadata["cost"] = 0.0
    additional_metadata["tokens"] = {}
    try:
        for span in spans:
            if span["name"] in ["ChatOpenAI", "ChatAnthropic"]:
                start_time = datetime.fromisoformat(span["start_time"][:-1])  # Remove 'Z' and parse
                end_time = datetime.fromisoformat(span["end_time"][:-1])    # Remove 'Z' and parse
                additional_metadata["latency"] = (end_time - start_time).total_seconds()
                additional_metadata["model_name"] = span["attributes"]["llm.model_name"]
                additional_metadata["model"] = additional_metadata["model_name"]
                additional_metadata["tokens"]["prompt"] = span["attributes"]["llm.token_count.prompt"]
                additional_metadata["tokens"]["completion"] = span["attributes"]["llm.token_count.completion"]
                additional_metadata["tokens"]["total"] = additional_metadata["tokens"]["prompt"] + additional_metadata["tokens"]["completion"]

    except Exception as e:
        logger.error(f"Error getting additional metadata: {str(e)}")
    
    try:
        import pdb; pdb.set_trace()
        if custom_model_cost.get(additional_metadata['model_name']):
            model_cost_data = custom_model_cost[additional_metadata['model_name']]
        else:
            model_cost_data = model_cost_dict[additional_metadata['model_name']]
        if 'tokens' in additional_metadata and all(k in additional_metadata['tokens'] for k in ['prompt', 'completion']):
            prompt_cost = additional_metadata["tokens"]["prompt"]*model_cost_data["input_cost_per_token"]
            completion_cost = additional_metadata["tokens"]["completion"]*model_cost_data["output_cost_per_token"]
            additional_metadata["cost"] = prompt_cost + completion_cost 
            additional_metadata["total_cost"] = additional_metadata["cost"]
            additional_metadata["total_latency"] = additional_metadata["latency"]
            additional_metadata["prompt_tokens"] = additional_metadata["tokens"]["prompt"]
            additional_metadata["completion_tokens"] = additional_metadata["tokens"]["completion"]
    except Exception as e:
        logger.error(f"Error getting model cost data: {str(e)}")
    
    try:
        additional_metadata.pop("tokens", None)
    except Exception as e:
        logger.error(f"Error removing tokens from additional metadata: {str(e)}")

    return additional_metadata