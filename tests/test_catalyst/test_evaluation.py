from unittest.mock import patch
import time
import pytest
import os
import dotenv
dotenv.load_dotenv()
import pandas as pd
from datetime import datetime 
from typing import Dict, List
from ragaai_catalyst import Evaluation, RagaAICatalyst

# Simplified model configurations
MODEL_CONFIGS = [
    {"provider": "openai", "model": "gpt-4"},  # Only one OpenAI model
    {"provider": "gemini", "model": "gemini-1.5-flash"}  # Only one Gemini model
]

# Common metrics to test
CORE_METRICS = [
    'Hallucination',
    'Faithfulness',
    'Response Correctness',
    'Context Relevancy'
]

CHAT_METRICS = [
    'Agent Quality',
    'User Chat Quality'
]

@pytest.fixture
def base_url():
    return os.getenv("RAGAAI_CATALYST_BASE_URL")

@pytest.fixture
def access_keys():
    return {
        "access_key": os.getenv("RAGAAI_CATALYST_ACCESS_KEY"),
        "secret_key": os.getenv("RAGAAI_CATALYST_SECRET_KEY")
    }

@pytest.fixture
def evaluation(base_url, access_keys):
    """Create evaluation instance with specific project and dataset"""
    os.environ["RAGAAI_CATALYST_BASE_URL"] = base_url
    catalyst = RagaAICatalyst(
        access_key=access_keys["access_key"],
        secret_key=access_keys["secret_key"]
    )
    return Evaluation(
        project_name="prompt_metric_dataset_sk", 
        dataset_name="dataset_19feb_1"
    )

@pytest.fixture
def chat_evaluation(base_url, access_keys):
    """Create evaluation instance for chat metrics"""
    os.environ["RAGAAI_CATALYST_BASE_URL"] = base_url
    catalyst = RagaAICatalyst(
        access_key=access_keys["access_key"],
        secret_key=access_keys["secret_key"]
    )
    return Evaluation(
        project_name="prompt_metric_dataset_sk", 
        dataset_name="dataset_19feb_1"
    )

# Basic initialization tests
def test_evaluation_initialization(evaluation):
    """Test if evaluation is initialized correctly"""
    assert evaluation.project_name == "prompt_metric_dataset_sk"
    assert evaluation.dataset_name == "dataset_19feb_1"

def test_project_does_not_exist():
    """Test initialization with non-existent project"""
    with pytest.raises(ValueError, match="Project not found"):
        Evaluation(project_name="non_existent_project", dataset_name="dataset")

# Parameterized validation tests
@pytest.mark.parametrize("provider_config", MODEL_CONFIGS)
def test_metric_validation_checks(evaluation, provider_config):
    """Test all validation checks in one parameterized test"""
    schema_mapping = {
        'Query': 'Prompt',
        'Response': 'Response',
        'Context': 'Context',
    }
    
    # Test missing schema_mapping
    with pytest.raises(ValueError):
        evaluation.add_metrics([{
            "name": "Hallucination",
            "config": provider_config,
            "column_name": "test_column"
        }])
    
    # Test missing column_name
    with pytest.raises(ValueError):
        evaluation.add_metrics([{
            "name": "Hallucination",
            "config": provider_config,
            "schema_mapping": schema_mapping
        }])
    
    # Test missing metric name
    with pytest.raises(ValueError):
        evaluation.add_metrics([{
            "config": provider_config,
            "column_name": "test_column",
            "schema_mapping": schema_mapping
        }])

# Core metric evaluation test
@pytest.mark.parametrize("metric_name", CORE_METRICS)
@pytest.mark.parametrize("provider_config", MODEL_CONFIGS)
def test_core_metrics_evaluation(evaluation, metric_name, provider_config, capfd):
    """Test evaluation of core metrics with different providers"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics = [{
        "name": metric_name,
        "config": provider_config,
        "column_name": f"{metric_name}_column_{timestamp}",
        "schema_mapping": {
            'Query': 'prompt',
            'Response': 'response',
            'Context': 'context',
            'ExpectedResponse': 'expectedresponse'
        }
    }]
    
    evaluation.add_metrics(metrics=metrics)
    out, _ = capfd.readouterr()
    assert "Metric Evaluation Job scheduled successfully" in out
    assert evaluation.jobId is not None
    
    # Basic status check without long wait
    evaluation.get_status()
    out, _ = capfd.readouterr()
    assert "Job" in out  # Just checking we got some status

# Chat metric evaluation test
@pytest.mark.parametrize("metric_name", CHAT_METRICS)
@pytest.mark.parametrize("provider_config", MODEL_CONFIGS)
def test_chat_metrics_evaluation(chat_evaluation, metric_name, provider_config, capfd):
    """Test evaluation of chat metrics with different providers"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics = [{
        "name": metric_name,
        "config": provider_config,
        "column_name": f"{metric_name}_column_{timestamp}",
        "schema_mapping": {
            '_traceid': 'ChatID',
            'Query': 'Chat',
            'Context': 'Instructions'
        }
    }]
    
    chat_evaluation.add_metrics(metrics=metrics)
    out, _ = capfd.readouterr()
    assert "Metric Evaluation Job scheduled successfully" in out
    assert chat_evaluation.jobId is not None
    
    # Basic status check without long wait
    chat_evaluation.get_status()
    out, _ = capfd.readouterr()
    assert "Job" in out  # Just checking we got some status