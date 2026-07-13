import pytest
from src.data_prep import load_config, load_all_data

def test_load_config():
    config = load_config("configs/baseline.yaml")
    assert "dataset" in config

def test_load_all_data():
    config = load_config("configs/baseline.yaml")
    data = load_all_data(config, nrows=10)
    assert isinstance(data, dict)
    assert "sms" in data
    assert len(data["sms"]) <= 10
