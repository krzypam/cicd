import os
from src.train import train_model

def test_model_training():
    train_model()
    assert os.path.exists('models/model.pkl'), "Model was not saved correctly"