import pandas as pd
from pathlib import Path
from typing import Dict
import yaml

base_dir = Path(__file__).resolve().parent.parent.parent
data_dir = base_dir / "data"

train_path = data_dir / "train.csv"
test_path = data_dir / "test.csv"

config_path = base_dir / "src" / "config"
preprocessing_path = config_path / "preprocessing_config.yaml"
model_config_path = config_path / "model_config.yaml"
best_hyperparams_path = config_path / "best_hyperparams.yaml"

target = "emission"

def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.replace(" ", "_")
    return df

def load_yaml(path: Path) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)