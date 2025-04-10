import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project Root Folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"

def load_yaml(file_name: str):
    """Load YAML configuration from the config directory."""
    config_path = CONFIG_DIR / file_name
    if not config_path.exists():
        raise FileNotFoundError(f"Config file {file_name} not found in {CONFIG_DIR}")

    with open(config_path, "r") as file:
        return yaml.safe_load(file)



# Example usage
if __name__ == "__main__":
    chatbot_config = load_yaml("chatbot_config.yaml")
    print(chatbot_config)