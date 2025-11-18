"""
Configuration utilities.

In a later step, you can extend this to read from:
- YAML files in configs/
- Environment variables for deployment
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = BASE_DIR / "configs"
