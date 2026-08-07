"""Convenience command for executing the complete training pipeline."""

import json

from main import main

if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
