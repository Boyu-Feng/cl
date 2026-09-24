#!/usr/bin/env python3
"""
Centralized entrypoint for running continual learning benchmarks.

This script provides a unified CLI for running any system-task combination
with configurable parameters.
"""

import sys
from src.cli import main

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
