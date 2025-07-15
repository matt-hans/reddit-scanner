#!/usr/bin/env python3
"""
Simple test runner for Reddit Scanner tests.
This script provides easy commands to run different types of tests.
"""

import subprocess
import sys
import os


def run_command(cmd):
    """Run a command and return the result"""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.stdout:
        print("STDOUT:")
        print(result.stdout)
    
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
    
    return result.returncode == 0


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_tests.py [command]")
        print("Commands:")
        print("  all          - Run all tests")
        print("  basic        - Run basic tests only")
        print("  auth         - Run authentication tests only")
        print("  nlp          - Run NLP-related tests only")
        print("  tools        - Run MCP tool tests only")
        print("  coverage     - Run tests with coverage report")
        print("  install      - Install test dependencies")
        return

    command = sys.argv[1]
    
    if command == "install":
        print("Installing test dependencies...")
        success = run_command([sys.executable, "-m", "pip", "install", "-e", ".[test]"])
        if success:
            print("✅ Test dependencies installed successfully!")
        else:
            print("❌ Failed to install test dependencies")
        return
    
    # Check if pytest is available
    try:
        import pytest
    except ImportError:
        print("❌ pytest not found. Run: python run_tests.py install")
        return
    
    base_cmd = [sys.executable, "-m", "pytest", "test_reddit_scanner.py", "-v"]
    
    if command == "all":
        success = run_command(base_cmd)
    elif command == "basic":
        success = run_command(base_cmd + ["-k", "not (nlp or tool)"])
    elif command == "auth":
        success = run_command(base_cmd + ["-k", "Authentication"])
    elif command == "nlp":
        success = run_command(base_cmd + ["-k", "nlp or NLP"])
    elif command == "tools":
        success = run_command(base_cmd + ["-k", "MCP"])
    elif command == "coverage":
        success = run_command(base_cmd + ["--cov=reddit_scanner", "--cov-report=html", "--cov-report=term"])
    else:
        print(f"Unknown command: {command}")
        return
    
    if success:
        print("✅ Tests completed successfully!")
    else:
        print("❌ Some tests failed")


if __name__ == "__main__":
    main()