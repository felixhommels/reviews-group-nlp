#!/usr/bin/env python3
"""Test runner for the NLP Analysis module."""

import os
import sys
import argparse
import pytest
from rich.console import Console
from rich.panel import Panel

# Add src directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

console = Console()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run NLP Analysis tests')
    parser.add_argument(
        '--unit',
        action='store_true',
        help='Run unit tests only'
    )
    parser.add_argument(
        '--integration',
        action='store_true',
        help='Run integration tests only'
    )
    parser.add_argument(
        '--coverage',
        action='store_true',
        help='Generate coverage report'
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose output'
    )
    return parser.parse_args()

def run_tests(args):
    """Run the test suite based on command line arguments."""
    # Build pytest arguments
    pytest_args = []
    
    if args.unit:
        pytest_args.append('tests/unit')
    elif args.integration:
        pytest_args.append('tests/integration')
    else:
        pytest_args.append('tests')
    
    if args.verbose:
        pytest_args.append('-v')
    
    if args.coverage:
        pytest_args.extend([
            '--cov=src',
            '--cov-report=term-missing',
            '--cov-report=html'
        ])
    
    # Run tests
    console.print(Panel(
        "Starting test suite...",
        title="Test Runner",
        style="bold blue"
    ))
    
    exit_code = pytest.main(pytest_args)
    
    if exit_code == 0:
        console.print(Panel(
            "All tests passed!",
            title="Success",
            style="bold green"
        ))
    else:
        console.print(Panel(
            f"Tests failed with exit code {exit_code}",
            title="Failure",
            style="bold red"
        ))
    
    return exit_code

if __name__ == '__main__':
    args = parse_args()
    sys.exit(run_tests(args))
