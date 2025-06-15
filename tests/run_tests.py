#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test Runner

This script runs all the tests in the project.
"""

import os
import sys
import unittest
import argparse

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_tests(test_module=None, verbose=True):
    """
    Run the tests.
    
    Args:
        test_module (str): Name of specific test module to run (without .py extension)
        verbose (bool): Whether to use verbose output
    """
    # Set the verbosity level
    verbosity = 2 if verbose else 1
    
    # Create test suite
    loader = unittest.TestLoader()
    
    if test_module:
        # If a specific module is specified, run only those tests
        try:
            suite = loader.loadTestsFromName(f"tests.{test_module}")
        except ImportError:
            print(f"Test module not found: {test_module}")
            return False
    else:
        # Otherwise, discover and run all tests
        suite = loader.discover('tests')
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    # Return True if all tests passed
    return result.wasSuccessful()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the tests for Metro Booking Voice Assistant")
    parser.add_argument('-m', '--module', help='Specific test module to run (without .py extension)')
    parser.add_argument('-q', '--quiet', action='store_true', help='Quiet output')
    args = parser.parse_args()
    
    # Run the tests
    success = run_tests(args.module, not args.quiet)
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1)
