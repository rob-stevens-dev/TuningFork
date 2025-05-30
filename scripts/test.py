#!/usr/bin/env python3
"""Test runner script for TuningFork.

This script provides a convenient interface for running different types of tests
with appropriate configuration and reporting.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def run_command(cmd: List[str], *, cwd: Optional[Path] = None) -> int:
    """Run command and return exit code.
    
    Args:
        cmd: Command to run as list of strings
        cwd: Working directory for command
        
    Returns:
        Exit code from command
    """
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    return result.returncode


def run_tests(
    test_type: str = "all",
    *,
    coverage: bool = False,
    verbose: bool = False,
    parallel: bool = False,
    benchmark: bool = False,
    fail_fast: bool = False,
    html_report: bool = False,
) -> int:
    """Run tests with specified configuration.
    
    Args:
        test_type: Type of tests to run (all, unit, integration, performance, system)
        coverage: Enable coverage reporting
        verbose: Enable verbose output
        parallel: Run tests in parallel
        benchmark: Run benchmark tests only
        fail_fast: Stop on first failure
        html_report: Generate HTML coverage report
        
    Returns:
        Exit code from pytest
    """
    cmd = ["python", "-m", "pytest"]
    
    # Add test type filter
    if test_type != "all":
        cmd.extend(["-m", test_type])
    
    # Add coverage options
    if coverage:
        cmd.extend([
            "--cov=src/tuningfork",
            "--cov-report=term-missing:skip-covered",
            "--cov-report=xml:coverage.xml",
            "--cov-fail-under=95",
        ])
        
        if html_report:
            cmd.append("--cov-report=html:htmlcov")
    
    # Add benchmark options
    if benchmark:
        cmd.append("--benchmark-only")
    
    # Add parallel execution
    if parallel:
        cmd.extend(["-n", "auto"])
    
    # Add verbose output
    if verbose:
        cmd.append("-v")
    
    # Add fail fast
    if fail_fast:
        cmd.append("-x")
    
    # Add timing information
    cmd.append("--durations=10")
    
    return run_command(cmd)


def run_quality_checks() -> int:
    """Run code quality checks.
    
    Returns:
        Exit code (0 if all checks pass)
    """
    checks = [
        # Format checking
        (["black", "--check", "src", "tests"], "Code formatting (black)"),
        (["isort", "--check-only", "src", "tests"], "Import sorting (isort)"),
        
        # Linting
        (["flake8", "src", "tests"], "Code linting (flake8)"),
        
        # Type checking
        (["mypy", "src"], "Type checking (mypy)"),
    ]
    
    failed_checks = []
    
    for cmd, description in checks:
        print(f"\n{'='*60}")
        print(f"Running {description}")
        print(f"{'='*60}")
        
        exit_code = run_command(cmd)
        if exit_code != 0:
            failed_checks.append(description)
    
    if failed_checks:
        print(f"\n{'='*60}")
        print("❌ Quality checks failed:")
        for check in failed_checks:
            print(f"  - {check}")
        print(f"{'='*60}")
        return 1
    else:
        print(f"\n{'='*60}")
        print("✅ All quality checks passed!")
        print(f"{'='*60}")
        return 0


def run_format() -> int:
    """Run code formatting tools.
    
    Returns:
        Exit code (0 if successful)
    """
    format_commands = [
        (["black", "src", "tests"], "Code formatting (black)"),
        (["isort", "src", "tests"], "Import sorting (isort)"),
    ]
    
    for cmd, description in format_commands:
        print(f"\nRunning {description}")
        exit_code = run_command(cmd)
        if exit_code != 0:
            print(f"❌ {description} failed")
            return exit_code
    
    print("✅ Code formatting completed!")
    return 0


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(
        description="TuningFork test runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Run all tests
  %(prog)s --type unit              # Run unit tests only
  %(prog)s --coverage --html        # Run with coverage and HTML report
  %(prog)s --quality                # Run quality checks only
  %(prog)s --format                 # Format code
  %(prog)s --benchmark              # Run benchmark tests only
  %(prog)s --type performance -v    # Run performance tests with verbose output
        """
    )
    
    # Test execution options
    parser.add_argument(
        "--type", "-t",
        choices=["all", "unit", "integration", "performance", "system"],
        default="all",
        help="Type of tests to run (default: all)"
    )
    
    parser.add_argument(
        "--coverage", "-c",
        action="store_true",
        help="Enable coverage reporting"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--parallel", "-p",
        action="store_true",
        help="Run tests in parallel"
    )
    
    parser.add_argument(
        "--benchmark", "-b",
        action="store_true",
        help="Run benchmark tests only"
    )
    
    parser.add_argument(
        "--fail-fast", "-x",
        action="store_true",
        help="Stop on first failure"
    )
    
    parser.add_argument(
        "--html",
        action="store_true",
        help="Generate HTML coverage report"
    )
    
    # Quality and formatting options
    parser.add_argument(
        "--quality", "-q",
        action="store_true",
        help="Run code quality checks (format, lint, type check)"
    )
    
    parser.add_argument(
        "--format", "-f",
        action="store_true",
        help="Format code with black and isort"
    )
    
    # Full suite option
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full test suite (format, quality, tests with coverage)"
    )
    
    args = parser.parse_args()
    
    # Change to project root directory
    project_root = Path(__file__).parent.parent
    
    # Handle different execution modes
    if args.format:
        return run_format()
    
    elif args.quality:
        return run_quality_checks()
    
    elif args.full:
        # Run full test suite
        print("🚀 Running full test suite...")
        
        # Format code
        print("\n1️⃣ Formatting code...")
        exit_code = run_format()
        if exit_code != 0:
            return exit_code
        
        # Quality checks
        print("\n2️⃣ Running quality checks...")
        exit_code = run_quality_checks()
        if exit_code != 0:
            return exit_code
        
        # Tests with coverage
        print("\n3️⃣ Running tests with coverage...")
        exit_code = run_tests(
            test_type="all",
            coverage=True,
            html_report=True,
            verbose=args.verbose,
            parallel=args.parallel,
        )
        
        if exit_code == 0:
            print("\n🎉 Full test suite completed successfully!")
        else:
            print("\n❌ Test suite failed!")
        
        return exit_code
    
    else:
        # Run tests
        return run_tests(
            test_type=args.type,
            coverage=args.coverage,
            verbose=args.verbose,
            parallel=args.parallel,
            benchmark=args.benchmark,
            fail_fast=args.fail_fast,
            html_report=args.html,
        )


if __name__ == "__main__":
    sys.exit(main())