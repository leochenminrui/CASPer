#!/usr/bin/env python3
"""
Verify PEM project setup.

This script checks that all required files and directories are in place
and that the Python environment is correctly configured.

Usage:
    python scripts/00_verify_setup.py
"""

import sys
from pathlib import Path
from typing import List, Tuple

# Colors for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


def check_item(name: str, condition: bool, details: str = "") -> bool:
    """Check and print status of a verification item."""
    if condition:
        print(f"{GREEN}✓{RESET} {name}")
        if details:
            print(f"  → {details}")
        return True
    else:
        print(f"{RED}✗{RESET} {name}")
        if details:
            print(f"  → {details}")
        return False


def verify_setup() -> Tuple[int, int]:
    """
    Verify the PEM project setup.

    Returns:
        (passed_checks, total_checks)
    """
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}PEM Project Setup Verification{RESET}")
    print(f"{BLUE}{'='*70}{RESET}\n")

    passed = 0
    total = 0

    # Project root
    project_root = Path(__file__).parent.parent
    print(f"Project root: {project_root}\n")

    # Check Python version
    print(f"{BLUE}[1/7] Python Environment{RESET}")
    total += 1
    if check_item(
        "Python version >= 3.8",
        sys.version_info >= (3, 8),
        f"Current: Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    ):
        passed += 1
    print()

    # Check directory structure
    print(f"{BLUE}[2/7] Directory Structure{RESET}")
    required_dirs = [
        "config",
        "data/raw",
        "data/census",
        "data/processed",
        "data/splits",
        "src/data",
        "src/utils",
        "scripts",
        "reports/01_census",
        "reports/02_processing",
        "reports/03_results",
        "tests"
    ]

    dir_checks_passed = 0
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        total += 1
        if check_item(f"{dir_path}/", full_path.is_dir()):
            passed += 1
            dir_checks_passed += 1

    print(f"  Directories: {dir_checks_passed}/{len(required_dirs)}\n")

    # Check configuration files
    print(f"{BLUE}[3/7] Configuration Files{RESET}")
    config_files = [
        "config/datasets.yaml",
        "config/parsing_rules.yaml"
    ]

    config_checks_passed = 0
    for config_file in config_files:
        full_path = project_root / config_file
        total += 1
        if check_item(config_file, full_path.is_file()):
            passed += 1
            config_checks_passed += 1

    print(f"  Config files: {config_checks_passed}/{len(config_files)}\n")

    # Check source code files
    print(f"{BLUE}[4/7] Source Code{RESET}")
    source_files = [
        "src/__init__.py",
        "src/data/__init__.py",
        "src/data/schema.py",
        "src/data/census.py",
        "src/utils/__init__.py",
        "src/utils/logging.py",
        "src/models/__init__.py",
        "src/evaluation/__init__.py"
    ]

    source_checks_passed = 0
    for source_file in source_files:
        full_path = project_root / source_file
        total += 1
        if check_item(source_file, full_path.is_file()):
            passed += 1
            source_checks_passed += 1

    print(f"  Source files: {source_checks_passed}/{len(source_files)}\n")

    # Check scripts
    print(f"{BLUE}[5/7] Executable Scripts{RESET}")
    scripts = [
        "scripts/01_preprocess_cycpeptmpdb.py",
        "scripts/run_census.py"
    ]

    script_checks_passed = 0
    for script in scripts:
        full_path = project_root / script
        total += 1
        if check_item(script, full_path.is_file()):
            passed += 1
            script_checks_passed += 1

    print(f"  Scripts: {script_checks_passed}/{len(scripts)}\n")

    # Check documentation
    print(f"{BLUE}[6/7] Documentation{RESET}")
    docs = [
        "README.md",
        "SUMMARY.md",
        "QUICKSTART.md",
        "ROADMAP.md",
        "DATA_ACQUISITION.md",
        "IMPLEMENTATION_STATUS.md",
        "PROJECT_LOG.md",
        "requirements.txt",
        ".gitignore"
    ]

    doc_checks_passed = 0
    for doc in docs:
        full_path = project_root / doc
        total += 1
        if check_item(doc, full_path.is_file()):
            passed += 1
            doc_checks_passed += 1

    print(f"  Documentation files: {doc_checks_passed}/{len(docs)}\n")

    # Check for datasets (optional at this stage)
    print(f"{BLUE}[7/7] Datasets (Optional){RESET}")
    datasets = [
        ("data/raw/cycpeptmpdb_pampa.csv", "CycPeptMPDB (PAMPA)"),
    ]

    datasets_found = 0
    for dataset_path, description in datasets:
        full_path = project_root / dataset_path
        if full_path.is_file():
            print(f"{GREEN}✓{RESET} {description}")
            print(f"  → Found: {dataset_path}")
            datasets_found += 1
        else:
            print(f"{YELLOW}○{RESET} {description}")
            print(f"  → Not found: {dataset_path} (download required)")

    if datasets_found > 0:
        print(f"\n  Datasets available: {datasets_found}/3")
    else:
        print(f"\n  {YELLOW}No datasets found. See DATA_ACQUISITION.md{RESET}")

    print()

    # Try importing key modules
    print(f"{BLUE}Module Import Test{RESET}")
    import_passed = 0
    import_total = 0

    modules_to_test = [
        ("src.utils.logging", "AuditLogger"),
        ("src.data.schema", "DataPoint, Peptide, Modification"),
        ("src.data.census", "DatasetCensus")
    ]

    for module_name, description in modules_to_test:
        import_total += 1
        try:
            __import__(module_name)
            print(f"{GREEN}✓{RESET} {module_name} ({description})")
            import_passed += 1
        except ImportError as e:
            print(f"{RED}✗{RESET} {module_name} ({description})")
            print(f"  → Error: {e}")

    print(f"  Module imports: {import_passed}/{import_total}\n")

    # Summary
    print(f"{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}Summary{RESET}")
    print(f"{BLUE}{'='*70}{RESET}\n")

    print(f"Checks passed: {passed}/{total}")

    if passed == total:
        print(f"\n{GREEN}✓ Setup complete! Ready to proceed.{RESET}\n")
        print(f"Next steps:")
        print(f"1. Acquire datasets (see DATA_ACQUISITION.md)")
        print(f"2. Run: python scripts/run_census.py")
        print(f"\nFor getting started: Read QUICKSTART.md")
        return_code = 0
    elif passed >= total * 0.9:
        print(f"\n{YELLOW}⚠ Setup mostly complete, but some items missing.{RESET}\n")
        print(f"Review the checks above and fix any issues.")
        return_code = 0
    else:
        print(f"\n{RED}✗ Setup incomplete. Please fix the issues above.{RESET}\n")
        return_code = 1

    return passed, total


def main():
    """Main verification function."""
    try:
        passed, total = verify_setup()
        sys.exit(0 if passed >= total * 0.9 else 1)
    except Exception as e:
        print(f"\n{RED}Error during verification: {e}{RESET}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
