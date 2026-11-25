"""Zero-skip unittest runner for the Eir test suite."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import unittest


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m tests",
        description="Discover and execute Eir unit tests with zero-skip guarantees.",
    )
    parser.add_argument(
        "-p",
        "--pattern",
        default="test_*.py",
        help="Glob pattern for test files under api/tests (default: test_*.py)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose unittest output",
    )
    return parser.parse_args(argv)


def _discover_test_files(tests_dir: Path, pattern: str) -> list[Path]:
    return sorted(
        p for p in tests_dir.glob(pattern) if p.is_file() and not p.name.startswith("_")
    )


def _load_suites(test_files: list[Path]) -> tuple[unittest.TestSuite, list[str]]:
    loader = unittest.TestLoader()
    combined = unittest.TestSuite()
    bad_files: list[str] = []

    for path in test_files:
        module_name = f"tests.{path.stem}"
        before_errors = len(loader.errors)
        try:
            suite = loader.loadTestsFromName(module_name)
        except Exception as exc:
            print(f"ERROR loading {path.name} ({module_name}): {exc}", file=sys.stderr)
            bad_files.append(path.name)
            continue

        if len(loader.errors) > before_errors or suite.countTestCases() == 0:
            for err in loader.errors[before_errors:]:
                print(err, file=sys.stderr)
            bad_files.append(path.name)
            continue
        combined.addTest(suite)

    return combined, bad_files


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    tests_dir = Path(__file__).resolve().parent
    test_files = _discover_test_files(tests_dir, args.pattern)
    if not test_files:
        print(
            f"ERROR: No test files matching '{args.pattern}' found in {tests_dir}",
            file=sys.stderr,
        )
        return 1

    combined, bad_files = _load_suites(test_files)
    if bad_files:
        print(
            f"ERROR: Zero tests or loader errors in: {', '.join(bad_files)}",
            file=sys.stderr,
        )
        return 1

    runner = unittest.TextTestRunner(verbosity=2 if args.verbose else 1)
    result = runner.run(combined)
    return 0 if result.wasSuccessful() and result.testsRun > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
