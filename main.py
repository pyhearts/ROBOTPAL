"""ROBOTPAL command-line entry point."""
from __future__ import annotations
from typing import Sequence
from module.cli import run
from module.session_log import capture_output


def main(argv: Sequence[str] | None = None) -> int:
    with capture_output():
        return run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
