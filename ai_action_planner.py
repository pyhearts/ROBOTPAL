"""Interactive compatibility entry point for text planning and replies."""
from __future__ import annotations
from module.cli import run
from module.session_log import capture_output


def main() -> int:
    with capture_output():
        return run(["--plan"])


if __name__ == "__main__":
    raise SystemExit(main())
