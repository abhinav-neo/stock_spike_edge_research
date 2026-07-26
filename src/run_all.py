\
from __future__ import annotations

import subprocess
import sys


def run(command: list[str]) -> None:
    print("\n>", " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    py = sys.executable
    run([py, "-m", "src.universe"])
    run([py, "-m", "src.download_prices"])
    run([py, "-m", "src.event_study"])
    run([py, "-m", "src.analyze_edges"])


if __name__ == "__main__":
    main()
