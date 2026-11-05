"""Create a project-local environment and install the tested training packages."""

import argparse
from pathlib import Path
import subprocess
import sys
import venv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["cu130", "cpu"], default="cu130")
    parser.add_argument("--venv", type=Path, default=Path(__file__).resolve().parents[1] / ".venv")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without creating or installing")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    python = args.venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    index = f"https://download.pytorch.org/whl/{args.backend}"
    commands = [
        [str(python), "-m", "pip", "install", "--upgrade", "pip"],
        [str(python), "-m", "pip", "install", "torch==2.14.1", "torchvision==0.29.1", "--index-url", index],
        [str(python), "-m", "pip", "install", "-r", str(root / "requirements.txt")],
        [str(python), "-m", "pip", "check"],
    ]
    if args.dry_run:
        print(f"Create environment with Python {sys.version.split()[0]}: {args.venv}")
        for command in commands:
            print(subprocess.list2cmdline(command))
        return
    if sys.version_info[:2] != (3, 14):
        parser.error("Use Python 3.14 for the version-pinned environment documented by this project")
    if python.exists():
        parser.error(f"Environment already exists: {args.venv}; choose another --venv to avoid modifying it")
    venv.EnvBuilder(with_pip=True).create(args.venv)
    for command in commands:
        subprocess.run(command, check=True)
    print(f"Ready: {python}")


if __name__ == "__main__":
    main()
