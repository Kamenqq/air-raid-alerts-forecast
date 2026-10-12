"""One-command launcher (standard library only).

Creates a private environment in .venv, installs requirements.txt into it once,
then runs the app inside it:
    python3 run.py [--start YYYY-MM-DD] [--days N] [--fresh]
    python3 run.py test        # run the unit tests
    python3 run.py snapshot    # maintainer only: rebuild data/snapshot (slow, uses the network)
"""
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
REQUIREMENTS = ROOT / "requirements.txt"
STAMP = VENV / "installed-requirements.txt"


def venv_python():
    return VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def ensure_env():
    python = venv_python()
    if not python.exists():
        print("Creating a private environment in .venv (first run only)...")
        try:
            venv.EnvBuilder(with_pip=True, clear=True).create(VENV)  # clear: no stale install stamp survives
        except BaseException as error:  # incl. Ctrl-C; e.g. Debian/Ubuntu without the python3-venv package
            shutil.rmtree(VENV, ignore_errors=True)  # a half-made .venv would break every later run
            if isinstance(error, KeyboardInterrupt):
                raise
            sys.exit(
                f"Could not create .venv ({error}).\n"
                "Fix: install your system's venv package (Debian/Ubuntu: sudo apt install python3-venv),\n"
                "or install the libraries yourself: pip install -r requirements.txt, then run: python -m alerts.main"
            )
    wanted = REQUIREMENTS.read_text()
    if not STAMP.exists() or STAMP.read_text() != wanted:
        print("Installing libraries from requirements.txt (first run only, needs internet)...")
        pip = [str(python), "-m", "pip", "install", "--disable-pip-version-check", "-q"]
        try:
            subprocess.check_call(pip + ["--upgrade", "pip"])
            subprocess.check_call(pip + ["-r", str(REQUIREMENTS)])
        except subprocess.CalledProcessError:
            sys.exit("Installing the libraries failed (see pip's message above). Check the internet connection "
                     "and run the same command again, or install them yourself: pip install -r requirements.txt")
        STAMP.write_text(wanted)
    return python


def command(args):
    if args[:1] == ["test"]:
        return ["-m", "unittest", "discover", "-s", "tests", "-t", "."]
    if args[:1] == ["snapshot"]:
        return ["-m", "alerts.snapshot"]
    return ["-m", "alerts.main", *args]


def main():
    if sys.version_info < (3, 9):
        sys.exit("Python 3.9 or newer is required.")
    python = ensure_env()
    sys.exit(subprocess.call([str(python), *command(sys.argv[1:])], cwd=ROOT))


if __name__ == "__main__":
    main()
