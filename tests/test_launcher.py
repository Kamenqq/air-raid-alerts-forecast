import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import run


class LauncherTest(unittest.TestCase):
    def test_command_routes_subcommands(self):
        self.assertEqual(run.command(["test"])[:2], ["-m", "unittest"])
        self.assertEqual(run.command(["snapshot"]), ["-m", "alerts.snapshot"])
        self.assertEqual(run.command(["--days", "3"]), ["-m", "alerts.main", "--days", "3"])

    def test_venv_failure_gives_instructions(self):
        missing = mock.Mock()
        missing.exists.return_value = False
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(run, "VENV", Path(tmp) / ".venv"), \
                mock.patch.object(run, "venv_python", return_value=missing), \
                mock.patch("venv.EnvBuilder.create", side_effect=OSError("no ensurepip")):
            with self.assertRaises(SystemExit) as ctx:
                run.ensure_env()
        self.assertIn("pip install -r requirements.txt", str(ctx.exception.code))

    def test_failed_venv_is_removed_so_a_rerun_starts_clean(self):
        # A real failed venv creation leaves .venv/bin/python behind; the next run would then crash in pip.
        with tempfile.TemporaryDirectory() as tmp:
            venv_dir = Path(tmp) / ".venv"

            def half_create(builder, env_dir):
                (Path(env_dir) / "bin").mkdir(parents=True)
                (Path(env_dir) / "bin" / "python").touch()
                raise subprocess.CalledProcessError(1, "ensurepip")

            with mock.patch.object(run, "VENV", venv_dir), \
                    mock.patch.object(run, "venv_python", return_value=venv_dir / "bin" / "python"), \
                    mock.patch("venv.EnvBuilder.create", half_create):
                with self.assertRaises(SystemExit):
                    run.ensure_env()
            self.assertFalse(venv_dir.exists())

    def test_pip_failure_gives_a_message_not_a_traceback(self):
        existing = mock.Mock()
        existing.exists.return_value = True
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(run, "venv_python", return_value=existing), \
                mock.patch.object(run, "STAMP", Path(tmp) / "stamp"), \
                mock.patch("subprocess.check_call", side_effect=subprocess.CalledProcessError(1, "pip")):
            with self.assertRaises(SystemExit) as ctx:
                run.ensure_env()
        self.assertIn("internet", str(ctx.exception.code))

    def test_ctrl_c_during_setup_removes_half_made_venv(self):
        with tempfile.TemporaryDirectory() as tmp:
            venv_dir = Path(tmp) / ".venv"

            def interrupted_create(builder, env_dir):
                (Path(env_dir) / "bin").mkdir(parents=True)
                (Path(env_dir) / "bin" / "python").touch()
                raise KeyboardInterrupt

            with mock.patch.object(run, "VENV", venv_dir), \
                    mock.patch.object(run, "venv_python", return_value=venv_dir / "bin" / "python"), \
                    mock.patch("venv.EnvBuilder.create", interrupted_create):
                with self.assertRaises(KeyboardInterrupt):
                    run.ensure_env()
            self.assertFalse(venv_dir.exists())

    def test_venv_is_rebuilt_clean(self):
        # clear=True wipes a stale install stamp, so a rebuilt .venv always gets its libraries again
        options = {}

        class FakeBuilder:
            def __init__(self, **kwargs):
                options.update(kwargs)

            def create(self, env_dir):
                raise OSError("stop here")

        missing = mock.Mock()
        missing.exists.return_value = False
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(run, "VENV", Path(tmp) / ".venv"), \
                mock.patch.object(run, "venv_python", return_value=missing), \
                mock.patch("venv.EnvBuilder", FakeBuilder):
            with self.assertRaises(SystemExit):
                run.ensure_env()
        self.assertTrue(options.get("clear"))
