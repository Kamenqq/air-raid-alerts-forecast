import unittest
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
        with mock.patch.object(run, "venv_python", return_value=missing), \
                mock.patch("venv.EnvBuilder.create", side_effect=OSError("no ensurepip")):
            with self.assertRaises(SystemExit) as ctx:
                run.ensure_env()
        self.assertIn("pip install -r requirements.txt", str(ctx.exception.code))
