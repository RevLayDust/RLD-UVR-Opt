import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
BATCH_FILES = (
    ROOT_DIR / "run_uvr.bat",
    ROOT_DIR / "install" / "install_packages.bat",
)


class WindowsBatchLineEndingTests(unittest.TestCase):
    def test_windows_batch_files_use_crlf(self):
        for batch_file in BATCH_FILES:
            with self.subTest(batch_file=batch_file):
                content = batch_file.read_bytes()
                lone_line_feeds = content.replace(b"\r\n", b"")
                self.assertNotIn(
                    b"\n",
                    lone_line_feeds,
                    f"{batch_file} must use CRLF line endings",
                )


if __name__ == "__main__":
    unittest.main()
