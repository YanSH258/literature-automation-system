import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
from lcr.ingest.mineru_parser import parse_pdf_to_markdown, _find_md

class TestMineruParser(unittest.TestCase):
    @patch("shutil.which")
    @patch("subprocess.run")
    def test_mineru_not_installed(self, mock_run, mock_which):
        mock_which.return_value = None
        result = parse_pdf_to_markdown("dummy.pdf")
        self.assertIsNone(result)
        mock_run.assert_not_called()

    @patch("lcr.ingest.mineru_parser.Path.exists", return_value=False)
    def test_find_md_not_exists(self, mock_exists):
        result = _find_md(Path("/non/existent"))
        self.assertIsNone(result)

    @patch("lcr.ingest.mineru_parser.Path.rglob")
    @patch("lcr.ingest.mineru_parser.Path.exists", return_value=True)
    def test_find_md_success(self, mock_exists, mock_rglob):
        mock_file = MagicMock()
        mock_file.stat.return_value.st_size = 200
        mock_rglob.return_value = [mock_file]
        
        result = _find_md(Path("/fake"))
        self.assertEqual(result, mock_file)

    @patch("lcr.ingest.mineru_parser._find_md")
    @patch("shutil.which")
    @patch("subprocess.run")
    def test_parse_pdf_cache_hit(self, mock_run, mock_which, mock_find_md):
        mock_md_file = MagicMock()
        mock_md_file.read_text.return_value = "cached content"
        mock_find_md.return_value = mock_md_file
        
        result = parse_pdf_to_markdown("test.pdf")
        self.assertEqual(result, "cached content")
        mock_run.assert_not_called()

if __name__ == "__main__":
    unittest.main()
