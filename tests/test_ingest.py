import csv
import os
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
from lcr.ingest.normalizer import normalize_doi, reverse_doi, generate_manifest_csv
from lcr.ingest.zotero import ZoteroIngestor, ZoteroRecord


class TestIngest(unittest.TestCase):
    def test_doi_normalization(self):
        original = "10.1016/j.xxx"
        normalized = normalize_doi(original)
        self.assertEqual(normalized, "10.1016_j.xxx")
        
        reversed_doi_val = reverse_doi(normalized)
        self.assertEqual(reversed_doi_val, original.lower())

    def test_generate_manifest_csv(self):
        records = [
            {"file_location": "/path/1.pdf", "doi": "10.1/1", "title": "T1"},
            {"file_location": "/path/2.pdf", "doi": "10.1/2", "title": "T2"},
            {"file_location": "/path/3.pdf", "doi": "10.1/3", "title": "T3"},
        ]
        test_csv = "test_manifest.csv"
        try:
            generate_manifest_csv(records, test_csv)
            
            self.assertTrue(os.path.exists(test_csv))
            with open(test_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                
                self.assertEqual(reader.fieldnames, ["file_location", "doi", "title"])
                self.assertEqual(len(rows), 3)
                self.assertEqual(rows[0]["doi"], "10.1/1")
        finally:
            if os.path.exists(test_csv):
                os.remove(test_csv)

    def test_zotero_to_manifest_records(self):
        # Mock ZoteroIngestor initialization to avoid searching for real Zotero dir
        with patch.object(ZoteroIngestor, '__init__', return_value=None):
            ingestor = ZoteroIngestor()
            
            records = [
                ZoteroRecord(doi="10.1/1", item_id=1, item_key="A", pdf_path="/abs/path/1.pdf"),
                ZoteroRecord(doi="10.1/2", item_id=2, item_key="B", pdf_path=None),
                ZoteroRecord(doi="10.1/3", item_id=3, item_key="C", pdf_path="/abs/path/3.pdf"),
            ]
            
            manifest_records = ingestor.to_manifest_records(records)
            
            self.assertEqual(len(manifest_records), 2)
            self.assertEqual(manifest_records[0]["doi"], "10.1/1")
            self.assertEqual(manifest_records[0]["file_location"], "/abs/path/1.pdf")
            self.assertEqual(manifest_records[1]["doi"], "10.1/3")

    @patch("sqlite3.connect")
    @patch("lcr.ingest.zotero.getpass.getuser", return_value="testuser")
    @patch("pathlib.Path.exists", return_value=True)
    def test_zotero_load_records_mock(self, mock_exists, mock_getuser, mock_connect):
        # Test load_records with mocked SQLite
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        mock_cursor = mock_conn.cursor.return_value
        
        # Mock DOI_QUERY results
        mock_cursor.fetchall.return_value = [
            (1, "KEY1", "10.1001/test"),
            (2, "KEY2", "10.1002/test")
        ]
        
        # Mock ATTACHMENT_QUERY results
        # First call for item 1: has attachment
        # Second call for item 2: no attachment
        mock_cursor.fetchone.side_effect = [
            ("ATTKEY1", "storage:test.pdf", "application/pdf"),
            None
        ]
        
        # We need to mock Path.exists specifically for the storage path check inside load_records
        # But since we already patched Path.exists globally in this test, it will return True
        
        with patch("lcr.ingest.zotero.Path.absolute", side_effect=lambda: Path("/fake/zotero/storage/ATTKEY1/test.pdf")):
            ingestor = ZoteroIngestor(zotero_dir="/fake/zotero")
            records = ingestor.load_records()
            
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0].doi, "10.1001/test")
            self.assertIsNotNone(records[0].pdf_path)
            self.assertEqual(records[1].doi, "10.1002/test")
            self.assertIsNone(records[1].pdf_path)

if __name__ == "__main__":
    unittest.main()
