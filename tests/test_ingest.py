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
        
        # Mock fetchall results for all queries in order:
        # 1. ABSTRACT_QUERY
        # 2. META_ALL_QUERY
        # 3. item_keywords
        # 4. collections
        # 5. collectionItems
        # 6. DOI_QUERY
        # 7. itemAttachments
        mock_cursor.fetchall.side_effect = [
            [], # abstracts
            [(1, "T1", "2021"), (2, "T2", "2022")], # metadata
            [], # item_keywords
            [], # col_info
            [], # item_to_cols
            [(1, "KEY1", "10.1001/test"), (2, "KEY2", "10.1002/test")], # doi_rows
            [(1, "ATTKEY1", "storage:test.pdf")] # attachments
        ]
        
        # We need to mock Path.absolute specifically for the storage path check inside load_records
        with patch("lcr.ingest.zotero.Path.absolute", side_effect=lambda *args, **kwargs: Path("/fake/zotero/storage/ATTKEY1/test.pdf")):
            ingestor = ZoteroIngestor(zotero_dir="/fake/zotero")
            records = ingestor.load_records()
            
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0].doi, "10.1001/test")
            self.assertIsNotNone(records[0].pdf_path)
            self.assertEqual(records[1].doi, "10.1002/test")
            self.assertIsNone(records[1].pdf_path)


class TestChromaIngestor(unittest.TestCase):
    def test_chunk_text_references(self):
        from lcr.ingest.chroma_ingestor import ChromaIngestor
        with patch.object(ChromaIngestor, '__init__', return_value=None):
            ingestor = ChromaIngestor()
            text = "## Introduction\nSome text.\n## References\nRef 1. Ref 2."
            chunks = ingestor.chunk_text(text)
            for c in chunks:
                self.assertNotIn("References", c)
                self.assertNotIn("Ref 1.", c)

    def test_chunk_text_short(self):
        from lcr.ingest.chroma_ingestor import ChromaIngestor
        with patch.object(ChromaIngestor, '__init__', return_value=None):
            ingestor = ChromaIngestor()
            text = "## Intro\nShort."
            chunks = ingestor.chunk_text(text, chunk_size=100)
            self.assertEqual(len(chunks), 1)
            self.assertIn("## Intro", chunks[0])

    def test_chunk_text_empty(self):
        from lcr.ingest.chroma_ingestor import ChromaIngestor
        with patch.object(ChromaIngestor, '__init__', return_value=None):
            ingestor = ChromaIngestor()
            self.assertEqual(ingestor.chunk_text(""), [])


if __name__ == "__main__":
    unittest.main()
