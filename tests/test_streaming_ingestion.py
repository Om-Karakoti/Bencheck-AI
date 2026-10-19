import io
import os
import tempfile
import unittest

import pandas as pd

from src.data.streaming_ingestion import ingest_uploaded_files, remove_source


class NamedUpload(io.BytesIO):
    def __init__(self, name, content):
        super().__init__(content.encode("utf-8"))
        self.name = name


class StreamingIngestionTests(unittest.TestCase):
    def test_ingestion_and_single_file_removal_keep_latest_telemetry(self):
        first = NamedUpload("first.csv", "timestamp,session_id,bytes_per_flow\n1,a,10\n2,a,20\n")
        second = NamedUpload("second.csv", "timestamp,session_id,bytes_per_flow\n3,b,30\n4,b,40\n")
        with tempfile.TemporaryDirectory() as directory:
            data_file = os.path.join(directory, "dataset.parquet")
            meta_file = os.path.join(directory, "meta.json")
            combined, meta = ingest_uploaded_files([first, second], directory, data_file, meta_file)

            self.assertEqual(meta["total_flows"], 4)
            self.assertEqual(combined["timestamp"].tolist(), [1, 2, 3, 4])
            self.assertEqual(set(combined["_source_file"]), {"first.csv", "second.csv"})

            retained, retained_meta = remove_source("second.csv", directory, data_file, meta_file)
            self.assertEqual(retained_meta["total_flows"], 2)
            self.assertEqual(retained["timestamp"].tolist(), [1, 2])

