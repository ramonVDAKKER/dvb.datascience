import unittest

import pandas as pd

import dvb.datascience as ds


class TestCsv(unittest.TestCase):
    def setUp(self):
        self.pipeline = ds.Pipeline()

        self.train_data = pd.DataFrame(
            [
                ["jan", 20, "M", 180],
                ["marie", 21, "W", 164],
                ["piet", 23, "M", 194],
                ["helen", 24, "W", 177],
                ["jan", 60, "U", 188],
            ],
            columns=["name", "age", "gender", "length"],
        ).sort_index(axis=1)

        self.file_path = "test/data/train.csv"
        with open(self.file_path) as f:
            self.content = f.read()

    def test_read_file(self):
        p = self.pipeline
        p.addPipe("read", ds.data.CSVDataImportPipe(file_path=self.file_path))
        p.transform()
        df = p.get_pipe_output("read")["df"]
        self.assertTrue(pd.DataFrame.equals(df, self.train_data))

    def test_read_content(self):
        p = self.pipeline
        p.addPipe("read", ds.data.CSVDataImportPipe(content=self.content))
        p.transform()
        df = p.get_pipe_output("read")["df"]
        self.assertTrue(pd.DataFrame.equals(df, self.train_data))


    def test_read_init_params(self):
        p = self.pipeline
        p.addPipe("read", ds.data.CSVDataImportPipe())
        p.transform(
            data=None, transform_params={"read": {"file_path": "test/data/train.csv"}}
        )
        df = p.get_pipe_output("read")["df"]
        self.assertTrue(pd.DataFrame.equals(df, self.train_data))

    def test_read_custom_separator(self):
        pass

    def test_read_custom_agent(self):
        pass

    def test_read_custom_index(self):
        pass

    def test_write(self):
        output_file = "tmp/unittest-csv_test-test_write_output.csv"

        p = self.pipeline
        p.addPipe("read", ds.data.CSVDataImportPipe(file_path="test/data/test.csv"))
        p.addPipe(
            "write",
            ds.data.CSVDataExportPipe(file_path=output_file),
            [("read", "df", "df")],
        )
        p.transform()
        output = p.get_pipe_output("write")
        self.assertEqual(output, dict())

        # Inspect the file on disk
        with open(output_file, "r") as content_file:
            content = content_file.read()
            self.assertEqual(
                content, ",age,gender,length,name\n0,25,W,161,gea\n1,65,M,181,marc\n"
            )

    def test_write_init_params(self):
        pass

    def test_write_custom_separator(self):
        pass
