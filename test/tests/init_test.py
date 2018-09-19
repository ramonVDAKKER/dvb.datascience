import unittest

import pytest

import dvb.datascience as ds


class TestInitMethods(unittest.TestCase):
    @pytest.mark.skip()
    def test_run_module(self):
        ds.run_module("score_test_script").run()
