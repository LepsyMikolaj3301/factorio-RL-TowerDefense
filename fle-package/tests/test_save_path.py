"""Test that FactorioInstance records save_path correctly."""

import pytest
from unittest.mock import patch, MagicMock


class TestSavePath:
    @patch("fle.env.instance.FactorioInstance.connect_to_server")
    @patch("fle.env.instance.FactorioInstance.initialise")
    def test_save_path_stored(self, mock_init, mock_connect):
        from fle.env.instance import FactorioInstance

        instance = FactorioInstance.__new__(FactorioInstance)
        instance.save_path = "/tmp/test_save.zip"
        assert instance.save_path == "/tmp/test_save.zip"

    @patch("fle.env.instance.FactorioInstance.connect_to_server")
    @patch("fle.env.instance.FactorioInstance.initialise")
    def test_save_path_default_none(self, mock_init, mock_connect):
        from fle.env.instance import FactorioInstance

        instance = FactorioInstance.__new__(FactorioInstance)
        instance.save_path = None
        assert instance.save_path is None
