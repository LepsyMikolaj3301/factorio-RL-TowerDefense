"""Test that TaskResponse is importable from its new location."""

import pytest


class TestTaskResponseMigration:
    def test_import_from_commons(self):
        from fle.commons.models.task_response import TaskResponse

        assert TaskResponse is not None

    def test_create_instance(self):
        from fle.commons.models.task_response import TaskResponse

        resp = TaskResponse(success=True, meta={"key": "value"})
        assert resp.success is True
        assert resp.meta["key"] == "value"

    def test_failure_response(self):
        from fle.commons.models.task_response import TaskResponse

        resp = TaskResponse(success=False, meta={})
        assert resp.success is False
