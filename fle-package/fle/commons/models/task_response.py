from typing import Any, Dict

from pydantic import BaseModel


class TaskResponse(BaseModel):
    meta: Dict[str, Any] = {}
    success: bool
