import builtins
import inspect
import math
import pickle
import sys
from typing import Dict, List, Optional, Set, Tuple, Union

from pydantic import BaseModel

from fle.env import entities as ent

from fle.env.entities import Entity
from fle.env.game_types import (
    Prototype,
    RecipeName,
    Resource,
    Technology,
    prototype_by_name,
)


class FactorioNamespace:
    def __init__(self, instance, agent_index):
        self.logging_results = {}
        self.line_value = 0
        self.persistent_vars = {}
        self.instance = instance
        self.tcp_port = instance.tcp_port
        self.max_sequential_exception_count = 1
        self._sequential_exception_count = 0
        self.log_counter = 0
        self.player_location = ent.Position(x=0, y=0)
        self.agent_index = agent_index
        self.agent_id = str(agent_index)

        # Add all builtins to the namespace
        for name in dir(builtins):
            if not name.startswith("_"):  # Skip private/special names
                try:
                    setattr(self, name, getattr(builtins, name))
                    self.persistent_vars[name] = getattr(builtins, name)
                except Exception as e:
                    print(f"Failed to add builtin {name}: {e}")

        # Define assert function
        def assert_(expr, msg=None):
            if not expr:
                raise AssertionError(msg)

        # Add specific builtins that we definitely want to expose
        self.essential_builtins = {
            "print": print,
            "len": len,
            "range": range,
            "int": int,
            "str": str,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
            "tuple": tuple,
            "set": set,
            "sum": sum,
            "min": min,
            "max": max,
            "enumerate": enumerate,
            "zip": zip,
            "map": map,
            "filter": filter,
            "any": any,
            "all": all,
            "sorted": sorted,
            "reversed": reversed,
            "round": round,
            "abs": abs,
            "isinstance": isinstance,
            "type": type,
            "assert": assert_,
        }

        for name, func in self.essential_builtins.items():
            setattr(self, name, func)
            self.persistent_vars[name] = func

        # Available objects that the agent can interact with
        self.Prototype = Prototype
        self.Resource = Resource
        self.Direction = ent.Direction
        self.Position = ent.Position
        self.EntityStatus = ent.EntityStatus
        self.BoundingBox = ent.BoundingBox
        self.BuildingBox = ent.BuildingBox
        self.BeltGroup = ent.BeltGroup
        self.Technology = Technology
        self.Recipe = ent.Recipe
        self.PipeGroup = ent.PipeGroup
        self.ElectricityGroup = ent.ElectricityGroup
        self.BeltGroup = ent.BeltGroup
        self.Pipe = ent.Pipe
        self.RecipeName = RecipeName

        self.prototype_by_name = prototype_by_name
        for name, value in self.prototype_by_name.items():
            if value.entity_class:
                setattr(self, value.name, value.entity_class)

        # Statically named directions
        self.UP, self.ABOVE, self.TOP = [ent.Direction.UP] * 3
        self.RIGHT, self.EAST = [ent.Direction.RIGHT] * 2
        self.LEFT, self.WEST = [ent.Direction.LEFT] * 2
        self.DOWN, self.BELOW, self.BOTTOM = [ent.Direction.DOWN] * 3

        # Math tools
        self.sqrt = math.sqrt
        self.sin = math.sin
        self.cos = math.cos
        self.tan = math.tan
        self.pi = math.pi
        self.floor = math.floor
        self.ceil = math.ceil
        self.abs = abs
        self.pow = pow

        # Type hints
        self.Optional = Optional
        self.Union = Union
        self.List = List
        self.Dict = Dict
        self.Tuple = Tuple
        self.Set = Set

        # Get all classes from the entities module
        entity_module = sys.modules[Entity.__module__]

        # Find all classes that inherit from BaseModel (which includes Entity and its subclasses)
        entity_classes = inspect.getmembers(
            entity_module,
            lambda member: (
                inspect.isclass(member)
                and issubclass(member, BaseModel)
                and member != BaseModel
            ),
        )

        # Add each entity class to both the namespace and persistent vars
        for name, entity_class in entity_classes:
            setattr(self, name, entity_class)

        # Add all the members of this class as static members so they can be accessed by the agent program.
        self._static_members = [
            attr
            for attr in dir(self)
            if not callable(getattr(self, attr)) and not attr.startswith("__")
        ]

        self._protected_names = set()

    def _freeze_protected_names(self):
        """Snapshot all current namespace names as protected."""
        self._protected_names = {
            attr for attr in dir(self) if not attr.startswith("__")
        }

    def load(self, namespace_str: bytes):
        try:
            env = pickle.loads(namespace_str)
            for key, value in env.items():
                self.persistent_vars[key] = value
                setattr(self, key, value)
        except Exception as e:
            print(f"Error restoring namespace: {e}")

    def reset(self):
        """
        Delete all variables that have accrued in the namespace, except for preexisting members.
        Also reset internal state variables to ensure clean test isolation.
        """
        self.logging_results = {}
        self.log_counter = 0
        self.line_value = 0
        self.player_location = ent.Position(x=0, y=0)
        self._sequential_exception_count = 0

        # Clear agent-defined variables from persistent_vars, keeping only builtins
        builtin_names = set(self.essential_builtins.keys()) | set(dir(builtins))
        self.persistent_vars = {
            k: v for k, v in self.persistent_vars.items() if k in builtin_names
        }

        # Clear agent-created attributes (non-callable, non-private, non-static)
        for attr in dir(self):
            if (
                not callable(getattr(self, attr))
                and attr[0] != "_"
                and attr not in self._static_members
            ):
                self[attr] = None

    def __getitem__(self, key):
        if key not in dir(self) or key.startswith("__"):
            raise KeyError(key)
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def log(self, *arg):
        self.log_counter += 1
        self.logging_results[self.log_counter] = [
            (self.line_value, repr(arg))
        ]
        return None

    def get_messages(self) -> List[Dict]:
        return []

    def load_messages(self, messages: List[Dict]):
        pass
