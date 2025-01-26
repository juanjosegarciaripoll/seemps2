import sys
import types
from typing import Any, Optional

if "./" not in sys.path:
    sys.path += ["./"]

import seemps


def module_tree_recurse(
    object: Any,
    name: Optional[str] = None,
    prefix="",
    remove_hidden=True,
    remove_builtins=True,
    root_path: Optional[str] = None,
    shown: set = set(),
):
    if name is None:
        name = object.__name__
    if root_path is None:
        root_path = object.__path__[0]

    def module_path(module: types.ModuleType):
        output = root_path
        if "__path__" in module.__dict__:
            output = module.__path__[0]
        else:
            spec = module.__spec__
            output = None if spec is None else spec.origin
        return output

    def outside_module(object: Any):
        return isinstance(object, types.ModuleType) and not module_path(
            object
        ).startswith(root_path)

    def to_be_shown(name: str, object: Any):
        if remove_hidden and name.startswith("__"):
            return False
        if outside_module(object) and remove_builtins:
            return False
        return True

    def to_be_traversed(object: Any):
        if isinstance(object, types.ModuleType):
            if object in shown:
                return False
            shown.add(object)
            return not outside_module(object)
        return False

    if to_be_shown(name, object):
        print(f"{prefix}{name} ({type(object)})")
        if to_be_traversed(object):
            shown.add(object)
            prefix += " |-"
            for name, o in object.__dict__.items():
                module_tree_recurse(
                    o,
                    name,
                    prefix=prefix,
                    remove_hidden=remove_hidden,
                    remove_builtins=remove_builtins,
                    root_path=root_path,
                )


def module_tree(root_module, **kwdargs):
    module_tree_recurse(root_module, shown=set(), **kwdargs)


module_tree(seemps)
