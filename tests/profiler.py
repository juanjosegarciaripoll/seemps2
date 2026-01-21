import os
import sys
import time
import atexit
from collections import defaultdict


def hook_our_profiler(module_prefixes: list[str] = ["seemps"]):
    """
    A simple profiler that starts immediately and prints a report at program exit.
    Call this function once at module import time to enable profiling.

    Args:
        module_prefixes: List of module prefixes to filter (e.g., ['seemps']).
                        Only functions from these modules will be shown.
    """
    stats = defaultdict(lambda: {"calls": 0, "time": 0.0})
    call_stack = []
    file_prefixes = [module.replace(".", "/") for module in module_prefixes]

    def get_class_name(frame) -> str:
        """Try to extract the class name from a method's frame."""
        # Check for 'self' (instance method) or 'cls' (class method)
        local_vars = frame.f_locals
        for var_name in ("self", "cls"):
            if var_name in local_vars:
                obj = local_vars[var_name]
                if var_name == "self":
                    return type(obj).__name__
                else:
                    # cls is the class itself
                    return obj.__name__ if hasattr(obj, "__name__") else str(obj)
        return ""

    def format_func_name_from_code(code, frame) -> tuple[str, bool]:
        """Format function name from code object and return (name, should_include)."""
        filename = code.co_filename
        func_name = code.co_name
        lineno = code.co_firstlineno

        # Try to get module name from frame globals
        module_name = frame.f_globals.get("__name__", "")

        # Convert to relative path from "src"
        rel_filename = filename
        try:
            if "~" not in filename:
                rel_filename = os.path.relpath(filename, start="src")
        except (ValueError, TypeError):
            pass

        # Check if function matches module filter using module name and file path
        if module_prefixes:
            matches = any(
                module_name.startswith(prefix) for prefix in module_prefixes
            ) or any(subpath in rel_filename for subpath in file_prefixes)
            if not matches:
                return "", False

        # Try to get class name for methods
        class_name = get_class_name(frame)
        if class_name:
            qualified_name = f"{class_name}.{func_name}"
        else:
            qualified_name = func_name
        if qualified_name.startswith("pybind11_detail"):
            qualified_name.split(".")[-1]

        # Format the function location
        if "~" in rel_filename:
            func_location = qualified_name
        else:
            func_location = f"{rel_filename}:{lineno}({qualified_name})"

        # Exclude <method entries
        if "<method" in func_location:
            return "", False

        return func_location, True

    def format_c_func_name(c_func, frame) -> tuple[str, bool]:
        """Format C function (built-in, Cython) name and return (name, should_include)."""
        # Get module from the C function
        module = getattr(c_func, "__module__", None) or ""
        func_name = getattr(c_func, "__name__", str(c_func))

        # Try to get qualified name (includes class for methods)
        qualified_name = getattr(c_func, "__qualified_name__", func_name)
        if qualified_name.startswith("pybind11_detail"):
            qualified_name.split(".")[-1]

        # Check if function matches module filter
        if module_prefixes:
            matches = any(module.startswith(prefix) for prefix in module_prefixes)
            if not matches:
                return "", False

        # Format as module:qualified_name for C functions
        if module:
            func_location = f"{module}:({qualified_name})"
        else:
            func_location = f":({qualified_name})"

        return func_location, True

    def profile_callback(frame, event, arg):
        current_time = time.perf_counter()
        match event:
            case "call":
                call_stack.append((frame, event, current_time, None))
            case "c_call":
                # arg is the C function being called
                call_stack.append((frame, event, current_time, arg))
            case "return":
                if call_stack:
                    call_frame, call_event, start_time, _ = call_stack.pop()
                    if call_frame is frame and call_event == "call":
                        code = frame.f_code
                        func_name, should_include = format_func_name_from_code(
                            code, frame
                        )
                        if should_include:
                            elapsed = current_time - start_time
                            stats[func_name]["calls"] += 1
                            stats[func_name]["time"] += elapsed
            case "c_return" | "c_exception":
                if call_stack:
                    call_frame, call_event, start_time, c_func = call_stack.pop()
                    if call_frame is frame and call_event == "c_call" and c_func:
                        func_name, should_include = format_c_func_name(c_func, frame)
                        if should_include:
                            elapsed = current_time - start_time
                            stats[func_name]["calls"] += 1
                            stats[func_name]["time"] += elapsed

    def print_report():
        sys.setprofile(None)
        if stats:
            sorted_stats = sorted(stats.items(), key=lambda x: x[0])
            print(
                f"\n{'ncalls':<15} {'tottime(ms)':<15} {'ms/call':<15} {'filename:lineno(function)'}"
            )
            print("-" * 90)
            for name, data in sorted_stats:
                time_ms = data["time"] * 1000
                ms_per_call = time_ms / data["calls"] if data["calls"] > 0 else 0
                print(
                    f"{data['calls']:<15} {time_ms:<15.3f} {ms_per_call:<15.3f} {name}"
                )

    atexit.register(print_report)
    sys.setprofile(profile_callback)


__all__ = ["hook_our_profiler"]
