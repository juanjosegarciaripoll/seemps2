import os
import sys
import time
import atexit
from collections import defaultdict

CURRENT_PROFILER: "Profiler | None" = None


class Profiler:
    """
    A simple profiler that tracks function calls and timing.

    Args:
        module_prefixes: List of module prefixes to filter (e.g., ['seemps']).
                        Only functions from these modules will be shown.
    """

    def __init__(self, module_prefixes: list[str] = ["seemps"]):
        self.module_prefixes = module_prefixes
        self.file_prefixes = [module.replace(".", "/") for module in module_prefixes]
        self.stats: dict[str, dict[str, float | int]] = defaultdict(
            lambda: {"calls": 0, "time": 0.0}
        )
        self.call_stack: list[tuple] = []

    def get_class_name(self, frame) -> str:
        """Try to extract the class name from a method's frame."""
        local_vars = frame.f_locals
        for var_name in ("self", "cls"):
            if var_name in local_vars:
                obj = local_vars[var_name]
                if var_name == "self":
                    return type(obj).__name__
                else:
                    return obj.__name__ if hasattr(obj, "__name__") else str(obj)
        return ""

    def format_func_name_from_code(self, code, frame) -> tuple[str, bool]:
        """Format function name from code object and return (name, should_include)."""
        filename = code.co_filename
        func_name = code.co_name
        lineno = code.co_firstlineno

        module_name = frame.f_globals.get("__name__", "")

        rel_filename = filename
        try:
            if "~" not in filename:
                rel_filename = os.path.relpath(filename, start="src")
        except (ValueError, TypeError):
            pass

        if self.module_prefixes:
            matches = any(
                module_name.startswith(prefix) for prefix in self.module_prefixes
            ) or any(subpath in rel_filename for subpath in self.file_prefixes)
            if not matches:
                return "", False

        class_name = self.get_class_name(frame)
        if class_name:
            qualified_name = f"{class_name}.{func_name}"
        else:
            qualified_name = func_name
        if qualified_name.startswith("pybind11_detail"):
            qualified_name.split(".")[-1]

        if "~" in rel_filename:
            func_location = qualified_name
        else:
            func_location = f"{rel_filename}:{lineno}({qualified_name})"

        if "<method" in func_location:
            return "", False

        return func_location, True

    def format_c_func_name(self, c_func) -> tuple[str, bool]:
        """Format C function (built-in, Cython) name and return (name, should_include)."""
        module = getattr(c_func, "__module__", None) or ""
        func_name = getattr(c_func, "__name__", str(c_func))

        qualified_name = getattr(c_func, "__qualified_name__", func_name)
        if qualified_name.startswith("pybind11_detail"):
            qualified_name.split(".")[-1]

        if self.module_prefixes:
            matches = any(module.startswith(prefix) for prefix in self.module_prefixes)
            if not matches:
                return "", False

        if module:
            func_location = f"{module}:({qualified_name})"
        else:
            func_location = f":({qualified_name})"

        return func_location, True

    def __call__(self, frame, event, arg):
        """Profile callback - compatible with sys.setprofile()."""
        current_time = time.perf_counter()
        match event:
            case "call":
                self.call_stack.append((frame, event, current_time, None))
            case "c_call":
                self.call_stack.append((frame, event, current_time, arg))
            case "return":
                if self.call_stack:
                    call_frame, call_event, start_time, _ = self.call_stack.pop()
                    if call_frame is frame and call_event == "call":
                        code = frame.f_code
                        func_name, should_include = self.format_func_name_from_code(
                            code, frame
                        )
                        if should_include:
                            elapsed = current_time - start_time
                            self.stats[func_name]["calls"] += 1
                            self.stats[func_name]["time"] += elapsed
            case "c_return" | "c_exception":
                if self.call_stack:
                    call_frame, call_event, start_time, c_func = self.call_stack.pop()
                    if call_frame is frame and call_event == "c_call" and c_func:
                        func_name, should_include = self.format_c_func_name(c_func)
                        if should_include:
                            elapsed = current_time - start_time
                            self.stats[func_name]["calls"] += 1
                            self.stats[func_name]["time"] += elapsed

    def print_report(self):
        """Print the profiling report and disable profiling."""
        sys.setprofile(None)
        if self.stats:
            sorted_stats = sorted(self.stats.items(), key=lambda x: x[0])
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


def hook_our_profiler(module_prefixes: list[str] = ["seemps"]):
    """
    A simple profiler that starts immediately and prints a report at program exit.
    Call this function once at module import time to enable profiling.

    Args:
        module_prefixes: List of module prefixes to filter (e.g., ['seemps']).
                        Only functions from these modules will be shown.
    """
    global CURRENT_PROFILER
    CURRENT_PROFILER = Profiler(module_prefixes)
    atexit.register(CURRENT_PROFILER.print_report)
    sys.setprofile(CURRENT_PROFILER)


__all__ = ["hook_our_profiler", "Profiler", "CURRENT_PROFILER"]
