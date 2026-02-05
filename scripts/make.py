#!/usr/bin/python3
import os
import re
import sys
import shutil
import subprocess
import argparse
import pstats
import tempfile

incremental: bool = True
debug: bool = False
use_sanitizer: str = "no" if "SANITIZE" not in os.environ else os.environ["SANITIZE"]
ld_preload: str = ""
valgrind: list[str] = []
python: list[str] = ["python" if sys.platform == "win32" else "python3"]
uv_run: list[str] = []


def run(command: list[str], *args, **kwdargs) -> bool:
    print(f"Running command:\n{' '.join(command)}", file=sys.stderr, flush=True)
    return subprocess.run(command, *args, **kwdargs).returncode == 0


def asan_library() -> str:
    if sys.platform in ["win32", "cygwin"]:
        return ""
    else:
        return run_output(
            [
                "/bin/sh",
                "-c",
                r"ldconfig -p |grep asan | sed 's/^\(.*=> \)\(.*\)$/\2/g' | sed '/^[[:space:]]*$/d'",
            ]
        )


def cpp_library() -> str:
    if sys.platform in ["win32", "cygwin"]:
        return ""
    else:
        return run_output(
            [
                "/bin/sh",
                "-c",
                r"ldconfig -p |grep stdc++ | sed 's/^\(.*=> \)\(.*\)$/\2/g' | sed '/^[[:space:]]*$/d'",
            ]
        )


def run_many(commands: list[list[str]], *args, chain: bool = True, **kwdargs) -> bool:
    ok = True
    for c in commands:
        if not run(c, *args, **kwdargs):
            ok = False
            if chain:
                return ok
    return ok


def run_output(command: list[str], *args, **kwdargs) -> str:
    s = str(subprocess.check_output(command, *args, **kwdargs), encoding="utf-8")
    return s.rstrip(" \n\r")


def delete_directories(patterns: list[str], root: str = "."):
    to_delete = []
    for dirname, _, _ in os.walk(root):
        if os.path.basename(dirname) in patterns:
            to_delete.append(dirname)
    for path in to_delete:
        if os.path.exists(path):
            print(f"Removing {path}")
            shutil.rmtree(path)


def delete_files(patterns: list[str], root: str = "."):
    regexp = [re.compile(p) for p in patterns]
    for actual_root, _, files in os.walk(root):
        for f in files:
            if any(p.match(f) for p in regexp):
                path = actual_root + "/" + f
                print(f"Removing {path}")
                os.remove(path)


def clean() -> None:
    for path in [
        "build",
        "dist",
        "_site",
        "docs/generated",
        "docs/algorithms/generated",
    ]:
        if os.path.exists(path):
            print(f"Removing {path}")
            shutil.rmtree(path)
    delete_directories(["SeeMPS.egg-info", "seemps.egg-info"], root="src")
    delete_directories(["__pycache__"], root="src")
    delete_directories(["__pycache__"], root="test")
    delete_files([r".*.rst"], root="docs/api/class")
    delete_files([r".*.rst"], root="docs/api/function")
    delete_files([r".*\.so", r".*\.c", r".*\.cc", r".*\.pyd", r".*\.pyc"], root="src")


def cProfile_formatter(report: str) -> str:
    def function_name(line: str) -> str:
        n = line.find("{")
        if n < 0:
            n = line.rfind(" ") + 1
        name = line[n:].strip()
        if "seemps.cython" in line:
            return " " + name
        else:
            return name

    lines = report.split("\n")
    while "ncalls" not in lines[0]:
        lines = lines[1:]
    lines = lines[0:1] + sorted(lines[1:], key=function_name)
    return "\n".join([l for l in lines if l])


def cProfile_list(filename: str, module_prefixes: list[str] = []) -> str:
    """
    Read a cProfile output file and produce a formatted listing.

    Args:
        filename: Path to the cProfile output file
        module_prefixes: Optional list of module prefixes to filter (e.g., ['seemps', 'test'])
                        If None, shows all functions

    Returns:
        A string containing the formatted profile statistics with:
        - Number of calls
        - Total time (in milliseconds)
        - Time per call (in milliseconds)
        - Function name
    """
    # Load the profile data
    stats = pstats.Stats(filename)

    # Build the output manually for better control over formatting
    lines = []
    lines.extend(module_prefixes if module_prefixes else ["No filters"])
    lines.append(
        f"{'ncalls':<15} {'tottime(ms)':<15} {'ms/call':<15} {'filename:lineno(function)'}"
    )
    lines.append("-" * 90)

    # Get all stats sorted by total time
    stats.sort_stats("tottime")

    # Access the internal stats dictionary
    # stats.stats is a dict: {(filename, lineno, funcname): (cc, nc, tt, ct, callers)}
    stats_dict = stats.stats  # type: ignore

    # Filter and format each entry
    file_prefixes = [module.replace(".", "/") for module in module_prefixes]
    sorted_lines: list[tuple[str, str]] = []
    for func, (cc, nc, tt, _, _) in stats_dict.items():
        filename_str, line, func_name = func

        # Convert filename to relative path
        rel_filename = filename_str
        try:
            if "~" not in filename_str:
                rel_filename = os.path.relpath(filename_str, start="src")
        except (ValueError, TypeError):
            # If relpath fails, use the original filename
            pass

        # Apply module prefix filter if specified
        if module_prefixes:
            if not (
                any(prefix in func_name for prefix in module_prefixes)
                or any(subpath in rel_filename for subpath in file_prefixes)
            ):
                continue

        # Format call count (handles recursive calls)
        if cc != nc:
            ncalls_str = f"{nc}/{cc}"
        else:
            ncalls_str = str(nc)

        # Convert times to milliseconds for better resolution
        tt_ms = tt * 1000
        percall_tt = (tt_ms / nc) if nc != 0 else 0

        # Format the function location
        if "~" in rel_filename:
            func_location = func_name
        else:
            func_location = f"{rel_filename}:{line}({func_name})"

        # Format the line
        line_str = (
            f"{ncalls_str:<15} {tt_ms:<15.3f} {percall_tt:<15.3f} {func_location}"
        )
        if "<method" not in func_location:
            sorted_lines.append((func_location, line_str))
    sorted_lines.sort(key=lambda x: x[0])

    return "\n".join(lines + [line for _, line in sorted_lines])


def run_tests(
    verbose=False,
    coverage=False,
    cProfile: bool = False,
    filter: str | None = None,
    test_profiler: bool = False,
    capture_output: bool = False,
    **kwdargs,
) -> bool | str:
    """
    Run unit tests with various profiling options.

    Args:
        verbose: Enable verbose test output
        coverage: Enable coverage reporting
        cProfile: Enable cProfile profiling
        filter: Test filter pattern
        test_profile: If True, enable SEEMPS_TEST_PROFILE=calls and return
                     the collected call counts as a dict instead of bool

    Returns:
        If test_profile is False: bool indicating success
        If test_profile is True: dict mapping test names to function call counts
    """
    env: dict[str, str] | None
    if use_sanitizer != "no":
        env = os.environ.copy()
        env["LD_PRELOAD"] = asan_library() + " " + cpp_library()
        print(f"LD_PRELOAD={env['LD_PRELOAD']}")
    else:
        env = os.environ.copy() if test_profiler else None
    profile_filename = ""
    command = valgrind + uv_run + python
    if test_profiler:
        env = env or os.environ.copy()
        env["SEEMPS_TEST_PROFILE"] = "calls"
    if cProfile:
        profile_fd, profile_filename = tempfile.mkstemp(suffix=".prof")
        os.close(profile_fd)
        command += ["-m", "cProfile", "-o", profile_filename]
    command += ["-m", "unittest", "-f"]
    if filter:
        if isinstance(filter, list) and len(filter) == 1:
            filter = filter[0]
        elif isinstance(filter, list):
            filter = r"\(" + "|".join(filter) + r"\)"
        command.extend(["-k", filter])
    if verbose:
        command.append("-v")

    if capture_output:
        # Capture stdout only (let stderr go to terminal)
        result = subprocess.run(command, env=env, stdout=subprocess.PIPE, text=True)
        return result.stdout if result.returncode == 0 else "{}"
    else:
        ok = run(command, env=env, **kwdargs)
        print(f"Output of tests is {ok}", file=sys.stderr, flush=True)
        if cProfile and ok:
            print(cProfile_list(profile_filename, module_prefixes=["seemps"]))
            os.unlink(profile_filename)
        return ok


def code_coverage_report() -> bool:
    return run(["uv", "run", "coverage", "lcov"])


def build_documentation(verbose=False) -> bool:
    for path in ["_site", "docs/generated", "docs/algorithms/generated"]:
        if os.path.exists(path):
            shutil.rmtree(path)
    return run(uv_run + ["sphinx-build", "-q", "docs", "_site"])


def install_hooks() -> bool:
    return run(["uv", "run", "pre-commit", "install", "--install-hooks", "--overwrite"])


def mypy() -> bool:
    return run(uv_run + ["mypy", "src/seemps"])


def ruff() -> bool:
    return run(uv_run + ["ruff", "check", "src"])


def basedpyright() -> bool:
    return run(uv_run + ["basedpyright", "src/seemps"])


def pydocstyle() -> bool:
    return run(uv_run + ["ruff", "check", "--select", "D", "src/seemps"])


def darglint() -> bool:
    return run(uv_run + ["darglint", "src/seemps"])


def compare_tests_with_different_cores(filter: str | None = None) -> list[dict]:
    """
    Run tests with and without SEEMPS_PYBIND and compare function call counts.

    Returns:
        List of discrepancies, each containing:
        - test_name: Name of the test
        - function: Name of the function
        - pybind_calls: Number of calls with PYBIND
        - no_pybind_calls: Number of calls without PYBIND
    """
    import json

    def rename_core_functions(data: str) -> str:
        return data.replace(
            "pybind11_detail_function_record_v1_system_libstdcpp_gxx_abi_1xxx_use_cxx11_abi_1.",
            "",
        )

    def output_to_json(stdout: str | bool) -> dict:
        # Find and parse JSON from stdout
        if not isinstance(stdout, str):
            stdout = "{}"
        json_start = stdout.rfind("\n{")
        if json_start == -1:
            json_start = stdout.find("{")
        if json_start != -1:
            json_str = rename_core_functions(stdout[json_start:].strip())
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                raise Exception(f"Failed to parse JSON: {e}")
        else:
            raise Exception("No JSON output found in test results")

    # Save and restore SEEMPS_PYBIND state
    original_pybind = os.environ.get("SEEMPS_PYBIND")

    # Run tests with PYBIND=ON
    print("Running tests with SEEMPS_PYBIND=ON...", file=sys.stderr, flush=True)
    os.environ["SEEMPS_PYBIND"] = "ON"
    pybind_results = output_to_json(
        run_tests(filter=filter, test_profiler=True, capture_output=True)
    )
    assert isinstance(pybind_results, dict)

    # Run tests without PYBIND
    print("Running tests with SEEMPS_PYBIND=OFF...", file=sys.stderr, flush=True)
    if "SEEMPS_PYBIND" in os.environ:
        del os.environ["SEEMPS_PYBIND"]
    no_pybind_results = output_to_json(
        run_tests(filter=filter, test_profiler=True, capture_output=True)
    )
    assert isinstance(no_pybind_results, dict)

    # Restore original state
    if original_pybind is not None:
        os.environ["SEEMPS_PYBIND"] = original_pybind

    discrepancies: list[dict] = []

    # Get all test names from both runs
    all_tests = set(pybind_results.keys()) | set(no_pybind_results.keys())

    for test_name in sorted(all_tests):
        pybind_funcs = {
            func.split(".")[-1]: count
            for func, count in pybind_results.get(test_name, {}).items()
            if ("seemps.cython.pybind" in func)
            if (".get_" not in func)
            if (".replace" not in func)
        }
        no_pybind_funcs = {
            func.split(".")[-1]: count
            for func, count in no_pybind_results.get(test_name, {}).items()
            if ("seemps.cython.core" in func)
        }

        # Get all function names for this test
        all_funcs = set(pybind_funcs.keys()) | set(no_pybind_funcs.keys())

        for func_name in sorted(all_funcs):
            pybind_calls = pybind_funcs.get(func_name, 0)
            no_pybind_calls = no_pybind_funcs.get(func_name, 0)

            if pybind_calls != no_pybind_calls:
                discrepancies.append(
                    {
                        "test_name": test_name,
                        "function": func_name,
                        "pybind_calls": pybind_calls,
                        "no_pybind_calls": no_pybind_calls,
                    }
                )

    # Print summary
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS: SEEMPS_PYBIND=ON vs OFF")
    print("=" * 80)

    if not discrepancies:
        print("\nNo discrepancies found! All function call counts match.")
    else:
        print(f"\nFound {len(discrepancies)} discrepancies:\n")
        print(f"{'Function':<40} {'PYBIND':<10} {'No PYBIND':<10}")
        print("-" * 110)

        current_test = None
        for d in discrepancies:
            test_display = d["test_name"] if d["test_name"] != current_test else ""
            current_test = d["test_name"]
            print(test_display)
            print(
                f"  {d['function']:<38} {d['pybind_calls']:<10} {d['no_pybind_calls']:<10}"
            )

    print("\n" + "=" * 80)
    return discrepancies


def check(verbose: bool = False, filter: str | None = None) -> bool:
    return (
        bool(run_tests(filter=filter)) and mypy() and ruff() and basedpyright()
        # and pydocstyle()
        # and darglint()
    )


def build() -> bool:
    env = os.environ.copy()
    extra: list[str] = []
    if use_sanitizer != "no":
        env["SEEMPS_ASAN"] = use_sanitizer
    if debug:
        env["SEEMPS_DEBUG"] = "ON"
    if incremental:
        extra = ["--no-build-isolation", "-ve"] + extra
    return run(["pip", "install"] + extra + ["."], env=env)


def install() -> bool:
    return run(["pip", "install", "."])


parser = argparse.ArgumentParser(prog="make", description="SeeMPS build driver")
parser.add_argument("--uv", action="store_true", help="Run jobs under 'uv'")
parser.add_argument("--debug", action="store_true", help="Build debug versions")
parser.add_argument("--here", action="store_true", help="Run jobs incrementally")
parser.add_argument("--pybind", action="store_true", help="Activate pybind11 bindings")
parser.add_argument("--leak", action="store_true", help="Link against leak sanitizer")
parser.add_argument(
    "--memcheck", action="store_true", help="Run inside valgrind to check memory leaks"
)
parser.add_argument("--asan", action="store_true", help="Use ASAN as leak sanitizer")
parser.add_argument(
    "--clean",
    action="store_true",
    help="Clean temporary files (including C/C++ Cython files)",
)
parser.add_argument("--build", action="store_true", help="Build library")
parser.add_argument("--install", action="store_true", help="Install library")
parser.add_argument(
    "--install-hooks", action="store_true", help="Install git pre-commit hooks"
)
parser.add_argument("--check", action="store_true", help="Run tests")
parser.add_argument(
    "--coverage", action="store_true", help="Run tests with coverage report"
)
parser.add_argument("--verbose", action="store_true", help="Verbose mode")
parser.add_argument("--tests", action="store_true", help="Run unit tests")
parser.add_argument("--filter", nargs=1, help="Regular expression to select tests")
parser.add_argument("--cProfile", action="store_true", help="Run tests with cProfile")
parser.add_argument(
    "--compare-test", action="store_true", help="Compare unit tests with cores"
)
parser.add_argument("--pyright", action="store_true", help="Run pyright")
parser.add_argument("--ruff", action="store_true", help="Run ruff")
parser.add_argument("--mypy", action="store_true", help="Run mypy")
parser.add_argument("--docs", action="store_true", help="Run Sphynx")
parser.add_argument("--pydocstyle", action="store_true", help="Run pydocstyle")
parser.add_argument("--darglint", action="store_true", help="Run darglint")

args = parser.parse_args()

if args.verbose:
    print(f"Running from directory {os.getcwd()}")
    print("Environment variables:")
    for key, value in os.environ.items():
        print(f"{key} = {value}")

debug = args.debug
incremental = args.here
if args.uv or "UV_RUN_RECURSION_DEPTH" in os.environ:
    uv_run = ["uv", "run"]
    python = ["python"]
if args.leak:
    use_sanitizer = "leak"
    debug = True
if args.memcheck:
    valgrind = [
        "valgrind",
        "--leak-check=full",
        "--show-leak-kinds=all",
        "--track-origins=yes",
        "--verbose",
        "--log-file=valgrind-out.txt",
    ]
if args.asan:
    use_sanitizer = "address"
if args.pybind:
    os.environ["SEEMPS_PYBIND"] = "ON"
if args.install_hooks:
    install_hooks()

if args.clean:
    clean()
if args.build:
    raise Exception("Build failed")
if args.install:
    if not install():
        raise Exception("Install failed")
if args.compare_test:
    compare_tests_with_different_cores(filter=args.filter)
if args.docs:
    build_documentation()
if args.check:
    if not check(args.verbose, filter=args.filter):
        raise Exception("Tests failed")
else:
    if args.coverage:
        if not run_tests(args.verbose, args.coverage, filter=args.filter):
            raise Exception("Unit tests failed")
        if not code_coverage_report():
            raise Exception("Unable to produce coverage report")
    elif args.tests:
        if not run_tests(args.verbose, cProfile=args.cProfile, filter=args.filter):
            raise Exception("Unit tests failed")
    if args.pyright:
        if not basedpyright():
            raise Exception("Basedpyright failed")
    if args.mypy:
        if not mypy():
            raise Exception("mypy failed")
    if args.ruff:
        if not ruff():
            raise Exception("ruff failed")
    if args.pydocstyle:
        if not pydocstyle():
            raise Exception("pydocstyle failed")
    if args.darglint:
        if not darglint():
            raise Exception("darglint failed")
