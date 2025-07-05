#!/usr/bin/python3
import os
import re
import sys
import shutil
import subprocess
import argparse

incremental: bool = True
debug: bool = False
use_sanitizer: str = "no" if "SANITIZE" not in os.environ else os.environ["SANITIZE"]
ld_preload: str = ""
valgrind: list[str] = []
python: list[str] = ["python" if sys.platform == "win32" else "python3"]
uv_run: list[str] = []


def run(command: list[str], *args, **kwdargs) -> bool:
    print(f"Running command:\n{' '.join(command)}", flush=True)
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
    for dirname, _, _ in os.walk("."):
        if os.path.basename(dirname) in patterns:
            to_delete.append(dirname)
    for path in to_delete:
        if os.path.exists(path):
            print(f"Removing {path}")
            shutil.rmtree(path)


def delete_files(patterns: list[str], root: str = "."):
    regexp = [re.compile(p) for p in patterns]
    for root, _, files in os.walk("."):
        for f in files:
            if any(p.match(f) for p in regexp):
                path = root + "/" + f
                print(f"Removing {path}")
                os.remove(path)


def clean() -> None:
    for path in ["build", "dist"]:
        if os.path.exists(path):
            print(f"Removing {path}")
            shutil.rmtree(path)
    delete_directories(["seemps.egg-info", "__pycache__"])
    delete_files([r".*\.so", r".*\.pyd", r".*\.pyc"])


def run_tests(verbose=False) -> bool:
    env = os.environ.copy()
    if use_sanitizer != "no":
        env["LD_PRELOAD"] = asan_library() + " " + cpp_library()
        print(f"LD_PRELOAD={env['LD_PRELOAD']}")
    return run(
        valgrind
        + uv_run
        + python
        + ["-m", "unittest", "discover", "-fv" if verbose else "-f"]
    )


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


def check(verbose: bool = False) -> bool:
    return (
        run_tests()
        and mypy()
        and ruff()
        and basedpyright()
        and pydocstyle()
        and darglint()
    )


def build() -> bool:
    env = os.environ.copy()
    extra: list[str] = []
    if use_sanitizer != "no":
        env["SANITIZE"] = use_sanitizer
    if debug:
        env["SEEMPS2DEBUG"] = "ON"
    if incremental:
        extra = ["--no-build-isolation", "-ve"] + extra
    return run(["pip", "install"] + extra + ["."], env=env)


def install() -> bool:
    return run(["pip", "install", "."])


parser = argparse.ArgumentParser(prog="make", description="SeeMPS build driver")
parser.add_argument("--uv", action="store_true", help="Run jobs under 'uv'")
parser.add_argument("--debug", action="store_true", help="Build debug versions")
parser.add_argument("--here", action="store_true", help="Run jobs incrementally")
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
parser.add_argument("--check", action="store_true", help="Run tests")
parser.add_argument("--verbose", action="store_true", help="Verbose mode")
parser.add_argument("--tests", action="store_true", help="Run unit tests")
parser.add_argument("--pyright", action="store_true", help="Run pyright")
parser.add_argument("--ruff", action="store_true", help="Run ruff")
parser.add_argument("--mypy", action="store_true", help="Run mypy")
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

if args.clean:
    clean()
if args.build:
    raise Exception("Build failed")
if args.install:
    if not install():
        raise Exception("Install failed")
if args.check:
    if not check(args.verbose):
        raise Exception("Tests failed")
else:
    if args.tests:
        if not run_tests(args.verbose):
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
