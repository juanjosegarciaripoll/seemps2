#!/usr/bin/python3
import os
import sys
import shutil
import subprocess

inplace: bool = True
use_sanitizer: str = "no"
ld_preload: str = ""
valgrind: list[str] = []
python: str = "python" if sys.platform == "win32" else "python3"


def run(command: list[str], *args, **kwdargs) -> bool:
    print(f"Running command:\n{' '.join(command)}")
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


def run_many(commands: list[list[str]], *args, chain: bool = True, **kwdargs) -> bool:
    ok = True
    for c in commands:
        if not run(c, *args, **kwdargs):
            ok = False
            if chain:
                return ok
    return ok


def run_output(command: str, *args, **kwdargs) -> str:
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


def clean() -> None:
    for path in ["build", "dist"]:
        if os.path.exists(path):
            print(f"Removing {path}")
            shutil.rmtree(path)
    delete_directories(["SeeMPS.egg-info", "__pycache__"])


def check() -> bool:
    env = os.environ.copy()
    if use_sanitizer != "no":
        env["LD_PRELOAD"] = asan_library()
    return run_many(
        [
            valgrind + [python, "-m", "unittest", "discover", "-f"],
            ["mypy", "src/seemps"],
            ["ruff", "check", "src"],
        ],
        env=env,
    )


def build() -> bool:
    env = os.environ.copy()
    if use_sanitizer != "no":
        env["SANITIZE"] = use_sanitizer
    if inplace:
        return run([python, "setup.py", "build_ext", "-j", "4", "--inplace"], env=env)
    else:
        return run([python, "setup.py", "build", "-j", "4"], env=env)


def install() -> bool:
    return run(["pip", "install", "."])


for option in sys.argv[1:]:
    match option:
        case "here":
            inplace = True
            os.environ["PYTHONPATH"] = os.getcwd() + "/src"
        case "leak":
            use_sanitizer = "leak"
        case "asan":
            use_sanitizer = "address"
        case "clean":
            clean()
        case "build":
            if not build():
                raise Exception("Build failed")
        case "install":
            if not install():
                raise Exception("Install failed")
        case "check":
            if not check():
                raise Exception("Tests failed")
        case _:
            raise Exception(f"Unknown option: {option}")
