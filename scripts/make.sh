#!/bin/bash
set -x

function clean {
	for i in '*.egg-info' '__pycache__' '*.pyd'; do
		find . -name $i | xargs rm -rf;
	done
	rm -rf build dist
}

function check {
	LD_PRELOAD="$THE_PRELOAD" python3 -m unittest discover -f && \
	mypy src/seemps && \
	ruff check src
}

function asan_dll {
	ldconfig -p |grep asan | sed 's/^\(.*=> \)\(.*\)$/\2/g'
}


function build {
	if [ -n "$PYTHONPATH" ]; then
		python3 setup.py build_ext -j 4 --inplace
	else
		python3 setup.py build -j 4
	fi
}

for option in $*; do
	case "$option" in
		*here) export PYTHONPATH=`pwd`/src;;
		*leak) export SANITIZE=leak;;
		*asan) export SANITIZE=address
			   export THE_PRELOAD=`asan_dll`
			   ;;
		*clean) clean;;
		*check) check || exit 1;;
		*install) pip install --upgrade . || exit 1;;
		*build) build || exit 1;;
		*) echo "Unknown option $optoin"; exit 1;;
	esac
done

exit 0

@echo off
if "x%1"=="xclean" goto :clean
if "x%1"=="xcheck" goto :check
if "x%1"=="xinstall" goto :install
echo Unrecognized option %1
goto :eof

:clean
for /r . %%i in (SeeMPS.egg-info __pycache__) do if exist "%%i" rmdir /S /Q "%%i"
for %%i in (build dist) do if exist %%i rmdir /S /Q "%%i"
for /r . %%i in (*.pyd) do if exist "%%i" del /S /Q "%%i"
goto :eof

:install
pip install --upgrade .

:check
mypy src/seemps
ruff check src
rem flake8 src/seemps --count --select=E9,F63,F7,F82 --show-source --statistics
rem flake8 src/seemps --count --exit-zero --max-complexity=10 --max-line-length=150 --ignore W503,E741,E203,C901 --statistics
