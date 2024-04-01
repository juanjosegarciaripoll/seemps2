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
