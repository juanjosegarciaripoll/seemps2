@echo off
if "x%1"=="xclean" goto :clean
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
