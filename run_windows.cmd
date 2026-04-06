@echo off
setlocal EnableExtensions

cd /d "%~dp0"
set "REPO_ROOT=%CD%"
set "VENV_PYTHON=%REPO_ROOT%\.venv\Scripts\python.exe"

if not exist "%VENV_PYTHON%" (
  echo [FAIL] Missing "%VENV_PYTHON%". Run bootstrap_windows.cmd first.
  exit /b 1
)

"%VENV_PYTHON%" "%REPO_ROOT%\scripts\manage_install.py" run %*
exit /b %ERRORLEVEL%
