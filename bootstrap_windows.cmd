@echo off
setlocal EnableExtensions

cd /d "%~dp0"
set "REPO_ROOT=%CD%"
set "VENV_PYTHON=%REPO_ROOT%\.venv\Scripts\python.exe"

call :resolve_base_python
if errorlevel 1 exit /b 1

if not exist "%VENV_PYTHON%" (
  echo [INFO] Creating project virtual environment with supported Python
  call %BASE_PYTHON% -m venv "%REPO_ROOT%\.venv"
  if errorlevel 1 (
    echo [FAIL] Failed to create .venv with a supported Python interpreter
    exit /b 1
  )
)

if not exist "%VENV_PYTHON%" (
  echo [FAIL] Missing venv interpreter at "%VENV_PYTHON%"
  exit /b 1
)

echo [INFO] Bootstrapping The Oracle through the repo venv
"%VENV_PYTHON%" "%REPO_ROOT%\scripts\manage_install.py" bootstrap --skip-doctor --include-dev %*
if errorlevel 1 exit /b 1

call "%REPO_ROOT%\doctor_windows.cmd"
exit /b %ERRORLEVEL%

:resolve_base_python
py -3.12 -c "import sys" >nul 2>&1
if not errorlevel 1 (
  set "BASE_PYTHON=py -3.12"
  exit /b 0
)

py -3.11 -c "import sys" >nul 2>&1
if not errorlevel 1 (
  set "BASE_PYTHON=py -3.11"
  exit /b 0
)

python -c "import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] < (3, 13) else 1)" >nul 2>&1
if not errorlevel 1 (
  set "BASE_PYTHON=python"
  exit /b 0
)

echo [FAIL] Python 3.11 or 3.12 is required. Install it and make either "py -3.12", "py -3.11", or "python" resolve to a supported version.
exit /b 1
