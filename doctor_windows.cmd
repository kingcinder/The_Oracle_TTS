@echo off
setlocal EnableExtensions

cd /d "%~dp0"
set "REPO_ROOT=%CD%"
set "VENV_PYTHON=%REPO_ROOT%\.venv\Scripts\python.exe"

if not exist "%VENV_PYTHON%" (
  echo [FAIL] Missing "%VENV_PYTHON%". Run bootstrap_windows.cmd first.
  exit /b 1
)

echo [INFO] Verifying supported Python runtime
"%VENV_PYTHON%" -c "import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] < (3, 13) else 1)"
if errorlevel 1 (
  echo [FAIL] The repo venv must use Python 3.11 or 3.12.
  exit /b 1
)

echo [INFO] Running managed doctor in smoke-friendly mode
"%VENV_PYTHON%" "%REPO_ROOT%\scripts\manage_install.py" doctor --skip-model-init --ci %*
if errorlevel 1 exit /b 1

echo [INFO] Running Windows smoke tests
"%VENV_PYTHON%" -m pytest -q tests\test_smoke_render.py tests\test_repo_portability.py tests\test_windows_cmd_surface.py
exit /b %ERRORLEVEL%
