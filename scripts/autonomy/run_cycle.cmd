@echo off
setlocal
for %%I in ("%~dp0..\..") do set "FOCUS_ROOT=%%~fI"
set "FOCUS_PYTHON=%FOCUS_ROOT%\.venv\Scripts\python.exe"

if not exist "%FOCUS_PYTHON%" (
  echo Project Python is missing: %FOCUS_PYTHON% 1>&2
  exit /b 2
)

pushd "%FOCUS_ROOT%"
if "%~1"=="" (
  "%FOCUS_PYTHON%" scripts\autonomy\run_codex_loop.py --mode preflight --hypothesis H001-forward-influence-routing
) else (
  "%FOCUS_PYTHON%" scripts\autonomy\run_codex_loop.py %*
)
set "FOCUS_EXIT=%ERRORLEVEL%"
popd
exit /b %FOCUS_EXIT%
