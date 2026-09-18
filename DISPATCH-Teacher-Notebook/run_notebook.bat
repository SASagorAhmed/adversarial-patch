@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv_notebook\Scripts\python.exe" (
  echo ERROR: .venv_notebook is missing.
  exit /b 1
)
echo Executing dispatch_defense_experiment.ipynb ...
echo Working directory: %CD%
".venv_notebook\Scripts\python.exe" -m jupyter nbconvert --to notebook --execute "dispatch_defense_experiment.ipynb" --inplace --ExecutePreprocessor.timeout=7200 --ExecutePreprocessor.kernel_name=dispatch-teacher
if errorlevel 1 (
  echo FAILURE: notebook execution failed.
  exit /b 1
)
echo SUCCESS: notebook executed and saved with outputs.
endlocal
