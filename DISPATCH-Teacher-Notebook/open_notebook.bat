@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv_notebook\Scripts\python.exe" (
  echo ERROR: .venv_notebook is missing. Create it first.
  exit /b 1
)
echo Opening Jupyter Notebook in:
echo %CD%
".venv_notebook\Scripts\python.exe" -m notebook "dispatch_defense_experiment.ipynb"
endlocal
