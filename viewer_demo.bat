@echo off
rem The new common image viewer on a test project, outside the client (until it is wired into the views).
cd /d "%~dp0"
"<user folder>\sandbox\ComfyUI\.venv\Scripts\python.exe" tools\imageview_demo.py --lang fr %*
if errorlevel 1 pause
