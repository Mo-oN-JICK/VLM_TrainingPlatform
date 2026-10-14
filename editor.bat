@echo off
rem Kept for old habits. line.bat opens any line.yaml, project.yaml or folder.
rem With no argument it opens the 11-node example (project.yaml), as before.
rem ASCII on purpose. Keep CRLF line endings; .gitattributes pins them.
set "SPEC=%~1"
if "%SPEC%"=="" set "SPEC=solutions\vlm_open\projects\01_open\project.yaml"
call "%~dp0line.bat" "%SPEC%"
exit /b %errorlevel%
