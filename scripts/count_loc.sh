:: This batch file reads cpp and h files recursively and counts number of lines
echo off
setlocal
echo "--------------------------------"
echo "Reading number of code lines"
echo "--------------------------------"
echo off
set /a totalNumLines = 0
SETLOCAL ENABLEDELAYEDEXPANSION
for /r %%f in (*.cc *.cpp *.h ) do (
for /f %%C in ('Find /V /C "" ^< %%f') do set Count=%%C
echo !Count! lines in %%f
set /a totalNumLines+=!Count!
)
echo Total Number of Code Lines: %totalNumLines%
