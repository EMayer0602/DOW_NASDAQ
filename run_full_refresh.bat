@echo off
setlocal EnableDelayedExpansion

rem Toggle detailed signal logging by running: set DEBUG=1 && run_full_refresh.bat
set "DEBUG_FLAG="
if /I "%DEBUG%"=="1" set "DEBUG_FLAG=--debug-signals"
if not defined INDICATOR_FILTER set "INDICATOR_FILTER=jma"
set "INDICATOR_FLAG="
if defined INDICATOR_FILTER set "INDICATOR_FLAG=--indicators %INDICATOR_FILTER%"

cd /d "%~dp0"

set "LOCK_FILE=%~dp0refresh.lock"
if exist "%LOCK_FILE%" goto :lock_found
type nul > "%LOCK_FILE%"
goto :begin_refresh

:lock_found
echo [WARN] Another refresh or trading job is running (found %LOCK_FILE%).
exit /b 1

:begin_refresh
set "ARCHIVE_ROOT=report_archives"
if not exist "%ARCHIVE_ROOT%" mkdir "%ARCHIVE_ROOT%"
set "ARCHIVE_DIR=%ARCHIVE_ROOT%\refresh_%RANDOM%%RANDOM%%RANDOM%"

echo [%date% %time%] Activating Conda environment...
call "%USERPROFILE%\anaconda3\Scripts\activate.bat" base || goto :error

echo [%date% %time%] Regenerating overall-best reports via Supertrend_5Min.py...
python Supertrend_5Min.py || goto :error

echo [%date% %time%] Running paper_trader full refresh snapshot...
python paper_trader.py --simulate --refresh-params --reset-state --clear-outputs %INDICATOR_FLAG% %DEBUG_FLAG% || goto :error

echo [%date% %time%] Archiving outputs to %ARCHIVE_DIR% ...
mkdir "%ARCHIVE_DIR%" >nul 2>&1
for %%F in (
	paper_trading_log.csv
	paper_trading_simulation_log.csv
	paper_trading_simulation_log.json
	paper_trading_open_positions.csv
	paper_trading_open_positions.json
	paper_trading_simulation_summary.html
	paper_trading_simulation_summary.json
	paper_trading_state.json
) do call :copy_if_exists "%%F"

echo [%date% %time%] Refresh complete. Outputs archived in %ARCHIVE_DIR%.
del "%LOCK_FILE%" >nul 2>&1
exit /b 0

:copy_if_exists
if exist "%~1" copy "%~1" "%ARCHIVE_DIR%" >nul
goto :eof

:error
echo [ERROR] Step failed with exit code %errorlevel%.
del "%LOCK_FILE%" >nul 2>&1
exit /b %errorlevel%
