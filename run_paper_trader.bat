@echo off
setlocal
if not defined INDICATOR_FILTER set "INDICATOR_FILTER=jma"
set "INDICATOR_FLAG="
if defined INDICATOR_FILTER set "INDICATOR_FLAG=--indicators %INDICATOR_FILTER%"
if not defined SIGNAL_INTERVAL set "SIGNAL_INTERVAL=15"
if not defined SPIKE_INTERVAL set "SPIKE_INTERVAL=5"
if not defined ATR_MULT set "ATR_MULT=2.5"
if not defined POLL_SECONDS set "POLL_SECONDS=30"
set "TESTNET_FLAG="
if not defined USE_TESTNET (
	set "TESTNET_FLAG=--testnet"
) else (
	set "LC_USE_TESTNET=%USE_TESTNET%"
	if /I "%LC_USE_TESTNET%"=="0" (
		set "TESTNET_FLAG="
	) else (
		if /I "%LC_USE_TESTNET%"=="false" (
			set "TESTNET_FLAG="
		) else (
			set "TESTNET_FLAG=--testnet"
		)
	)
)
cd /d "%~dp0"

set "LOCK_FILE=%~dp0refresh.lock"
if exist "%LOCK_FILE%" (
	echo [PaperTrader] Refresh in progress - found "%LOCK_FILE%" - skipping run.
	exit /b 0
)

call "%USERPROFILE%\anaconda3\Scripts\activate.bat" base
set "MONITOR_ARGS=--monitor --signal-interval %SIGNAL_INTERVAL% --spike-interval %SPIKE_INTERVAL% --atr-mult %ATR_MULT% --poll-seconds %POLL_SECONDS%"
python paper_trader.py %MONITOR_ARGS% %TESTNET_FLAG% %INDICATOR_FLAG%