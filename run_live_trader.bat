@echo off
setlocal

rem ===== User-tunable settings =====
set "USE_TESTNET=1"
set "INDICATOR_FILTER=jma"
set "STAKE=1000"
set "SIGNAL_INTERVAL=15"
set "SPIKE_INTERVAL=5"
set "ATR_MULT=2.5"
set "PLACE_ORDERS=1"
set "MAX_OPEN_POSITIONS=50"
set "CLEAR_ALL=0"
set "REFRESH_LAST48=1"
set "FORCE_ENTRY="
set "SYMBOLS="
set "NOTIFY_SMS=0"
set "SMS_TO="
rem ===== End settings =====

rem Build flags from settings
set "TESTNET_FLAG="
if "%USE_TESTNET%"=="" set "USE_TESTNET=1"
if /I "%USE_TESTNET%"=="1" set "TESTNET_FLAG=--testnet"
if /I "%USE_TESTNET%"=="true" set "TESTNET_FLAG=--testnet"

set "INDICATOR_FLAG="
if not "%INDICATOR_FILTER%"=="" set "INDICATOR_FLAG=--indicators %INDICATOR_FILTER%"

set "STAKE_FLAG="
if not "%STAKE%"=="" set "STAKE_FLAG=--stake %STAKE%"

set "SIGNAL_FLAG="
if not "%SIGNAL_INTERVAL%"=="" set "SIGNAL_FLAG=--signal-interval %SIGNAL_INTERVAL%"

set "SPIKE_FLAG="
if not "%SPIKE_INTERVAL%"=="" set "SPIKE_FLAG=--spike-interval %SPIKE_INTERVAL%"

set "ATR_FLAG="
if not "%ATR_MULT%"=="" set "ATR_FLAG=--atr-mult %ATR_MULT%"

set "PLACE_FLAG="
if /I "%PLACE_ORDERS%"=="1" set "PLACE_FLAG=--place-orders"
if /I "%PLACE_ORDERS%"=="true" set "PLACE_FLAG=--place-orders"

set "MAX_OPEN_FLAG="
if not "%MAX_OPEN_POSITIONS%"=="" set "MAX_OPEN_FLAG=--max-open-positions %MAX_OPEN_POSITIONS%"

set "CLEAR_FLAG="
if /I "%CLEAR_ALL%"=="1" set "CLEAR_FLAG=--clear-all"
if /I "%CLEAR_ALL%"=="true" set "CLEAR_FLAG=--clear-all"

set "FORCE_FLAG="
if not "%FORCE_ENTRY%"=="" set "FORCE_FLAG=--force-entry %FORCE_ENTRY%"

set "SYMBOLS_FLAG="
if not "%SYMBOLS%"=="" set "SYMBOLS_FLAG=--symbols %SYMBOLS%"

set "SMS_FLAG="
if /I "%NOTIFY_SMS%"=="1" set "SMS_FLAG=--notify-sms"
if /I "%NOTIFY_SMS%"=="true" set "SMS_FLAG=--notify-sms"

set "SMS_TO_FLAG="
if not "%SMS_TO%"=="" set "SMS_TO_FLAG=--sms-to %SMS_TO%"

cd /d "%~dp0"

set "LOCK_FILE=%~dp0refresh.lock"
if exist "%LOCK_FILE%" (
	echo [LiveTrader] Refresh in progress - found "%LOCK_FILE%" - skipping start.
	exit /b 0
)

call "%USERPROFILE%\anaconda3\Scripts\activate.bat" base
python Supertrend_5Min.py

if /I "%REFRESH_LAST48%"=="1" (
	powershell -NoLogo -Command "$ErrorActionPreference='Stop'; Set-Location '%~dp0'; $start=(Get-Date).AddDays(-7).ToString('s'); $end=(Get-Date).ToString('s'); $syms='%SYMBOLS%'; if ([string]::IsNullOrWhiteSpace($syms)) { $syms = 'BTC/EUR,ETH/EUR,XRP/EUR,LINK/EUR,LUNC/USDT,SOL/EUR,SUI/EUR,TNSR/USDC,ZEC/USDC' }; python paper_trader.py --simulate --start $start --end $end --symbols $syms --summary-html report_html/trading_summary.html --summary-json report_html/trading_summary.json --sim-log report_html/initial_trades.csv --sim-json report_html/initial_trades.json --open-log report_html/paper_trading_actual_trades.csv --open-json report_html/paper_trading_actual_trades.json"
	echo [LiveTrader] Merging initial trades into history...
	python merge_history.py
)

python paper_trader.py --monitor %TESTNET_FLAG% %PLACE_FLAG% %CLEAR_FLAG% %STAKE_FLAG% %SIGNAL_FLAG% %SPIKE_FLAG% %ATR_FLAG% %INDICATOR_FLAG% %SYMBOLS_FLAG% %SMS_FLAG% %SMS_TO_FLAG% %FORCE_FLAG% %MAX_OPEN_FLAG%
