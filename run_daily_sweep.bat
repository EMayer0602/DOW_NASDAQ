@echo off
REM Daily Bar Parameter Sweep for DOW/NASDAQ Stocks
REM Runs 2-year backtest, excludes last month for testing

echo ============================================
echo DAILY BAR PARAMETER SWEEP
echo ============================================

REM Run sweep on default trading symbols
python stock_sweep_daily.py

REM For quick sweep (fewer parameters):
REM python stock_sweep_daily.py --quick

REM For all DOW 30 stocks:
REM python stock_sweep_daily.py --dow

REM For all stocks:
REM python stock_sweep_daily.py --all

echo.
echo Sweep complete! Check report_stocks/best_params_daily.csv
pause
