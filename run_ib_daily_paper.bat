@echo off
REM Daily Bar Stock Trader - IB Paper Trading
REM Runs at market open and close with daily bars

echo ============================================
echo DAILY BAR TRADER - IB PAPER TRADING
echo Runs at market OPEN and CLOSE
echo Lower frequency = Lower commission costs!
echo ============================================

REM Make sure TWS/IB Gateway is running on port 7497

python stock_daily_trader.py --ib --loop

pause
