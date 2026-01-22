@echo off
REM Stock Paper Trader - IB Paper Trading Mode
REM Requires TWS or IB Gateway running with API enabled on port 7497

echo ============================================
echo STOCK PAPER TRADER - IB PAPER TRADING
echo ============================================
echo.
echo Prerequisites:
echo - TWS Paper Trading running
echo - API enabled in TWS settings
echo - Port 7497 (TWS) or 4002 (Gateway)
echo.

python stock_paper_trader.py --ib --symbols AAPL MSFT NVDA AMZN GOOGL META TSLA JPM V HD --loop --interval 3600

pause
