@echo off
REM Stock Paper Trader - Simulation Mode (yfinance)
REM Runs without IB connection for testing

echo ============================================
echo STOCK PAPER TRADER - SIMULATION MODE
echo ============================================

python stock_paper_trader.py --symbols AAPL MSFT NVDA AMZN GOOGL META TSLA JPM V HD

pause
