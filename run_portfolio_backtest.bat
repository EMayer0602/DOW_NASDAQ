@echo off
REM Multi-Symbol Portfolio Backtest
REM Tests the strategy with best parameters on out-of-sample data

echo ============================================
echo PORTFOLIO BACKTEST - DAILY BARS
echo ============================================

python stock_portfolio_backtest.py

echo.
echo Backtest complete! Check report_stocks/portfolio_backtest_trades.csv
pause
