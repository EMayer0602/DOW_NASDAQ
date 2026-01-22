"""Volatility indicators - compatibility shim for ta library."""
import numpy as np
import pandas as pd


class AverageTrueRange:
    """Average True Range indicator.

    Compatible with the ta library API.
    """

    def __init__(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 14,
        fillna: bool = False,
    ):
        self._high = high
        self._low = low
        self._close = close
        self._window = window
        self._fillna = fillna
        self._atr = self._calculate_atr()

    def _calculate_atr(self) -> pd.Series:
        """Calculate ATR using Wilder's smoothing method."""
        high = self._high.values
        low = self._low.values
        close = self._close.values

        # True Range components
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))

        # True Range is max of the three
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        tr[0] = tr1[0]  # First bar has no previous close

        # Wilder's smoothing (EMA with alpha = 1/window)
        atr = np.zeros_like(tr)
        atr[:self._window] = np.nan

        if len(tr) >= self._window:
            # First ATR is SMA
            atr[self._window - 1] = np.mean(tr[:self._window])

            # Subsequent ATRs use Wilder's smoothing
            alpha = 1.0 / self._window
            for i in range(self._window, len(tr)):
                atr[i] = (1 - alpha) * atr[i - 1] + alpha * tr[i]

        result = pd.Series(atr, index=self._high.index)

        if self._fillna:
            result = result.fillna(0)

        return result

    def average_true_range(self) -> pd.Series:
        """Return ATR values."""
        return self._atr
