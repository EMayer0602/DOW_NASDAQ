"""Compatibility wrapper for live trading cycles.

Delegates to paper_trader so simulation, paper, and live modes
share the exact same execution engine.
"""
from __future__ import annotations

import sys
from typing import Sequence

import paper_trader as pt


def main(argv: Sequence[str] | None = None) -> None:
    args = list(argv if argv is not None else sys.argv[1:])
    if "--place-orders" not in args:
        args.insert(0, "--place-orders")
    pt.run_cli(args)


if __name__ == "__main__":
    main()
