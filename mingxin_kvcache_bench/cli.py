# -*- coding: utf-8 -*-
"""kvcache-bench CLI: query signed KV-cache tiering benchmark results and estimate ROI.

Honesty rules baked in:
- `results` / `summary` print only measured numbers from signed reports (bundled JSON).
- `roi` labels every input band as measured vs estimated and prints mid-scenario
  results as estimates, never commitments.
"""
from __future__ import annotations

import argparse
import json
import sys
from importlib import resources


def _load_results() -> dict:
    with resources.files("mingxin_kvcache_bench").joinpath("data/kvcache_bench_results.json").open(
        encoding="utf-8"
    ) as f:
        return json.load(f)


def cmd_results(_args: argparse.Namespace) -> int:
    print(json.dumps(_load_results(), ensure_ascii=False, indent=2))
    return 0


def cmd_summary(_args: argparse.Namespace) -> int:
    data = _load_results()
    print(f"mingxin-kvcache-bench results v{data['version']} ({data['export_date']})")
    print(f"Platform: {data['platform']['gpu']}; DUT: {data['platform']['device_under_test']}")
    print()
    for exp in data["experiments"]:
        print(f"[{exp['report']}] {exp['title']}")
        for k, v in exp.get("headline", {}).items():
            print(f"    {k}: {v}")
    print()
    print("Signed report PDFs: https://mingxinstorage.xyz/en/evidence")
    return 0


def cmd_roi(args: argparse.Namespace) -> int:
    # Faithful port of accel_value.py account #1 (same constants as site lib/roi.ts)
    ARRAY_CNY, USD_CNY, CARD_USD = 371_200, 7.2, 12_000
    gpus = args.nodes * args.gpus_per_node
    array_usd = ARRAY_CNY / USD_CNY * args.arrays
    print(f"Cluster: {args.nodes} nodes x {args.gpus_per_node} GPUs = {gpus} GPUs; "
          f"{args.arrays} x FX100 (ref. CNY {ARRAY_CNY:,} each)")
    print("Scenario bands: uplift 29-40% is MEASURED (R2/R3); "
          "cold-recovery share 10-50% is ESTIMATED (to be backfilled by pilot).")
    print()
    scenarios = [("conservative", 0.29, 0.10), ("mid", 0.35, 0.30), ("upper", 0.40, 0.50)]
    if args.uplift is not None or args.cold_share is not None:
        scenarios.append(("custom", args.uplift or 0.35, args.cold_share or 0.30))
    for tag, uplift, cold in scenarios:
        gpu_equiv = uplift * cold * gpus
        value = gpu_equiv * CARD_USD
        roi = value / array_usd
        print(f"  {tag:>12}: uplift {uplift:.0%} x cold-share {cold:.0%} "
              f"-> {gpu_equiv:.1f} GPU-equivalents freed, ~${value:,.0f}, ROI {roi:.2f}x")
    print()
    print("Mid-scenario estimates, not commitments. Interactive: https://mingxinstorage.xyz/en/roi")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="kvcache-bench",
                                description="Signed KV-cache tiering benchmark results + ROI estimator")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("results", help="dump the full signed results JSON").set_defaults(fn=cmd_results)
    sub.add_parser("summary", help="headline numbers per experiment").set_defaults(fn=cmd_summary)
    roi = sub.add_parser("roi", help="ROI estimate for a GPU cluster")
    roi.add_argument("--nodes", type=int, default=16)
    roi.add_argument("--gpus-per-node", type=int, default=8)
    roi.add_argument("--arrays", type=int, default=8)
    roi.add_argument("--uplift", type=float, default=None, help="custom uplift (0.29-0.40 measured band)")
    roi.add_argument("--cold-share", type=float, default=None, help="custom cold share (0.1-0.5 estimated band)")
    roi.set_defaults(fn=cmd_roi)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
