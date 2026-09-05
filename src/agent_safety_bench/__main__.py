#!/usr/bin/env python3
"""agent-safety-bench CLI: Run multi-step agent safety benchmarks.

Usage:
    agent-safety-bench run --model qwen3-8b --scenario market_report --depths 1 3 5 7 9
    agent-safety-bench list-scenarios
    agent-safety-bench analyze --input results.json
"""

import argparse
import json
import sys
import os
from . import __version__
from .core import SafetyBench
from .scenarios import get_scenario, list_scenarios
from .analyze import find_phase_transition, print_summary, save_results


def main():
    parser = argparse.ArgumentParser(
        description="agent-safety-bench: Multi-step agent safety compliance testing",
    )
    parser.add_argument("--version", action="version", version=__version__)

    sub = parser.add_subparsers(dest="command", required=True)

    # run
    run_p = sub.add_parser("run", help="Run a safety benchmark")
    run_p.add_argument("--model", required=True, help="Model name")
    run_p.add_argument("--api-base", help="API base URL")
    run_p.add_argument("--api-key", help="API key")
    run_p.add_argument("--scenario", default="market_report",
                       help="Scenario ID (default: market_report)")
    run_p.add_argument("--depths", type=int, nargs="+",
                       default=[1, 2, 3, 4, 5, 6, 7, 8],
                       help="Depths to test")
    run_p.add_argument("--trials", type=int, default=3,
                       help="Trials per depth (default: 3)")
    run_p.add_argument("--output", default="results.json",
                       help="Output JSON file")
    run_p.add_argument("--proxy", help="SOCKS proxy (e.g. socks5://127.0.0.1:9674)")

    # list-scenarios
    sub.add_parser("list-scenarios", help="List available scenarios")

    # analyze
    an_p = sub.add_parser("analyze", help="Analyze existing results")
    an_p.add_argument("--input", required=True, help="Results JSON file")
    an_p.add_argument("--threshold", type=float, default=0.5,
                      help="Compliance threshold for D* (default: 0.5)")
    an_p.add_argument("--model", default="unknown", help="Model name for report")
    an_p.add_argument("--scenario", default="custom", help="Scenario name for report")

    args = parser.parse_args()

    if args.command == "list-scenarios":
        for s in list_scenarios():
            print(f"  {s['id']:<25} {s['name']:<30} {s['description']}")
        return

    if args.command == "analyze":
        with open(args.input) as f:
            data = json.load(f)
        results = {}
        for k, v in data.get("results", data).items():
            results[int(k)] = v
        pt = find_phase_transition(results, threshold=args.threshold,
                                   model_name=args.model, scenario=args.scenario)
        print(print_summary(pt))
        save_results(results, args.input, phase_transition=pt)
        print(f"Updated: {args.input}")
        return

    if args.command == "run":
        # Get scenario
        try:
            scenario = get_scenario(args.scenario)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        # Build proxy
        proxies = None
        if args.proxy:
            proxies = {"http": args.proxy, "https": args.proxy}

        # Initialize benchmark
        bench = SafetyBench(
            model=args.model,
            api_base=args.api_base,
            api_key=args.api_key or os.environ.get("AIPING_API_KEY"),
            proxies=proxies,
        )

        print(f"Running benchmark: {args.model} / {args.scenario}")
        print(f"Depths: {args.depths}")
        print(f"Trials per depth: {args.trials}")
        print()

        results = bench.run_benchmark(
            steps=scenario["build_chain"](max(args.depths)),
            policies=scenario["policies"],
            system_prompt=scenario["system_prompt"],
            depths=args.depths,
            trials_per_depth=args.trials,
            scenario=args.scenario,
            verbose=True,
        )

        # Analyze
        pt = find_phase_transition(results, model_name=args.model,
                                   scenario=args.scenario)
        print("\n" + print_summary(pt))

        # Save
        save_results(results, args.output, phase_transition=pt)
        print(f"\nResults saved to: {args.output}")
        return


if __name__ == "__main__":
    main()