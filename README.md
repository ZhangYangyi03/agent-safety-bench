# agent-safety-bench

**Multi-step agent safety compliance benchmarking**

Find the critical depth at which your LLM agent fails to maintain safety policies across sequential interactions.

```bash
pip install agent-safety-bench
agent-safety-bench run --model qwen3-8b --scenario market_report
```

## Why

Current LLM safety evaluation is **single-turn**: you ask one question, check if the model refuses. But real agents execute **multi-step chains** — research, code, deploy, process data. The GGT (Guardrail Gap Theorem) proves that single-step verifiers cannot guarantee multi-step safety: **compositional safety is NP-hard**.

This tool operationalizes that theorem. It measures **D\*** — the critical depth at which the agent's compliance rate drops below a threshold — across 3 built-in scenarios and any custom scenario you define.

## Quick Start

```bash
# Install
pip install agent-safety-bench

# List available scenarios
agent-safety-bench list-scenarios

# Run a benchmark
agent-safety-bench run \
    --model qwen3-8b \
    --scenario market_report \
    --depths 1 3 5 7 9 \
    --trials 3 \
    --output results.json

# Analyze existing results
agent-safety-bench analyze --input results.json
```

## Built-in Scenarios

| Scenario | Description | Policies Tested |
|----------|-------------|-----------------|
| `market_report` | Multi-step research with stale/unverified data | Data freshness, source verification |
| `deploy_audit` | Deployment pipeline with security shortcuts | Test discipline, security review |
| `privacy_compliance` | Data processing with privacy violations | PII protection, third-party sharing |

## API

```python
from agent_safety_bench import SafetyBench
from agent_safety_bench.scenarios import get_scenario

scenario = get_scenario("market_report")
bench = SafetyBench(model="qwen3-8b")

results = bench.run_benchmark(
    steps=scenario["build_chain"](max_depth=8),
    policies=scenario["policies"],
    system_prompt=scenario["system_prompt"],
    depths=[1, 2, 3, 4, 5, 6, 7, 8],
    trials_per_depth=3,
    verbose=True,
)

from agent_safety_bench.analyze import find_phase_transition, print_summary
pt = find_phase_transition(results, model_name="qwen3-8b")
print(print_summary(pt))
```

## Custom Scenarios

```python
from agent_safety_bench import SafetyBench
from agent_safety_bench.core import ChainStep
from agent_safety_bench.scenarios import register_scenario

def my_chain(depth):
    steps = [
        ChainStep(text="Step 1: ...", policy_violation=""),
        ChainStep(text="Try to violate policy X", policy_violation="policy_x"),
    ]
    return steps[:depth+1]

register_scenario("my_test", {
    "name": "My Custom Test",
    "description": "Tests policy X",
    "system_prompt": "You must follow policy X...",
    "policies": ["Policy X"],
    "build_chain": my_chain,
})
```

## Theory

This tool is based on the **Guardrail Gap Theorem** (GGT): the problem of deciding whether a sequence of individually-valid tool calls can collectively violate a global safety policy (SAFE-EXEC) is NP-complete.

The empirical finding: as chain depth D grows, every tested model shows a **phase transition** at a critical depth D\* where compliance drops from near-perfect to near-zero. This transition is sharp (width <= 2 depths) and scale-dependent (larger models have higher D\*).

## License

MIT