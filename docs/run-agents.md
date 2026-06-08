# Run — with agents

Beyond writing Python yourself, you can let a coding agent drive Rootstock for you. Rootstock ships an [agent skill](https://github.com/Garden-AI/rootstock/blob/main/skill/skill.md) that teaches an agent to discover what is deployed on a cluster and to call `RootstockCalculator` correctly.

## What the skill does

The skill covers two phases:

1. **Discover** what's deployed where, by reading the live dashboard manifest.
2. **Call** `RootstockCalculator` with a `(cluster, checkpoint)` pair, including how to forward `setup_kwargs` and avoid common pitfalls (login nodes without GPUs, cold-start model loads, stale checkpoints).

It triggers on requests like "compute an adsorption energy on Perlmutter", "relax this structure with MACE on Della", or "what MLIPs are available on this cluster" — even when the user doesn't say "Rootstock" by name.

## Using the skill

The full skill lives in the Rootstock repo at [`skill/skill.md`](https://github.com/Garden-AI/rootstock/blob/main/skill/skill.md). Point your agent at it so it can:

- Fetch the manifest and answer "is checkpoint X available on cluster Y?"
- Pick a model appropriate to the system, bounded by what's actually deployed and verified.
- Read each env's `setup()` signature for the kwargs it accepts.

Getting Python code onto a compute node (job submission, file staging, credentials) is out of scope for the skill itself — see the "Getting code onto a cluster" section there for the common paths.
