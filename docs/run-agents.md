# Run — with agents

"Agent" means two different things here, and Rootstock works with both. A **coding agent** writes Rootstock code for you, and Rootstock ships a [skill](https://github.com/Garden-AI/rootstock/blob/main/skill/skill.md) that teaches one to discover what's deployed on a cluster and call `RootstockCalculator` correctly. An **autonomous agent**, built with a framework like [Academy](https://github.com/academy-agents/academy), holds a `RootstockCalculator` as its own state and serves MLIP calls to whatever else is running. The skill comes first, Academy after it.

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

## Academy

[Academy](https://github.com/academy-agents/academy) is Globus Labs' middleware for stateful agents on distributed and federated infrastructure. An agent is a class with `@action` methods peers can call and `@loop` methods that run on their own; a manager places agents on remote resources through executors, and they talk through an exchange.

Rootstock fits Academy more cleanly than it fits most tools, and the reason is lifecycle. Academy agents persist, and they have real startup and shutdown hooks. Build the calculator in `agent_on_startup()`, keep it as agent state, close it in `agent_on_shutdown()`. The worker spins up once when the agent lands on the node, stays hot across every action it serves, and dies when the agent does. This is the same shape Academy's own [HPC guide](https://docs.academy-agents.org/stable/guides/hpc/) uses for a Parsl DFK.

```bash
pip install academy-py
```

An agent that serves MLIP evaluations:

```python
from academy.agent import Agent, action
from ase import Atoms

from rootstock import RootstockCalculator


class MLIPAgent(Agent):
    def __init__(self, checkpoint: str, cluster: str, device: str = "cuda") -> None:
        super().__init__()
        self.checkpoint = checkpoint
        self.cluster = cluster
        self.device = device
        self.calc: RootstockCalculator | None = None

    async def agent_on_startup(self) -> None:
        # One worker, started once the agent is running on the compute node.
        self.calc = RootstockCalculator(
            checkpoint=self.checkpoint,
            cluster=self.cluster,
            device=self.device,
        )

    async def agent_on_shutdown(self) -> None:
        if self.calc is not None:
            self.calc.close()
            self.calc = None

    @action
    async def potential_energy(self, atoms: Atoms) -> float:
        atoms.calc = self.calc
        return atoms.get_potential_energy()

    @action
    async def forces(self, atoms: Atoms) -> list[list[float]]:
        atoms.calc = self.calc
        return atoms.get_forces().tolist()
```

Launch it and call it by handle. `kwargs` are forwarded to the agent's `__init__` on the worker:

```python
from concurrent.futures import ThreadPoolExecutor

from academy.exchange import LocalExchangeFactory
from academy.manager import Manager

async with await Manager.from_exchange_factory(
    factory=LocalExchangeFactory(),
    executors=ThreadPoolExecutor(),
) as manager:
    mlip = await manager.launch(
        MLIPAgent,
        kwargs={"checkpoint": "mace-mp-0-medium", "cluster": "sophia"},
    )

    energy = await mlip.potential_energy(atoms)

    await manager.shutdown(mlip, blocking=True)
```

### Onto a cluster

The local exchange and thread pool above keep everything in-process, which is fine for development but misses the point. To put the agent on a compute node, swap in a Globus Compute executor and the hosted exchange. Rootstock and Academy are both Globus Labs projects, so this is the well-trodden path:

```python
from academy.exchange.cloud.client import HttpExchangeFactory
from academy.manager import Manager
from globus_compute_sdk import Executor as GCExecutor

async with await Manager.from_exchange_factory(
    factory=HttpExchangeFactory(
        "https://exchange.academy-agents.org",
        auth_method="globus",
    ),
    executors=GCExecutor(endpoint_id),
) as manager:
    mlip = await manager.launch(
        MLIPAgent,
        kwargs={"checkpoint": "mace-mp-0-medium", "cluster": "sophia"},
    )
```

The agent now runs where the GPU and the Rootstock install are, and the Rootstock env resolves locally on that node. See Academy's [Building HPC Agents](https://docs.academy-agents.org/stable/guides/hpc/) guide for endpoint setup.

**What the agent's environment needs.** Rootstock, ASE, Academy, and whatever the agent itself does. Not `torch`, not `mace-torch`, not any model dependency. Those live in the isolated env on the other side of the i-PI socket, which is the whole reason to reach for Rootstock here rather than importing the model into the agent.

Pass `cluster=` for a registered cluster, or `root="/path/to/rootstock"` for a local install (not both). `device` defaults to `cpu` in `RootstockCalculator`; set `cuda` on GPU nodes. The target environment must already be built with `rootstock install`.