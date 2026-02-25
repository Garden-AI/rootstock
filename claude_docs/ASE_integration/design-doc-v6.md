# Rootstock: Adding UMA and TensorNet Environments

## Overview

This document specifies how to add UMA (from FAIRChem) and TensorNet (from MatGL) environments to Rootstock. These are additional MLIP calculators requested by collaborators at Princeton/Della.

Target deployment: Princeton Della cluster, along the existing Rootstock installation.

---

## 1. Model Specifications

### UMA (Universal Atomistic Model)

**Source**: FAIRChem / Meta  
**Package**: `fairchem-core>=2.0.0`  
**Default checkpoint**: `uma-s-1p1`

UMA is Meta's universal atomistic model, accessed via the FAIRChem library. The model weights are downloaded automatically from HuggingFace via `pretrained_mlip.get_predict_unit()`.

**Calculator creation pattern**:
```python
from fairchem.core import FAIRChemCalculator, pretrained_mlip

predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device="cuda")
calc = FAIRChemCalculator(predictor, task_name="omat")
```

### TensorNet

**Source**: Materials Virtual Lab (MatGL)  
**Package**: `matgl` (installed from git for latest version)  
**Default checkpoint**: `TensorNet-MatPES-PBE-v2025.1-PES`

TensorNet is accessed via the MatGL library. Models are downloaded automatically from the MatGL model hub.

**Calculator creation pattern**:
```python
import matgl
from matgl.ext.ase import PESCalculator

model = matgl.load_model("TensorNet-MatPES-PBE-v2025.1-PES")
calc = PESCalculator(potential=model)
```

---

## 2. Environment Files

### `uma_env.py`

```python
# /// script
# requires-python = ">=3.10,<3.11"
# dependencies = [
#     "torch>=2.4.0",
#     "fairchem-core>=2.0.0",
#     "ase>=3.22",
#     "torch-geometric",
# ]
#
# [tool.uv]
# find-links = ["https://data.pyg.org/whl/torch-2.4.0+cu121.html"]
# ///
"""
UMA (Universal Atomistic Model) environment for Rootstock.

This environment provides access to Meta's UMA foundation model
via the FAIRChem library.

Models:
    - "uma-s-1p1": UMA small model (default)
    - Other UMA variants as released by FAIRChem
"""


def setup(model: str = "uma-s-1p1", device: str = "cuda"):
    """
    Load a UMA calculator.

    Args:
        model: Model identifier (e.g., "uma-s-1p1"). Passed directly to
               pretrained_mlip.get_predict_unit().
        device: PyTorch device string (e.g., "cuda", "cuda:0", "cpu")

    Returns:
        ASE-compatible calculator
    """
    from fairchem.core import FAIRChemCalculator, pretrained_mlip

    predictor = pretrained_mlip.get_predict_unit(model, device=device)
    return FAIRChemCalculator(predictor, task_name="omat")
```

### `tensornet_env.py`

```python
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch>=2.4.0",
#     "ase>=3.22",
#     "matgl @ git+https://github.com/materialsvirtuallab/matgl.git",
#     "torch-geometric",
#     "torch-scatter",
#     "torch-sparse", 
#     "torch-cluster",
#     "torch-spline-conv",
# ]
#
# [tool.uv]
# find-links = ["https://data.pyg.org/whl/torch-2.4.0+cu121.html"]
# ///
"""
TensorNet environment for Rootstock.

This environment provides access to TensorNet models via the MatGL library
from the Materials Virtual Lab.

Models:
    - "TensorNet-MatPES-PBE-v2025.1-PES": MatPES PBE functional (default)
    - Other MatGL models as available
"""


def setup(model: str = "TensorNet-MatPES-PBE-v2025.1-PES", device: str = "cuda"):
    """
    Load a TensorNet/MatGL calculator.

    Args:
        model: Model identifier (e.g., "TensorNet-MatPES-PBE-v2025.1-PES").
               Passed directly to matgl.load_model().
        device: PyTorch device string (currently MatGL handles device internally)

    Returns:
        ASE-compatible calculator
    """
    import matgl
    from matgl.ext.ase import PESCalculator

    pot = matgl.load_model(model)
    return PESCalculator(potential=pot)
```

---

## 3. Implementation Notes

### PyTorch Geometric Dependencies

Both UMA and TensorNet depend on `torch-geometric` and its extension packages (`torch-scatter`, `torch-sparse`, etc.). These packages require specific builds matching the PyTorch + CUDA version.

**Solution**: Each environment file specifies its `find-links` in a `[tool.uv]` section (see environment files above). The build system reads this generically.

**Note on Della**: CUDA 12.8 is available on Della. The cu121 wheels are forward compatible within CUDA 12.x.

### MatGL Installation

MatGL is installed from git to get the latest version with TensorNet support:
```
matgl @ git+https://github.com/materialsvirtuallab/matgl.git
```

This should work with uv's git dependency handling.

### UMA / FAIRChem

FAIRChem 2.0+ has a cleaner API than 1.x. The `pretrained_mlip` module handles model downloading automatically to `~/.cache/fairchem/` (or `XDG_CACHE_HOME/fairchem/`).

Rootstock already sets `XDG_CACHE_HOME` to `{root}/cache/`, so model weights should be cached correctly.

---

## 4. API Change: Separate `model` and `checkpoint` Parameters

The current API uses a combined model string (`model="mace-medium"`) that gets parsed into environment name and model argument. This is ambiguous for models like UMA where the checkpoint name contains hyphens (`uma-s-1p1`).

**Solution**: Split into two explicit parameters.

### New API

```python
# New: explicit parameters
with RootstockCalculator(
    cluster="della",
    model="mace",           # Maps to environment (mace_env)
    checkpoint="medium",    # Passed to setup() as model arg
    device="cuda",
) as calc:
    ...

# UMA example - no ambiguity
with RootstockCalculator(
    cluster="della",
    model="uma",
    checkpoint="uma-s-1p1",  # Full checkpoint name, passed as-is
    device="cuda",
) as calc:
    ...

# TensorNet example
with RootstockCalculator(
    cluster="della",
    model="tensornet",
    checkpoint="TensorNet-MatPES-PBE-v2025.1-PES",
    device="cuda",
) as calc:
    ...

# Defaults still work - checkpoint can be omitted for default models
with RootstockCalculator(
    cluster="della",
    model="uma",  # Uses default checkpoint defined in uma_env.py
    device="cuda",
) as calc:
    ...
```

### Code Changes

**`calculator.py`**:
```python
def __init__(
    self,
    model: str,                      # Required: environment family
    checkpoint: str | None = None,   # Optional: specific checkpoint
    cluster: str | None = None,
    root: str | Path | None = None,
    device: str = "cuda",
    ...
):
    self.env_name = f"{model}_env"
    self.model_arg = checkpoint or ""  # Empty string uses environment's default
    ...
```

### Update `clusters.py`

```python
# Add new environments to the known list (used for validation)
KNOWN_ENVIRONMENTS = ["mace", "chgnet", "orb", "alignn", "uma", "tensornet"]
```

Note: `parse_model_string()` can be removed or kept for other uses, but is no longer used by `RootstockCalculator`.

---

## 5. Build Process Updates

### `--find-links` via `[tool.uv]` in PEP 723

Environment files that need custom package indices can specify them in a `[tool.uv]` section. The build system reads this generically—no special casing per environment.

**Example in `tensornet_env.py`**:
```python
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch>=2.4.0",
#     "ase>=3.22",
#     "matgl @ git+https://github.com/materialsvirtuallab/matgl.git",
#     "torch-geometric",
#     "torch-scatter",
#     "torch-sparse", 
#     "torch-cluster",
#     "torch-spline-conv",
# ]
#
# [tool.uv]
# find-links = ["https://data.pyg.org/whl/torch-2.4.0+cu121.html"]
# ///
```

**Generic handling in `cmd_build()`**:
```python
metadata = parse_pep723_metadata(content)
dependencies = metadata.get("dependencies", [])

# Extract uv-specific config (generic, works for any environment)
uv_config = metadata.get("tool", {}).get("uv", {})
find_links = uv_config.get("find-links", [])

# Build pip install command
cmd = ["uv", "pip", "install", "--python", str(env_python)]
for link in find_links:
    cmd.extend(["--find-links", link])
cmd.extend(dependencies)
```

This keeps the build system generic while making environment files fully self-contained.

### Git Dependencies

MatGL is installed from git:
```
matgl @ git+https://github.com/materialsvirtuallab/matgl.git
```

This should work with uv's git dependency handling. If issues arise, we can pin to a specific commit or tag.

---

## 6. Testing Plan

### Della Testing

1. Copy environment files to rootstock root on della
2. Build on a GPU node (interactive or batch job)
3. Run a smoke test for the new environments.

---

## 7. Documentation Updates

### README Updates

**New API section**:
```python
# Recommended: explicit model and checkpoint
with RootstockCalculator(
    cluster="della",
    model="mace",
    checkpoint="medium",
) as calc:
    atoms.calc = calc
    energy = atoms.get_potential_energy()

# Checkpoint defaults to environment's default if omitted
with RootstockCalculator(cluster="della", model="uma") as calc:
    ...  # Uses uma-s-1p1 by default
```

**Available Models table**:
| Model | Environment | Default Checkpoint | Other Checkpoints |
|-------|-------------|-------------------|-------------------|
| `mace` | mace_env | `medium` | `small`, `large` |
| `chgnet` | chgnet_env | (pretrained) | — |
| `uma` | uma_env | `uma-s-1p1` | — |
| `tensornet` | tensornet_env | `TensorNet-MatPES-PBE-v2025.1-PES` | Other MatGL models |

### CLAUDE.md Updates

Add to the API section:
```python
# v0.5 API
with RootstockCalculator(
    cluster="della",
    model="uma",              # Environment family
    checkpoint="uma-s-1p1",   # Specific checkpoint (optional, uses default if omitted)
    device="cuda",
) as calc:
    ...
```

---

## 8. Implementation Checklist

### API Changes
- [ ] Add `checkpoint` parameter to `RootstockCalculator.__init__`
- [ ] Remove `parse_model_string()` usage from calculator (no backward compat)
- [ ] Update `KNOWN_ENVIRONMENTS` in `clusters.py`

### Environment Files
- [ ] Create `environments/uma_env.py`
- [ ] Create `environments/tensornet_env.py`

### Build System
- [ ] Add generic `[tool.uv]` parsing to `cmd_build()` (extract `find-links`, etc.)
- [ ] Test git dependency handling for matgl

### Testing
- [ ] Test UMA calculations
- [ ] Test TensorNet calculations

### Documentation
- [ ] Update README with new API and models
- [ ] Update CLAUDE.md with new environment info

### Deployment
- [ ] Confer with human to test on della

---

## 9. Resolved Design Decisions

1. **Python version**: Use Python 3.10 for UMA (matches FAIRChem requirements) and Python 3.12 for TensorNet (matches MatGL). Environment isolation makes this safe.

2. **CUDA version**: Della has CUDA 12.8 available. The PyG wheels for cu121 are forward compatible with CUDA 12.8.

3. **API for model/checkpoint**: Use two separate parameters (`model` for environment family, `checkpoint` for specific weights). No backward compatibility with combined strings — clean slate before release.