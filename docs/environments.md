# Writing Environment Files

Each MLIP is defined by a small Python file with [PEP 723](https://peps.python.org/pep-0723/) inline metadata specifying its dependencies and a `setup()` function that returns an ASE calculator.

## Basic Structure

```python
# /// script
# requires-python = ">=3.10"
# dependencies = ["mace-torch>=0.3.14", "ase>=3.22", "torch>=2.0,<2.10"]
# ///

def setup(model: str, device: str = "cuda"):
    from mace.calculators import mace_mp
    return mace_mp(model=model, device=device, default_dtype="float32")
```

## How It Works

1. **PEP 723 metadata**: The `# /// script` block defines Python version requirements and dependencies
2. **`setup()` function**: Called once when a worker starts; the returned calculator is reused for all calculations
3. **uv builds the environment**: Rootstock uses `uv` to create an isolated virtual environment from these dependencies

## Required Elements

### PEP 723 Metadata Block

```python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "mace-torch>=0.3.14",
#     "ase>=3.22",
#     "torch>=2.0,<2.10"
# ]
# ///
```

- `requires-python`: Minimum Python version
- `dependencies`: List of pip-installable packages with version constraints

### The `setup()` Function

```python
def setup(model: str, device: str = "cuda"):
    # Import the MLIP library
    from mace.calculators import mace_mp
    
    # Create and return an ASE calculator
    return mace_mp(model=model, device=device, default_dtype="float32")
```

The function receives:

- `model`: The checkpoint/model name passed by the user
- `device`: Either `"cuda"` or `"cpu"`

It must return an ASE-compatible calculator object.

## Examples

### MACE

```python
# /// script
# requires-python = ">=3.10"
# dependencies = ["mace-torch>=0.3.14", "ase>=3.22", "torch>=2.0,<2.10"]
# ///

def setup(model: str, device: str = "cuda"):
    from mace.calculators import mace_mp
    return mace_mp(model=model, device=device, default_dtype="float32")
```

### CHGNet

```python
# /// script
# requires-python = ">=3.10"
# dependencies = ["chgnet>=0.3.0", "ase>=3.22", "torch>=2.0"]
# ///

def setup(model: str = "pretrained", device: str = "cuda"):
    from chgnet.model import CHGNetCalculator
    return CHGNetCalculator(use_device=device)
```

### UMA (FAIRChem)

```python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "fairchem-core>=1.0.0",
#     "ase>=3.22",
#     "torch>=2.0"
# ]
# ///

def setup(model: str = "uma-s-1p1", device: str = "cuda"):
    from fairchem.core import OCPCalculator
    return OCPCalculator(checkpoint_path=model, device=device)
```

### TensorNet (MatGL)

```python
# /// script
# requires-python = ">=3.10"
# dependencies = ["matgl>=1.0.0", "ase>=3.22", "torch>=2.0"]
# ///

def setup(model: str = "TensorNet-MatPES-PBE-v2025.1-PES", device: str = "cuda"):
    import matgl
    from matgl.ext.ase import MatGLCalculator
    
    potential = matgl.load_model(model)
    return MatGLCalculator(potential, device=device)
```

## Best Practices

### Pin dependency versions

Use version constraints to ensure reproducible builds:

```python
# Good: pinned versions
# dependencies = ["mace-torch>=0.3.14,<0.4", "torch>=2.0,<2.10"]

# Avoid: unpinned versions
# dependencies = ["mace-torch", "torch"]
```

### Handle model loading errors gracefully

```python
def setup(model: str, device: str = "cuda"):
    from mace.calculators import mace_mp
    
    try:
        return mace_mp(model=model, device=device, default_dtype="float32")
    except Exception as e:
        raise RuntimeError(f"Failed to load MACE model '{model}': {e}")
```

### Document available checkpoints

Add a comment listing known-good checkpoints:

```python
# /// script
# requires-python = ">=3.10"
# dependencies = ["mace-torch>=0.3.14", "ase>=3.22", "torch>=2.0,<2.10"]
# ///
#
# Available checkpoints: small, medium, large

def setup(model: str, device: str = "cuda"):
    from mace.calculators import mace_mp
    return mace_mp(model=model, device=device, default_dtype="float32")
```

## Testing Your Environment

After creating an environment file, test it:

```bash
# Build the environment
rootstock install my_env.py --models default_checkpoint

# Check it was built successfully
rootstock status

# Test the calculator (if you have GPU access)
rootstock test --env my_env --model default_checkpoint
```
