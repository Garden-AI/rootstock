# Rootstock v0.6: Parser Fix and HOME Directory Handling

## Overview

Two fixes discovered during Della deployment:

1. **PEP 723 parser bug** — Bare `#` lines (no trailing space) break metadata parsing
2. **Model cache location** — Libraries like FAIRChem and MatGL ignore `XDG_CACHE_HOME` and write to `~/`, so weights don't end up in the shared rootstock directory

---

## 1. PEP 723 Parser Fix

### Problem

The current regex in `rootstock/pep723.py`:

```python
PEP723_PATTERN = re.compile(
    r"^# /// script\s*\n((?:# .*\n)*?)# ///$",
    re.MULTILINE,
)
```

The pattern `# .*` requires a space after `#`. A bare `#` line breaks the match:

```python
# /// script
# dependencies = ["foo"]
#                        # <-- This line breaks parsing (no space after #)
# [tool.uv]
# ///
```

### Solution

Update the regex to accept `#` with or without trailing content:

```python
PEP723_PATTERN = re.compile(
    r"^# /// script\s*\n((?:#[^\n]*\n)*?)# ///$",
    re.MULTILINE,
)
```

The pattern `#[^\n]*` matches `#` followed by any characters (including none) until newline.

Also update the TOML extraction logic to handle bare `#` lines:

```python
for line in match.group(1).splitlines():
    if line.startswith("# "):
        toml_lines.append(line[2:])
    elif line == "#":
        toml_lines.append("")  # Bare # becomes empty line
    else:
        toml_lines.append(line.lstrip("# "))
```

### Files to Change

- `rootstock/pep723.py`: Update `PEP723_PATTERN` and parsing logic

---

## 2. HOME Directory for Model Caches

### Problem

Different MLIP libraries cache weights in different locations:

| Library | Cache Location | Respects XDG_CACHE_HOME? |
|---------|---------------|--------------------------|
| MACE | `~/.cache/mace/` | ✓ Yes |
| FAIRChem | `~/.cache/fairchem/` | ✗ No |
| MatGL | `~/.matgl/` | ✗ No |
| HuggingFace | `~/.cache/huggingface/` | ✓ Yes (with HF_HOME) |

When `XDG_CACHE_HOME` is set to `{root}/cache/`, FAIRChem and MatGL still write to the user's actual home directory, which:
- Isn't shared with collaborators
- Doesn't get wiped when removing the rootstock directory
- Requires per-user symlink setup (fragile)

### Solution

Set `HOME={root}/home` for both build and worker processes. This redirects all `~/` references to a directory inside the rootstock root.

**Updated directory structure**:
```
{root}/
├── home/                    # Fake HOME for build & workers
│   ├── .cache/
│   │   └── fairchem/        # FAIRChem weights
│   └── .matgl/              # MatGL weights
├── cache/                   # XDG_CACHE_HOME (still useful for well-behaved libs)
│   ├── mace/
│   └── huggingface/
├── environments/
├── envs/
└── .python/
```

### Code Changes

**`rootstock/environment.py`** — Update `get_model_cache_env()`:

```python
def get_model_cache_env(root: Path) -> dict[str, str]:
    """
    Get environment variables to redirect model downloads to shared cache.

    We set HOME to redirect libraries that use ~/ for caching (FAIRChem, MatGL).
    We also set XDG_CACHE_HOME for libraries that respect it (MACE).
    """
    cache_dir = root / "cache"
    home_dir = root / "home"
    
    return {
        # Redirect HOME so libraries using ~/ find the shared cache
        "HOME": str(home_dir),
        # XDG base directory - catches MACE and other well-behaved libraries
        "XDG_CACHE_HOME": str(cache_dir),
        # HuggingFace explicit (some tools check these before XDG)
        "HF_HOME": str(cache_dir / "huggingface"),
        "HF_HUB_CACHE": str(cache_dir / "huggingface" / "hub"),
    }
```

**`rootstock/cli.py`** — Ensure `{root}/home` is created during build:

```python
def cmd_build(args) -> int:
    # ... existing setup ...
    
    # Ensure home directory exists for model downloads
    home_dir = root / "home"
    home_dir.mkdir(parents=True, exist_ok=True)
    
    # ... rest of build ...
```

The `get_model_cache_env()` function is already used by:
- `EnvironmentManager.get_environment_variables()` — for worker processes at runtime
- `cmd_build()` — for model downloads during build

So updating it in one place fixes both contexts.

### Migration

For existing installations where weights landed in the user's home directory:

```bash
# Move weights to shared location
mkdir -p $ROOTSTOCK_ROOT/home/.cache
mkdir -p $ROOTSTOCK_ROOT/home/.matgl

mv ~/.cache/fairchem $ROOTSTOCK_ROOT/home/.cache/
mv ~/.matgl/* $ROOTSTOCK_ROOT/home/.matgl/
```

Or simply rebuild the environments with `--force` after the code change.

---

## 3. Implementation Checklist

### Parser Fix
- [ ] Update `PEP723_PATTERN` regex in `rootstock/pep723.py`
- [ ] Update TOML line extraction to handle bare `#`
- [ ] Add test case for metadata with bare `#` lines

### HOME Directory
- [ ] Update `get_model_cache_env()` to include `HOME`
- [ ] Create `{root}/home` directory in `cmd_build()`
- [ ] Update CLAUDE.md to document the directory structure

### Testing
- [ ] Rebuild UMA environment, verify weights land in `{root}/home/.cache/fairchem/`
- [ ] Rebuild TensorNet environment, verify weights land in `{root}/home/.matgl/`
- [ ] Run smoke test calculations for both

---

## 4. Notes

- The `HOME` redirect only affects worker subprocesses and build-time downloads, not the user's shell
- Some libraries may create other dotfiles in `~/` — they'll all end up in `{root}/home/` which is fine
- This approach is more robust than trying to track library-specific env vars for each MLIP