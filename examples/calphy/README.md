# calphy + LAMMPS + rootstock on Della

Free-energy calculations with calphy driving a rootstock MLIP through
`pair_style rootstock`. Steps 1–3 are one-time per user.

Inputs in this directory:

- `input.yaml` + `job.slurm` — Cu fcc at 1000 K with `mace-mp-0-medium`,
  smoke-test scale; submit with `sbatch job.slurm`.
- `input-fe-gpu.yaml` — Fe bcc at 100 K with `mattersim-v1-0-0-5m`,
  production-scale steps; run inside a GPU allocation:
  `(unset $(env | grep -o '^SLURM_[A-Za-z0-9_]*'); calphy -i input-fe-gpu.yaml)`
- `input-fe-slurm.yaml` — same calculation, but calphy submits its own Slurm
  job: `calphy -i input-fe-slurm.yaml` from a login node.

## 1. Python environment

On a Della login node:

```bash
module purge
module load anaconda3/2024.10                   # or latest: module avail anaconda3
module load gcc-toolset/14 openmpi/gcc/4.1.8    # calphy's mpi4py dep builds against real MPI
conda create -n calphy python=3.12 -y
conda activate calphy
pip install calphy rootstock
```

Any env manager works; step 3 installs the LAMMPS module into whichever env
is active. `calphy` imports `lammps` at startup, so it won't run (even
`calphy --help`) until step 3 is done — don't work around that with
`pip install lammps`; the PyPI wheel has no rootstock styles.

## 2. LAMMPS build

```bash
git clone --depth 1 -b stable https://github.com/lammps/lammps.git
git clone https://github.com/Garden-AI/rootstock.git
./rootstock/lammps/install.sh ./lammps/src
cd lammps && mkdir build && cd build
module purge && module load gcc-toolset/14 openmpi/gcc/4.1.8 cmake/3.30.8
cmake ../cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=yes \
    -DPKG_MANYBODY=yes -DPKG_MISC=yes -DPKG_EXTRA-COMPUTE=yes \
    -DPKG_EXTRA-DUMP=yes -DPKG_EXTRA-FIX=yes -DPKG_EXTRA-PAIR=yes
make -j 8
./lmp -h | grep -i rootstock    # rootstock under pair styles AND fix styles
```

No CUDA toolkit, no per-MLIP flags — the GPU work happens in rootstock's
worker env; LAMMPS is a stock CPU build.

## 3. LAMMPS Python module into the calphy env

`make install-python` installs into the Python that CMake found at configure
time, not the currently active env. Point CMake at the env first:

```bash
source ~/envs/calphy/bin/activate
cd ~/lammps/build
cmake . -DPython_EXECUTABLE=$(which python)
make install-python
python -c "from lammps import lammps; l = lammps(cmdargs=['-log','none','-screen','none']); \
    print(l.has_style('pair','rootstock'), l.has_style('fix','ti/spring'))"
# True True — first is the rootstock style, second is calphy's switching fix
# (EXTRA-FIX package; False means the build is missing the PKG_* flags above)
```

If the make step prints "Installing wheel into system site-packages folder",
the cmake line didn't take — rerun both from the build directory with the env
active.

## 4. Preflight

```bash
rootstock status --root /scratch/gpfs/ROSENGROUP/common/rootstock
# the checkpoint your input names shows as fetched/verified
```

If a checkpoint is missing, provision it once into the shared root (needs
rosengroup write access):

```bash
rootstock install ~/rootstock-src/sample_model_configurations/nvidia_configs/<env>.py \
    --root /scratch/gpfs/ROSENGROUP/common/rootstock
rootstock add <checkpoint-id> --root /scratch/gpfs/ROSENGROUP/common/rootstock --no-verify
```

On a GPU node, before a long run:

```bash
rootstock smoke-test --checkpoint <checkpoint-id> --device cuda \
    --root /scratch/gpfs/ROSENGROUP/common/rootstock
```

## 5. Run

From a copy of this directory:

```bash
sbatch job.slurm
```

Roughly 30–45 min on one A100. Calphy writes a `fe-*` run directory next to
`input.yaml`.

## 6. Verify

```bash
cat fe-*/report.yaml
```

`results.free_energy` present and finite (eV/atom), and the
forward/backward switching error reported in `fe-*/calphy.log` is small
relative to the free energy.

## Troubleshooting

- `Unrecognized fix style 'ti/spring' ... EXTRA-FIX package` (mid-run, at the
  switching stage) → the LAMMPS build is missing the `PKG_*` flags from
  step 2. Rerun step 2's cmake with all flags, `make -j 8`, then redo step 3.
- Calphy crashes but the Slurm job keeps RUNNING at 0% GPU (cluster
  zero-utilization emails) → leftover LAMMPS/worker processes outlive the
  crashed kernel. Both example job paths carry a traceback watchdog that
  ends the job within ~3 minutes of a crash; if you removed it or wrote
  your own variant, `scancel` manually. The traceback is in
  `fe-*.sub.slurm.err` (slurm scheduler) or `local.err` (local scheduler).

- `OPAL ERROR: Unreachable in file ext3x_client.c` / "appears to have been
  direct launched using srun" → calphy's launcher uses srun inside Slurm
  allocations, and Della's OpenMPI can't be srun-launched. `job.slurm`
  already handles it; for interactive runs, prefix with:

  ```bash
  (unset $(env | grep -o '^SLURM_[A-Za-z0-9_]*'); calphy -i input.yaml)
  ```

- `make install-python` targets system Python → step 3's `cmake .
  -DPython_EXECUTABLE=$(which python)`, then rerun.
- "requires a single MPI rank" → set `queue.cores: 1` in `input.yaml`.
- 30–60 s stall at the start of each calphy stage → each stage is a fresh
  LAMMPS instance and the worker reloads the model. Normal.
- Run dies with no `report.yaml` → read `fe-*/calphy.log`, then the LAMMPS
  log in the same directory.
