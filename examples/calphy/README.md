# calphy + LAMMPS + rootstock on Della

Free-energy calculations with calphy driving a rootstock MLIP through
`pair_style rootstock`. The example computes the Cu fcc solid free energy at
1000 K with `mace-mp-0-medium`. Steps 1–3 are one-time per user.

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
is active.

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
python -c "from lammps import lammps; \
    print(lammps(cmdargs=['-log','none','-screen','none']).has_style('pair','rootstock'))"
# True
```

If the make step prints "Installing wheel into system site-packages folder",
the cmake line didn't take — rerun both from the build directory with the env
active.

## 4. Preflight

```bash
rootstock status --root /scratch/gpfs/ROSENGROUP/common/rootstock
# mace / mace-mp-0-medium: verified
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

- `make install-python` targets system Python → step 3's `cmake .
  -DPython_EXECUTABLE=$(which python)`, then rerun.
- "requires a single MPI rank" → set `queue.cores: 1` in `input.yaml`.
- 30–60 s stall at the start of each calphy stage → each stage is a fresh
  LAMMPS instance and the worker reloads the model. Normal.
- CUDA errors on MIG slices → keep `--constraint="intel&gpu40"` (full GPU).
- Run dies with no `report.yaml` → read `fe-*/calphy.log`, then the LAMMPS
  log in the same directory.
