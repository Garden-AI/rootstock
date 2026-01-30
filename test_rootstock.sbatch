#!/bin/bash
#SBATCH --job-name=rootstock-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=00:30:00

module purge
source ~/.venvs/rootstock/bin/activate
cd ~/rootstock

export ROOTSTOCK_ROOT="/scratch/gpfs/ROSENGROUP/ew2876/rootstock"
export XDG_CACHE_HOME="$ROOTSTOCK_ROOT/cache"

# Quick smoke test
python -c "
from ase.build import bulk
from rootstock import RootstockCalculator
import os

atoms = bulk('Cu', 'fcc', a=3.6) * (5, 5, 5)
with RootstockCalculator(root=os.environ['ROOTSTOCK_ROOT'], model='mace-medium', device='cuda') as calc:
    atoms.calc = calc
    print(f'Energy: {atoms.get_potential_energy():.4f} eV')
    print('SMOKE TEST PASSED')
"

# Sanity check benchmark
python -m benchmarks.sanity_check \
    --atoms 1000 \
    --steps 5000 \
    --root $ROOTSTOCK_ROOT \
    --device cuda