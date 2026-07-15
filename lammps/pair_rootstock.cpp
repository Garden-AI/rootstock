/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   pair_style rootstock - MLIP as a genuine pair style via i-PI sockets

   Usage:
     pair_style rootstock cluster <name> checkpoint <ckpt> \
                [device <dev>] [timeout <sec>] [cutoff <r>]
     pair_coeff * * <e1> <e2> ...

   The worker is auto-spawned via `rootstock serve`. Protocol lives in
   RootstockIPI (rootstock_ipi.cpp), shared with fix rootstock.
------------------------------------------------------------------------- */

#include "pair_rootstock.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "memory.h"
#include "neighbor.h"
#include "update.h"

#include <cstdlib>
#include <cstring>

using namespace LAMMPS_NS;

// ---------------------------------------------------------------------------
// Constructor / destructor
// ---------------------------------------------------------------------------
PairRootstock::PairRootstock(LAMMPS *lmp)
    : Pair(lmp), client_(lmp->error), cut_comm_(1.0) {
  single_enable = 0;      // no pairwise decomposition exists
  restartinfo = 0;        // nothing to write to restart files
  one_coeff = 1;          // single pair_coeff * * line
  manybody_flag = 1;
  no_virial_fdotr_compute = 1;    // virial comes whole from the worker
}

PairRootstock::~PairRootstock() {
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }
}

// ---------------------------------------------------------------------------
// settings — parse pair_style arguments
//   pair_style rootstock cluster <name> checkpoint <ckpt>
//              [device <dev>] [timeout <sec>] [cutoff <r>]
// ---------------------------------------------------------------------------
void PairRootstock::settings(int narg, char **arg) {
  int iarg = 0;
  while (iarg < narg) {
    std::string key = arg[iarg];

    if (key == "cluster" && iarg + 1 < narg) {
      client_.cluster = arg[++iarg];
    } else if (key == "checkpoint" && iarg + 1 < narg) {
      client_.checkpoint = arg[++iarg];
    } else if (key == "device" && iarg + 1 < narg) {
      client_.device = arg[++iarg];
    } else if (key == "timeout" && iarg + 1 < narg) {
      client_.timeout = std::atoi(arg[++iarg]);
    } else if (key == "cutoff" && iarg + 1 < narg) {
      cut_comm_ = std::atof(arg[++iarg]);
      if (cut_comm_ <= 0.0)
        error->all(FLERR, "pair_style rootstock: cutoff must be positive");
    } else {
      error->all(FLERR, "pair_style rootstock: unknown keyword '{}'", key);
    }
    iarg++;
  }

  if (client_.cluster.empty())
    error->all(FLERR, "pair_style rootstock: 'cluster' keyword is required");
  if (client_.checkpoint.empty())
    error->all(FLERR,
               "pair_style rootstock: 'checkpoint' keyword is required "
               "(canonical id, e.g. 'mace-mp-0-medium')");
}

// ---------------------------------------------------------------------------
// coeff — parse pair_coeff * * <e1> <e2> ...
// ---------------------------------------------------------------------------
void PairRootstock::coeff(int narg, char **arg) {
  if (!allocated) allocate();

  int ntypes = atom->ntypes;
  if (narg != 2 + ntypes)
    error->all(FLERR,
               "pair_coeff rootstock: expected '* * <{} element symbols>', "
               "got {} arguments",
               ntypes, narg);
  if (std::strcmp(arg[0], "*") != 0 || std::strcmp(arg[1], "*") != 0)
    error->all(FLERR, "pair_coeff rootstock: only '* *' is supported");

  elements_.resize(ntypes);
  for (int i = 0; i < ntypes; i++) {
    std::string sym = arg[2 + i];
    if (RootstockIPI::element_to_z(sym) < 0)
      error->all(FLERR, "pair_coeff rootstock: unknown element '{}'", sym);
    elements_[i] = sym;
  }

  for (int i = 1; i <= ntypes; i++)
    for (int j = i; j <= ntypes; j++) setflag[i][j] = 1;
}

// ---------------------------------------------------------------------------
// allocate — standard Pair arrays
// ---------------------------------------------------------------------------
void PairRootstock::allocate() {
  allocated = 1;
  int n = atom->ntypes + 1;

  memory->create(setflag, n, n, "pair:setflag");
  for (int i = 1; i < n; i++)
    for (int j = i; j < n; j++) setflag[i][j] = 0;

  memory->create(cutsq, n, n, "pair:cutsq");
}

// ---------------------------------------------------------------------------
// refresh_atomic_numbers — rebuild the type -> Z mapping for local atoms
// ---------------------------------------------------------------------------
void PairRootstock::refresh_atomic_numbers() {
  int nlocal = atom->nlocal;
  int *type = atom->type;
  numbers_.resize(nlocal);
  for (int i = 0; i < nlocal; i++)
    numbers_[i] = RootstockIPI::element_to_z(elements_[type[i] - 1]);
  client_.set_atomic_numbers(numbers_);
}

// ---------------------------------------------------------------------------
// init_style — validate, request a (nominal) neighbor list, start worker
// ---------------------------------------------------------------------------
void PairRootstock::init_style() {
  if (std::string(update->unit_style) != "metal")
    error->all(FLERR, "pair_style rootstock requires 'units metal'");

  // The worker sees all atoms and computes its own neighborhoods; there is
  // no force decomposition across ranks.
  if (comm->nprocs > 1)
    error->all(FLERR,
               "pair_style rootstock requires a single MPI rank (run with "
               "'mpirun -np 1')");

  if (elements_.empty())
    error->all(FLERR,
               "pair_style rootstock: pair_coeff * * <elements> is required");

  // The list is never read — the worker builds its own neighborhoods — but
  // LAMMPS's neighbor machinery expects every pair style to hold one.
  neighbor->add_request(this);

  // Only do socket/worker setup on first call.
  // LAMMPS calls init_style() at the start of every `run` command.
  if (client_.running()) return;

  // Warn about atom types with zero atoms
  int nlocal = atom->nlocal;
  int *type = atom->type;
  for (int t = 1; t <= atom->ntypes; t++) {
    int count = 0;
    for (int i = 0; i < nlocal; i++)
      if (type[i] == t) count++;
    if (count == 0)
      error->warning(FLERR, "pair_style rootstock: atom type {} ({}) has no atoms",
                     t, elements_[t - 1]);
  }

  refresh_atomic_numbers();
  client_.start("pair_style rootstock", "pair");
}

// ---------------------------------------------------------------------------
// init_one — nominal cutoff for neighbor/comm bookkeeping only
// ---------------------------------------------------------------------------
double PairRootstock::init_one(int /* i */, int /* j */) { return cut_comm_; }

// ---------------------------------------------------------------------------
// compute — every timestep: send positions, receive energy/forces/virial
// ---------------------------------------------------------------------------
void PairRootstock::compute(int eflag, int vflag) {
  ev_init(eflag, vflag);

  int nlocal = atom->nlocal;

  // Refresh every call: atom sorting reorders local atoms without changing
  // nlocal, which would silently desync a cached species mapping.
  refresh_atomic_numbers();

  double cell[3][3] = {
      {domain->boxhi[0] - domain->boxlo[0], 0.0, 0.0},
      {domain->xy, domain->boxhi[1] - domain->boxlo[1], 0.0},
      {domain->xz, domain->yz, domain->boxhi[2] - domain->boxlo[2]}};

  double **x = atom->x;
  pos_.resize(3 * nlocal);
  frc_.resize(3 * nlocal);
  for (int i = 0; i < nlocal; i++) {
    pos_[3 * i + 0] = x[i][0];
    pos_[3 * i + 1] = x[i][1];
    pos_[3 * i + 2] = x[i][2];
  }

  double energy;
  double v6[6];
  client_.exchange(cell, pos_.data(), nlocal, energy, frc_.data(), v6);

  double **f = atom->f;
  for (int i = 0; i < nlocal; i++) {
    f[i][0] += frc_[3 * i + 0];
    f[i][1] += frc_[3 * i + 1];
    f[i][2] += frc_[3 * i + 2];
  }

  if (eflag_global) eng_vdwl += energy;
  if (vflag_global)
    for (int k = 0; k < 6; k++) virial[k] += v6[k];
}
