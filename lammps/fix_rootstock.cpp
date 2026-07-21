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
   fix rootstock - MLIP forces added via i-PI protocol over Unix sockets

   Usage:
     fix <id> <group> rootstock cluster <name> checkpoint <ckpt> \
         device <dev> elements <e1> <e2> ...

   The worker is auto-spawned via `rootstock serve`. Protocol lives in
   RootstockIPI (rootstock_ipi.cpp), shared with pair_style rootstock.
------------------------------------------------------------------------- */

#include "fix_rootstock.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "update.h"

#include <cstdlib>

using namespace LAMMPS_NS;

// ---------------------------------------------------------------------------
// Constructor — parse keyword arguments
//   fix <id> <group> rootstock cluster <name> checkpoint <ckpt>
//       device <dev> timeout <sec> elements <e1> <e2> ...
// ---------------------------------------------------------------------------
FixRootstock::FixRootstock(LAMMPS *lmp, int narg, char **arg)
    : Fix(lmp, narg, arg), client_(lmp->error), energy_(0.0) {

  if (narg < 4) error->all(FLERR, "fix rootstock: not enough arguments");

  // Parse keyword arguments starting from arg[3]
  int iarg = 3;
  bool found_elements = false;

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
    } else if (key == "elements") {
      found_elements = true;
      iarg++;
      break;
    } else {
      error->all(FLERR, "fix rootstock: unknown keyword '{}'", key);
    }
    iarg++;
  }

  // Validate required keywords
  if (client_.cluster.empty())
    error->all(FLERR, "fix rootstock: 'cluster' keyword is required");
  if (client_.checkpoint.empty())
    error->all(FLERR, "fix rootstock: 'checkpoint' keyword is required "
                      "(canonical id, e.g. 'mace-mp-0-medium')");
  if (!found_elements)
    error->all(FLERR, "fix rootstock: 'elements' keyword is required");

  // Parse element list (remaining args after 'elements')
  int ntypes = atom->ntypes;
  int nelem = narg - iarg;
  if (nelem != ntypes)
    error->all(FLERR,
               "fix rootstock: {} elements given but {} atom types defined",
               nelem, ntypes);

  elements_.resize(ntypes);
  for (int i = 0; i < ntypes; i++) {
    std::string sym = arg[iarg + i];
    if (RootstockIPI::element_to_z(sym) < 0)
      error->all(FLERR, "fix rootstock: unknown element '{}'", sym);
    elements_[i] = sym;
  }

  // Enable thermo output: energy via compute_scalar()
  scalar_flag = 1;
  global_freq = 1;
  energy_global_flag = 1;
  extscalar = 1;

  // Enable virial contribution for NPT
  virial_global_flag = 1;
  thermo_virial = 1;
}

// ---------------------------------------------------------------------------
// setmask — we operate in post_force
// ---------------------------------------------------------------------------
int FixRootstock::setmask() { return FixConst::POST_FORCE; }

// ---------------------------------------------------------------------------
// refresh_atomic_numbers — rebuild the type -> Z mapping for local atoms
// ---------------------------------------------------------------------------
void FixRootstock::refresh_atomic_numbers() {
  int nlocal = atom->nlocal;
  int *type = atom->type;
  numbers_.resize(nlocal);
  for (int i = 0; i < nlocal; i++)
    numbers_[i] = RootstockIPI::element_to_z(elements_[type[i] - 1]);
  client_.set_atomic_numbers(numbers_);
}

// ---------------------------------------------------------------------------
// init — start the worker on first call
// ---------------------------------------------------------------------------
void FixRootstock::init() {
  // Validate units
  if (std::string(update->unit_style) != "metal")
    error->all(FLERR, "fix rootstock requires 'units metal'");

  // The worker sees all atoms and computes its own neighborhoods; there is
  // no force decomposition across ranks.
  if (comm->nprocs > 1)
    error->all(FLERR,
               "fix rootstock requires a single MPI rank (run with "
               "'mpirun -np 1')");

  // Only do socket/worker setup on first call.
  // LAMMPS calls init() at the start of every `run` command.
  if (client_.running()) return;

  // Warn about atom types with zero atoms
  int nlocal = atom->nlocal;
  int *type = atom->type;
  for (int t = 1; t <= atom->ntypes; t++) {
    int count = 0;
    for (int i = 0; i < nlocal; i++)
      if (type[i] == t) count++;
    if (count == 0)
      error->warning(FLERR, "fix rootstock: atom type {} ({}) has no atoms", t,
                     elements_[t - 1]);
  }

  refresh_atomic_numbers();
  client_.start("fix rootstock", std::string(id));
}

// ---------------------------------------------------------------------------
// setup — delegate to post_force (LAMMPS calls this for initial forces)
// ---------------------------------------------------------------------------
void FixRootstock::setup(int vflag) { post_force(vflag); }

// ---------------------------------------------------------------------------
// post_force — every timestep: send positions, receive forces
// ---------------------------------------------------------------------------
void FixRootstock::post_force(int /* vflag */) {
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

  double v6[6];
  client_.exchange(cell, pos_.data(), nlocal, energy_, frc_.data(), v6);

  double **f = atom->f;
  for (int i = 0; i < nlocal; i++) {
    f[i][0] += frc_[3 * i + 0];
    f[i][1] += frc_[3 * i + 1];
    f[i][2] += frc_[3 * i + 2];
  }

  for (int k = 0; k < 6; k++) virial[k] = v6[k];
}

// ---------------------------------------------------------------------------
// compute_scalar — return cached energy for thermo output
// ---------------------------------------------------------------------------
double FixRootstock::compute_scalar() { return energy_; }
