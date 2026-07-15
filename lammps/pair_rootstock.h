/* -*- c++ -*- ----------------------------------------------------------
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

   Communicates with a rootstock worker process that runs an MLIP model
   (MACE, CHGNet, UMA, TensorNet, etc.) in an isolated Python environment.
   The worker is auto-spawned via `rootstock serve`.

   As a pair style, the MLIP participates natively in thermo `pe`,
   `compute pair`, pressure, and pair_style hybrid/scaled — which is what
   pair-style-assuming drivers (e.g. calphy) require. This is the
   recommended integration when the MLIP is the only potential; use
   fix rootstock to ADD MLIP forces on top of another potential.

   Usage:
     pair_style rootstock cluster <name> checkpoint <ckpt> \
                [device <dev>] [timeout <sec>] [cutoff <r>]
     pair_coeff * * <e1> <e2> ...

   `checkpoint` is a canonical checkpoint id (e.g. 'mace-mp-0-medium'), the
   same id used by RootstockCalculator and the `rootstock` CLI. The
   elements on pair_coeff map atom types in order (type 1 = e1, ...).

   `cutoff` only sizes LAMMPS's neighbor/communication bookkeeping — the
   worker computes its own neighborhoods from the full cell. Default 1.0.

   Per-atom energy and per-atom stress (compute pe/atom, stress/atom) are
   not provided; the worker reports global quantities only.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(rootstock, PairRootstock)
// clang-format on
#else

#ifndef LMP_PAIR_ROOTSTOCK_H
#define LMP_PAIR_ROOTSTOCK_H

#include "pair.h"
#include "rootstock_ipi.h"

#include <string>
#include <vector>

namespace LAMMPS_NS {

class PairRootstock : public Pair {
 public:
  PairRootstock(class LAMMPS *);
  ~PairRootstock() override;

  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;

 private:
  RootstockIPI client_;
  std::vector<std::string> elements_;    // per atom type, 0-indexed
  double cut_comm_;

  // Per-call scratch buffers
  std::vector<int> numbers_;
  std::vector<double> pos_;
  std::vector<double> frc_;

  void allocate();
  void refresh_atomic_numbers();
};

}    // namespace LAMMPS_NS

#endif
#endif
