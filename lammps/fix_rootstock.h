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
   fix rootstock - MLIP forces added via i-PI protocol over Unix sockets

   Communicates with a rootstock worker process that runs an MLIP model
   (MACE, CHGNet, UMA, TensorNet, etc.) in an isolated Python environment.
   The worker is auto-spawned via `rootstock serve`.

   Forces are ADDED to existing forces, so the MLIP can be combined with
   a real pair style. When the MLIP is the only potential, prefer
   pair_style rootstock: it contributes to thermo `pe` and `compute pair`
   natively and composes with pair_style hybrid/scaled.

   Usage:
     fix <id> <group> rootstock cluster <name> checkpoint <ckpt> \
         device <dev> elements <e1> <e2> ...

   `checkpoint` is a canonical checkpoint id (e.g. 'mace-mp-0-medium'), the
   same id used by RootstockCalculator and the `rootstock` CLI.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(rootstock, FixRootstock)
// clang-format on
#else

#ifndef LMP_FIX_ROOTSTOCK_H
#define LMP_FIX_ROOTSTOCK_H

#include "fix.h"
#include "rootstock_ipi.h"

#include <string>
#include <vector>

namespace LAMMPS_NS {

class FixRootstock : public Fix {
 public:
  FixRootstock(class LAMMPS *, int, char **);

  int setmask() override;
  void init() override;
  void post_force(int) override;
  void setup(int) override;
  double compute_scalar() override;

 private:
  RootstockIPI client_;
  std::vector<std::string> elements_;    // per atom type, 0-indexed

  // Cached energy for thermo output
  double energy_;

  // Per-call scratch buffers
  std::vector<int> numbers_;
  std::vector<double> pos_;
  std::vector<double> frc_;

  void refresh_atomic_numbers();
};

}    // namespace LAMMPS_NS

#endif
#endif
