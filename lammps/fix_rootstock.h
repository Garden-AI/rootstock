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
   fix rootstock - MLIP calculator via i-PI protocol over Unix sockets

   Communicates with a rootstock worker process that runs an MLIP model
   (MACE, CHGNet, UMA, TensorNet, etc.) in an isolated Python environment.
   The worker is started externally via `rootstock serve`.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(rootstock, FixRootstock)
// clang-format on
#else

#ifndef LMP_FIX_ROOTSTOCK_H
#define LMP_FIX_ROOTSTOCK_H

#include "fix.h"
#include <string>
#include <vector>

namespace LAMMPS_NS {

class FixRootstock : public Fix {
 public:
  FixRootstock(class LAMMPS *, int, char **);
  ~FixRootstock() override;

  int setmask() override;
  void init() override;
  void post_force(int) override;
  double compute_scalar() override;

 private:
  // Socket state
  std::string socket_path_;
  int server_fd_;
  int client_fd_;

  // Atom info
  std::vector<int> atomic_numbers_;
  std::vector<std::string> elements_;

  // Cached energy for thermo output
  double energy_;

  // Socket I/O helpers
  void sendall(const void *buf, size_t len);
  void recvall(void *buf, size_t len);
  void sendmsg(const char *msg);
  std::string recvmsg();

  // Protocol helpers
  void send_init();
  void send_posdata();
  void recv_forceready();
  void send_status();
  std::string recv_status();

  // Element lookup
  static int element_to_z(const std::string &symbol);
};

}    // namespace LAMMPS_NS

#endif
#endif
