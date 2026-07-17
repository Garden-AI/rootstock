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
   RootstockIPI - shared i-PI dialect client for the rootstock styles

   Owns the worker subprocess and Unix socket, and speaks the rootstock
   i-PI dialect (JSON INIT payload, re-sent every force cycle). The wire
   format is a frozen contract with rootstock/protocol.py; both
   fix rootstock and pair_style rootstock delegate to this class so the
   protocol has exactly one implementation on the LAMMPS side.

   Not a LAMMPS style itself — no FixStyle/PairStyle registration.
------------------------------------------------------------------------- */

#ifndef LMP_ROOTSTOCK_IPI_H
#define LMP_ROOTSTOCK_IPI_H

#include <string>
#include <sys/types.h>
#include <vector>

namespace LAMMPS_NS {

class Error;

class RootstockIPI {
 public:
  explicit RootstockIPI(Error *error);
  ~RootstockIPI();

  // User-facing configuration; set before start().
  std::string cluster;
  std::string checkpoint;
  std::string device = "cuda";
  int timeout = 120;

  bool running() const { return client_fd_ >= 0; }

  // Species (atomic numbers) sent with every (re-)INIT. Callers must
  // refresh this before each exchange(): LAMMPS re-sorts atoms without
  // changing nlocal, so a mapping captured once goes stale silently.
  void set_atomic_numbers(std::vector<int> numbers);

  // Resolve the cluster root, bind the socket, spawn `rootstock serve`,
  // accept the worker connection, and complete the INIT handshake.
  // `style` prefixes error messages (e.g. "pair_style rootstock");
  // `tag` disambiguates the socket path within this process.
  void start(const std::string &style, const std::string &tag);

  // One force evaluation. `cell` is the row-major lower-triangular LAMMPS
  // box in Angstrom; `x` is 3*natoms positions in Angstrom. Outputs:
  // energy in eV, forces in eV/Angstrom (3*natoms, overwritten), virial
  // in eV in LAMMPS Voigt order (xx, yy, zz, xy, xz, yz).
  void exchange(const double cell[3][3], const double *x, int natoms,
                double &energy, double *forces, double virial6[6]);

  static int element_to_z(const std::string &symbol);

 private:
  Error *error_;
  std::string style_ = "rootstock";
  std::string socket_dir_;
  std::string socket_path_;
  int server_fd_ = -1;
  int client_fd_ = -1;
  pid_t worker_pid_ = -1;
  std::vector<int> atomic_numbers_;

  std::string resolve_cluster();
  pid_t spawn_worker(const std::string &root);

  void sendall(const void *buf, size_t len);
  void recvall(void *buf, size_t len);
  void sendmsg(const char *msg);
  std::string recvmsg();

  void send_init();
  void send_posdata(const double cell[3][3], const double *x, int natoms);
  void recv_forceready(int natoms, double &energy, double *forces,
                       double virial6[6]);
};

}    // namespace LAMMPS_NS

#endif
