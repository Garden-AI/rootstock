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

#include "rootstock_ipi.h"

#include "error.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>

#include <signal.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

using namespace LAMMPS_NS;

// ---------------------------------------------------------------------------
// Unit conversions — must match rootstock/protocol.py exactly
// ---------------------------------------------------------------------------
static constexpr double BOHR_TO_ANGSTROM = 0.52917721067;
static constexpr double HARTREE_TO_EV = 27.211386245988;
static constexpr double ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM;

// ---------------------------------------------------------------------------
// Periodic table: element symbol -> atomic number (1-indexed)
// ---------------------------------------------------------------------------
static const char *ELEMENT_SYMBOLS[] = {
    "",   "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na",
    "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",
    "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br",
    "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag",
    "Cd", "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr",
    "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi",
    "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U",  "Np", "Pu", "Am",
    "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh",
    "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"};
static constexpr int NUM_ELEMENTS = 118;

int RootstockIPI::element_to_z(const std::string &symbol) {
  for (int i = 1; i <= NUM_ELEMENTS; i++) {
    if (symbol == ELEMENT_SYMBOLS[i]) return i;
  }
  return -1;
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------
RootstockIPI::RootstockIPI(Error *error) : error_(error) {}

RootstockIPI::~RootstockIPI() {
  // Best-effort EXIT message
  if (client_fd_ >= 0) {
    char buf[12];
    std::memset(buf, ' ', 12);
    std::memcpy(buf, "EXIT", 4);
    ::send(client_fd_, buf, 12, MSG_NOSIGNAL);
    ::close(client_fd_);
  }
  if (server_fd_ >= 0) ::close(server_fd_);

  // Clean up worker process
  if (worker_pid_ > 0) {
    ::kill(worker_pid_, SIGTERM);
    int status;
    // Give worker up to 5 seconds to exit
    for (int i = 0; i < 50; i++) {
      if (::waitpid(worker_pid_, &status, WNOHANG) != 0) break;
      usleep(100000);    // 100ms
    }
    // Force kill if still alive
    if (::waitpid(worker_pid_, &status, WNOHANG) == 0) {
      ::kill(worker_pid_, SIGKILL);
      ::waitpid(worker_pid_, &status, 0);
    }
  }

  // Clean up socket file
  if (!socket_path_.empty()) ::unlink(socket_path_.c_str());
}

void RootstockIPI::set_atomic_numbers(std::vector<int> numbers) {
  atomic_numbers_ = std::move(numbers);
}

// ---------------------------------------------------------------------------
// resolve_cluster — call `rootstock resolve --cluster <name> --json`
// ---------------------------------------------------------------------------
std::string RootstockIPI::resolve_cluster() {
  std::string cmd = "rootstock resolve --cluster " + cluster + " --json";
  FILE *pipe = popen(cmd.c_str(), "r");
  if (!pipe)
    error_->all(FLERR,
                "{}: failed to run 'rootstock resolve'. "
                "Is rootstock installed? (pip install rootstock)",
                style_);

  char buffer[1024];
  std::string output;
  while (fgets(buffer, sizeof(buffer), pipe)) output += buffer;

  int status = pclose(pipe);
  if (status != 0)
    error_->all(FLERR,
                "{}: 'rootstock resolve --cluster {}' failed. Unknown cluster?",
                style_, cluster);

  // Parse "root" from JSON output: {"root": "/path/...", "cluster": "..."}
  // Simple string search — no JSON library needed for this minimal format
  std::string key = "\"root\": \"";
  auto pos = output.find(key);
  if (pos == std::string::npos)
    error_->all(FLERR, "{}: failed to parse 'rootstock resolve' output", style_);

  pos += key.size();
  auto end = output.find('"', pos);
  if (end == std::string::npos)
    error_->all(FLERR, "{}: failed to parse 'rootstock resolve' output", style_);

  return output.substr(pos, end - pos);
}

// ---------------------------------------------------------------------------
// spawn_worker — fork + execlp rootstock serve
// ---------------------------------------------------------------------------
pid_t RootstockIPI::spawn_worker(const std::string &root) {
  pid_t pid = fork();
  if (pid < 0) error_->all(FLERR, "{}: fork() failed", style_);

  if (pid == 0) {
    // Child: exec rootstock serve
    execlp("rootstock", "rootstock", "serve", checkpoint.c_str(), "--root",
           root.c_str(), "--socket", socket_path_.c_str(), "--device",
           device.c_str(), nullptr);
    // If exec fails, exit immediately
    _exit(127);
  }
  return pid;    // Parent: return child PID
}

// ---------------------------------------------------------------------------
// start — resolve cluster, create socket, spawn worker, accept, handshake
// ---------------------------------------------------------------------------
void RootstockIPI::start(const std::string &style, const std::string &tag) {
  style_ = style;
  if (running()) return;

  if (cluster.empty())
    error_->all(FLERR, "{}: 'cluster' keyword is required", style_);
  if (checkpoint.empty())
    error_->all(FLERR,
                "{}: 'checkpoint' keyword is required "
                "(canonical id, e.g. 'mace-mp-0-medium')",
                style_);
  if (atomic_numbers_.empty())
    error_->all(FLERR, "{}: atomic numbers not set before start", style_);

  // Resolve cluster root directory
  std::string root = resolve_cluster();

  // Generate unique socket path
  socket_path_ =
      "/tmp/rootstock_" + std::to_string(getpid()) + "_" + tag + ".sock";

  // Create Unix domain socket
  server_fd_ = ::socket(AF_UNIX, SOCK_STREAM, 0);
  if (server_fd_ < 0) error_->all(FLERR, "{}: socket() failed", style_);

  struct sockaddr_un addr;
  std::memset(&addr, 0, sizeof(addr));
  addr.sun_family = AF_UNIX;
  if (socket_path_.size() >= sizeof(addr.sun_path))
    error_->all(FLERR, "{}: socket path too long", style_);
  std::strncpy(addr.sun_path, socket_path_.c_str(), sizeof(addr.sun_path) - 1);

  // Remove stale socket file
  ::unlink(socket_path_.c_str());

  if (::bind(server_fd_, (struct sockaddr *) &addr, sizeof(addr)) < 0)
    error_->all(FLERR, "{}: bind() failed on {}", style_, socket_path_);

  if (::listen(server_fd_, 1) < 0)
    error_->all(FLERR, "{}: listen() failed", style_);

  // Spawn the worker
  worker_pid_ = spawn_worker(root);

  // Accept connection with configurable timeout
  fd_set fds;
  FD_ZERO(&fds);
  FD_SET(server_fd_, &fds);
  struct timeval tv;
  tv.tv_sec = timeout;
  tv.tv_usec = 0;

  int sel = ::select(server_fd_ + 1, &fds, nullptr, nullptr, &tv);
  if (sel <= 0)
    error_->all(FLERR,
                "{}: no worker connected within {} seconds. "
                "Worker may have failed to start. Check that an environment "
                "providing checkpoint '{}' is built on this cluster.",
                style_, timeout, checkpoint);

  client_fd_ = ::accept(server_fd_, nullptr, nullptr);
  if (client_fd_ < 0) error_->all(FLERR, "{}: accept() failed", style_);

  // INIT handshake: STATUS -> NEEDINIT -> INIT -> STATUS -> READY
  sendmsg("STATUS");
  std::string status = recvmsg();
  if (status != "NEEDINIT")
    error_->all(FLERR, "{}: expected NEEDINIT, got {}", style_, status);

  send_init();

  sendmsg("STATUS");
  status = recvmsg();
  if (status != "READY")
    error_->all(FLERR, "{}: expected READY after INIT, got {}", style_, status);
}

// ---------------------------------------------------------------------------
// exchange — one force cycle: positions out, energy/forces/virial back
// ---------------------------------------------------------------------------
void RootstockIPI::exchange(const double cell[3][3], const double *x,
                            int natoms, double &energy, double *forces,
                            double virial6[6]) {
  if ((int) atomic_numbers_.size() != natoms)
    error_->all(FLERR, "{}: {} atomic numbers set but {} atoms", style_,
                atomic_numbers_.size(), natoms);

  // STATUS -> check state
  sendmsg("STATUS");
  std::string status = recvmsg();

  // Worker returns to NEEDINIT after each FORCEREADY, so re-send INIT
  if (status == "NEEDINIT") {
    send_init();
    sendmsg("STATUS");
    status = recvmsg();
  }

  if (status != "READY")
    error_->all(FLERR, "{}: expected READY, got {}", style_, status);

  // Send positions
  send_posdata(cell, x, natoms);

  // STATUS -> HAVEDATA
  sendmsg("STATUS");
  status = recvmsg();
  if (status != "HAVEDATA")
    error_->all(FLERR, "{}: expected HAVEDATA, got {}", style_, status);

  // GETFORCE -> FORCEREADY
  sendmsg("GETFORCE");
  recv_forceready(natoms, energy, forces, virial6);
}

// ---------------------------------------------------------------------------
// send_init — send INIT message with JSON species data
// ---------------------------------------------------------------------------
void RootstockIPI::send_init() {
  // Build JSON: {"numbers": [29, 29, ...], "pbc": [true, true, true]}
  std::ostringstream json;
  json << "{\"numbers\": [";
  for (size_t i = 0; i < atomic_numbers_.size(); i++) {
    if (i > 0) json << ", ";
    json << atomic_numbers_[i];
  }
  json << "], \"pbc\": [true, true, true]}";
  std::string init_str = json.str();

  sendmsg("INIT");

  // bead index (int32)
  int32_t bead = 0;
  sendall(&bead, sizeof(bead));

  // init string length (int32) + bytes
  int32_t nbytes = (int32_t) init_str.size();
  sendall(&nbytes, sizeof(nbytes));
  sendall(init_str.data(), init_str.size());
}

// ---------------------------------------------------------------------------
// send_posdata — send cell + positions in atomic units
// ---------------------------------------------------------------------------
void RootstockIPI::send_posdata(const double cell[3][3], const double *x,
                                int natoms) {
  sendmsg("POSDATA");

  // LAMMPS box: lower-triangular cell
  //   [[lx,  0,  0],
  //    [xy, ly,  0],
  //    [xz, yz, lz]]
  double lx = cell[0][0];
  double ly = cell[1][1];
  double lz = cell[2][2];
  double xy = cell[1][0];
  double xz = cell[2][0];
  double yz = cell[2][1];

  // Transpose and convert to Bohr for i-PI column-major convention
  double cell_t[3][3];
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) cell_t[i][j] = cell[j][i] * ANGSTROM_TO_BOHR;

  // Inverse of lower-triangular 3x3
  double inv_lx = 1.0 / lx;
  double inv_ly = 1.0 / ly;
  double inv_lz = 1.0 / lz;

  double icell[3][3] = {{inv_lx, 0.0, 0.0},
                        {-xy * inv_lx * inv_ly, inv_ly, 0.0},
                        {(xy * yz - ly * xz) * inv_lx * inv_ly * inv_lz,
                         -yz * inv_ly * inv_lz, inv_lz}};

  // Transpose and convert to match protocol.py:
  //   icell_bohr = np.linalg.pinv(cell).T / ANGSTROM_TO_BOHR
  // The worker receives but discards icell (recv_posdata ignores it).
  double icell_t[3][3];
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) icell_t[i][j] = icell[j][i] / ANGSTROM_TO_BOHR;

  sendall(cell_t, sizeof(cell_t));
  sendall(icell_t, sizeof(icell_t));

  // Number of atoms
  int32_t n32 = natoms;
  sendall(&n32, sizeof(n32));

  // Positions in Bohr
  std::vector<double> pos(natoms * 3);
  for (int i = 0; i < 3 * natoms; i++) pos[i] = x[i] * ANGSTROM_TO_BOHR;
  sendall(pos.data(), pos.size() * sizeof(double));
}

// ---------------------------------------------------------------------------
// recv_forceready — receive energy, forces, virial from worker
// ---------------------------------------------------------------------------
void RootstockIPI::recv_forceready(int natoms, double &energy, double *forces,
                                   double virial6[6]) {
  std::string msg = recvmsg();
  if (msg != "FORCEREADY")
    error_->all(FLERR, "{}: expected FORCEREADY, got {}", style_, msg);

  // Energy in Hartree
  double energy_hartree;
  recvall(&energy_hartree, sizeof(energy_hartree));
  energy = energy_hartree * HARTREE_TO_EV;

  // Number of atoms
  int32_t natoms_recv;
  recvall(&natoms_recv, sizeof(natoms_recv));
  if (natoms_recv != natoms)
    error_->all(FLERR, "{}: natoms mismatch ({} vs {})", style_, natoms_recv,
                natoms);

  // Forces in Hartree/Bohr -> eV/Angstrom
  std::vector<double> forces_au(natoms * 3);
  recvall(forces_au.data(), forces_au.size() * sizeof(double));

  double force_conv = HARTREE_TO_EV / BOHR_TO_ANGSTROM;
  for (int i = 0; i < 3 * natoms; i++) forces[i] = forces_au[i] * force_conv;

  // Virial: 3x3 transposed in Hartree -> Voigt in eV.
  // i-PI sends column-major 3x3 (virial.T), so after receiving as row-major
  // we have the transpose. For a symmetric tensor, indices are interchangeable.
  double virial_au[9];
  recvall(virial_au, sizeof(virial_au));

  // Convert Hartree -> eV, store in LAMMPS Voigt order: xx, yy, zz, xy, xz, yz
  // (the order Pair::ev_tally and compute pressure use).
  virial6[0] = virial_au[0] * HARTREE_TO_EV;    // xx
  virial6[1] = virial_au[4] * HARTREE_TO_EV;    // yy
  virial6[2] = virial_au[8] * HARTREE_TO_EV;    // zz
  virial6[3] = virial_au[1] * HARTREE_TO_EV;    // xy
  virial6[4] = virial_au[2] * HARTREE_TO_EV;    // xz
  virial6[5] = virial_au[5] * HARTREE_TO_EV;    // yz

  // Extra bytes
  int32_t nextra;
  recvall(&nextra, sizeof(nextra));
  if (nextra > 0) {
    std::vector<char> extra(nextra);
    recvall(extra.data(), nextra);
  }
}

// ---------------------------------------------------------------------------
// Socket I/O helpers
// ---------------------------------------------------------------------------
void RootstockIPI::sendall(const void *buf, size_t len) {
  const char *p = (const char *) buf;
  size_t sent = 0;
  while (sent < len) {
    ssize_t n = ::send(client_fd_, p + sent, len - sent, 0);
    if (n <= 0) error_->all(FLERR, "{}: send failed", style_);
    sent += (size_t) n;
  }
}

void RootstockIPI::recvall(void *buf, size_t len) {
  char *p = (char *) buf;
  size_t received = 0;
  while (received < len) {
    ssize_t n = ::recv(client_fd_, p + received, len - received, 0);
    if (n <= 0)
      error_->all(FLERR, "{}: recv failed (connection closed?)", style_);
    received += (size_t) n;
  }
}

void RootstockIPI::sendmsg(const char *msg) {
  char buf[12];
  std::memset(buf, ' ', 12);
  size_t msglen = std::strlen(msg);
  if (msglen > 12) msglen = 12;
  std::memcpy(buf, msg, msglen);
  sendall(buf, 12);
}

std::string RootstockIPI::recvmsg() {
  char buf[12];
  recvall(buf, 12);
  // Trim trailing spaces
  int end = 11;
  while (end >= 0 && buf[end] == ' ') end--;
  return std::string(buf, end + 1);
}
