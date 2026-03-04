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
   fix rootstock - MLIP calculator via i-PI protocol over Unix sockets

   Usage:
     fix <id> <group> rootstock cluster <name> model <model> \
         checkpoint <ckpt> device <dev> elements <e1> <e2> ...

   The worker is auto-spawned via `rootstock serve`.
------------------------------------------------------------------------- */

#include "fix_rootstock.h"

#include "atom.h"
#include "domain.h"
#include "error.h"
#include "update.h"

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

int FixRootstock::element_to_z(const std::string &symbol) {
  for (int i = 1; i <= NUM_ELEMENTS; i++) {
    if (symbol == ELEMENT_SYMBOLS[i])
      return i;
  }
  return -1;
}

// ---------------------------------------------------------------------------
// Constructor — parse keyword arguments
//   fix <id> <group> rootstock cluster <name> model <model> checkpoint <ckpt>
//       device <dev> timeout <sec> elements <e1> <e2> ...
// ---------------------------------------------------------------------------
FixRootstock::FixRootstock(LAMMPS *lmp, int narg, char **arg)
    : Fix(lmp, narg, arg), server_fd_(-1), client_fd_(-1), worker_pid_(-1),
      energy_(0.0), device_("cuda"), checkpoint_("default"), timeout_(120) {

  if (narg < 4)
    error->all(FLERR, "fix rootstock: not enough arguments");

  // Parse keyword arguments starting from arg[3]
  int iarg = 3;
  bool found_elements = false;

  while (iarg < narg) {
    std::string key = arg[iarg];

    if (key == "cluster" && iarg + 1 < narg) {
      cluster_name_ = arg[++iarg];
    } else if (key == "model" && iarg + 1 < narg) {
      model_ = arg[++iarg];
    } else if (key == "checkpoint" && iarg + 1 < narg) {
      checkpoint_ = arg[++iarg];
    } else if (key == "device" && iarg + 1 < narg) {
      device_ = arg[++iarg];
    } else if (key == "timeout" && iarg + 1 < narg) {
      timeout_ = std::atoi(arg[++iarg]);
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
  if (cluster_name_.empty())
    error->all(FLERR, "fix rootstock: 'cluster' keyword is required");
  if (model_.empty())
    error->all(FLERR, "fix rootstock: 'model' keyword is required");
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
    int z = element_to_z(sym);
    if (z < 0)
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
// Destructor — send EXIT, kill worker, close sockets, unlink socket file
// ---------------------------------------------------------------------------
FixRootstock::~FixRootstock() {
  // Best-effort EXIT message
  if (client_fd_ >= 0) {
    char buf[12];
    std::memset(buf, ' ', 12);
    std::memcpy(buf, "EXIT", 4);
    ::send(client_fd_, buf, 12, MSG_NOSIGNAL);
    ::close(client_fd_);
  }
  if (server_fd_ >= 0)
    ::close(server_fd_);

  // Clean up worker process
  if (worker_pid_ > 0) {
    ::kill(worker_pid_, SIGTERM);
    int status;
    // Give worker up to 5 seconds to exit
    for (int i = 0; i < 50; i++) {
      if (::waitpid(worker_pid_, &status, WNOHANG) != 0)
        break;
      usleep(100000); // 100ms
    }
    // Force kill if still alive
    if (::waitpid(worker_pid_, &status, WNOHANG) == 0) {
      ::kill(worker_pid_, SIGKILL);
      ::waitpid(worker_pid_, &status, 0);
    }
  }

  // Clean up socket file
  if (!socket_path_.empty())
    ::unlink(socket_path_.c_str());
}

// ---------------------------------------------------------------------------
// setmask — we operate in post_force
// ---------------------------------------------------------------------------
int FixRootstock::setmask() { return FixConst::POST_FORCE; }

// ---------------------------------------------------------------------------
// resolve_cluster — call `rootstock resolve --cluster <name> --json`
// ---------------------------------------------------------------------------
std::string FixRootstock::resolve_cluster(const std::string &cluster) {
  std::string cmd = "rootstock resolve --cluster " + cluster + " --json";
  FILE *pipe = popen(cmd.c_str(), "r");
  if (!pipe)
    error->all(FLERR, "fix rootstock: failed to run 'rootstock resolve'. "
                       "Is rootstock installed? (pip install rootstock)");

  char buffer[1024];
  std::string output;
  while (fgets(buffer, sizeof(buffer), pipe))
    output += buffer;

  int status = pclose(pipe);
  if (status != 0)
    error->all(FLERR, "fix rootstock: 'rootstock resolve --cluster {}' failed. "
                       "Unknown cluster?",
               cluster);

  // Parse "root" from JSON output: {"root": "/path/...", "cluster": "..."}
  // Simple string search — no JSON library needed for this minimal format
  std::string key = "\"root\": \"";
  auto pos = output.find(key);
  if (pos == std::string::npos)
    error->all(FLERR, "fix rootstock: failed to parse 'rootstock resolve' output");

  pos += key.size();
  auto end = output.find('"', pos);
  if (end == std::string::npos)
    error->all(FLERR, "fix rootstock: failed to parse 'rootstock resolve' output");

  return output.substr(pos, end - pos);
}

// ---------------------------------------------------------------------------
// spawn_worker — fork + execlp rootstock serve
// ---------------------------------------------------------------------------
pid_t FixRootstock::spawn_worker(const std::string &root) {
  pid_t pid = fork();
  if (pid < 0)
    error->all(FLERR, "fix rootstock: fork() failed");

  if (pid == 0) {
    // Child: exec rootstock serve
    execlp("rootstock", "rootstock", "serve", model_.c_str(), "--root",
           root.c_str(), "--socket", socket_path_.c_str(), "--checkpoint",
           checkpoint_.c_str(), "--device", device_.c_str(), nullptr);
    // If exec fails, exit immediately
    _exit(127);
  }
  return pid; // Parent: return child PID
}

// ---------------------------------------------------------------------------
// init — resolve cluster, create socket, spawn worker, accept, handshake
// ---------------------------------------------------------------------------
void FixRootstock::init() {
  // Validate units
  if (std::string(update->unit_style) != "metal")
    error->all(FLERR, "fix rootstock requires 'units metal'");

  // Only do socket/worker setup on first call.
  // LAMMPS calls init() at the start of every `run` command.
  if (server_fd_ >= 0)
    return;

  // Resolve cluster root directory
  std::string root = resolve_cluster(cluster_name_);

  // Generate unique socket path
  socket_path_ = "/tmp/rootstock_" + std::to_string(getpid()) + "_" +
                  std::string(id) + ".sock";

  // Create Unix domain socket
  server_fd_ = ::socket(AF_UNIX, SOCK_STREAM, 0);
  if (server_fd_ < 0)
    error->all(FLERR, "fix rootstock: socket() failed");

  struct sockaddr_un addr;
  std::memset(&addr, 0, sizeof(addr));
  addr.sun_family = AF_UNIX;
  if (socket_path_.size() >= sizeof(addr.sun_path))
    error->all(FLERR, "fix rootstock: socket path too long");
  std::strncpy(addr.sun_path, socket_path_.c_str(), sizeof(addr.sun_path) - 1);

  // Remove stale socket file
  ::unlink(socket_path_.c_str());

  if (::bind(server_fd_, (struct sockaddr *)&addr, sizeof(addr)) < 0)
    error->all(FLERR, "fix rootstock: bind() failed on {}", socket_path_);

  if (::listen(server_fd_, 1) < 0)
    error->all(FLERR, "fix rootstock: listen() failed");

  // Spawn the worker
  worker_pid_ = spawn_worker(root);

  // Accept connection with configurable timeout
  fd_set fds;
  FD_ZERO(&fds);
  FD_SET(server_fd_, &fds);
  struct timeval tv;
  tv.tv_sec = timeout_;
  tv.tv_usec = 0;

  int sel = ::select(server_fd_ + 1, &fds, nullptr, nullptr, &tv);
  if (sel <= 0)
    error->all(FLERR,
               "fix rootstock: no worker connected within {} seconds. "
               "Worker may have failed to start — check that the '{}' "
               "environment is built.",
               timeout_, model_);

  client_fd_ = ::accept(server_fd_, nullptr, nullptr);
  if (client_fd_ < 0)
    error->all(FLERR, "fix rootstock: accept() failed");

  // Build atomic_numbers_ from LAMMPS atom types
  int nlocal = atom->nlocal;
  int *type = atom->type;
  atomic_numbers_.resize(nlocal);
  for (int i = 0; i < nlocal; i++) {
    atomic_numbers_[i] = element_to_z(elements_[type[i] - 1]);
  }

  // Warn about atom types with zero atoms
  for (int t = 1; t <= atom->ntypes; t++) {
    int count = 0;
    for (int i = 0; i < nlocal; i++)
      if (type[i] == t)
        count++;
    if (count == 0)
      error->warning(FLERR, "fix rootstock: atom type {} ({}) has no atoms", t,
                     elements_[t - 1]);
  }

  // INIT handshake: STATUS -> NEEDINIT -> INIT -> STATUS -> READY
  send_status();
  std::string status = recv_status();
  if (status != "NEEDINIT")
    error->all(FLERR, "fix rootstock: expected NEEDINIT, got {}", status);

  send_init();

  send_status();
  status = recv_status();
  if (status != "READY")
    error->all(FLERR, "fix rootstock: expected READY after INIT, got {}",
               status);
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
  int *type = atom->type;

  // Rebuild atomic_numbers_ if atom count changed
  if ((int)atomic_numbers_.size() != nlocal) {
    atomic_numbers_.resize(nlocal);
    for (int i = 0; i < nlocal; i++) {
      atomic_numbers_[i] = element_to_z(elements_[type[i] - 1]);
    }
  }

  // STATUS -> check state
  send_status();
  std::string status = recv_status();

  // Worker returns to NEEDINIT after each FORCEREADY, so re-send INIT
  if (status == "NEEDINIT") {
    send_init();
    send_status();
    status = recv_status();
  }

  if (status != "READY")
    error->all(FLERR, "fix rootstock post_force: expected READY, got {}",
               status);

  // Send positions
  send_posdata();

  // STATUS -> HAVEDATA
  send_status();
  status = recv_status();
  if (status != "HAVEDATA")
    error->all(FLERR, "fix rootstock post_force: expected HAVEDATA, got {}",
               status);

  // GETFORCE -> FORCEREADY
  sendmsg("GETFORCE");
  recv_forceready();
}

// ---------------------------------------------------------------------------
// compute_scalar — return cached energy for thermo output
// ---------------------------------------------------------------------------
double FixRootstock::compute_scalar() { return energy_; }

// ---------------------------------------------------------------------------
// send_init — send INIT message with JSON species data
// ---------------------------------------------------------------------------
void FixRootstock::send_init() {
  // Build JSON: {"numbers": [29, 29, ...], "pbc": [true, true, true]}
  std::ostringstream json;
  json << "{\"numbers\": [";
  for (size_t i = 0; i < atomic_numbers_.size(); i++) {
    if (i > 0)
      json << ", ";
    json << atomic_numbers_[i];
  }
  json << "], \"pbc\": [true, true, true]}";
  std::string init_str = json.str();

  sendmsg("INIT");

  // bead index (int32)
  int32_t bead = 0;
  sendall(&bead, sizeof(bead));

  // init string length (int32) + bytes
  int32_t nbytes = (int32_t)init_str.size();
  sendall(&nbytes, sizeof(nbytes));
  sendall(init_str.data(), init_str.size());
}

// ---------------------------------------------------------------------------
// send_posdata — send cell + positions in atomic units
// ---------------------------------------------------------------------------
void FixRootstock::send_posdata() {
  sendmsg("POSDATA");

  // LAMMPS box: lower-triangular cell
  //   [[lx,  0,  0],
  //    [xy, ly,  0],
  //    [xz, yz, lz]]
  double lx = domain->boxhi[0] - domain->boxlo[0];
  double ly = domain->boxhi[1] - domain->boxlo[1];
  double lz = domain->boxhi[2] - domain->boxlo[2];
  double xy = domain->xy;
  double xz = domain->xz;
  double yz = domain->yz;

  // Cell matrix (row-major, lower-triangular)
  double cell[3][3] = {{lx, 0.0, 0.0}, {xy, ly, 0.0}, {xz, yz, lz}};

  // Transpose and convert to Bohr for i-PI column-major convention
  double cell_t[3][3];
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      cell_t[i][j] = cell[j][i] * ANGSTROM_TO_BOHR;

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
    for (int j = 0; j < 3; j++)
      icell_t[i][j] = icell[j][i] / ANGSTROM_TO_BOHR;

  sendall(cell_t, sizeof(cell_t));
  sendall(icell_t, sizeof(icell_t));

  // Number of atoms
  int32_t nlocal = atom->nlocal;
  sendall(&nlocal, sizeof(nlocal));

  // Positions in Bohr
  double **x = atom->x;
  std::vector<double> pos(nlocal * 3);
  for (int i = 0; i < nlocal; i++) {
    pos[3 * i + 0] = x[i][0] * ANGSTROM_TO_BOHR;
    pos[3 * i + 1] = x[i][1] * ANGSTROM_TO_BOHR;
    pos[3 * i + 2] = x[i][2] * ANGSTROM_TO_BOHR;
  }
  sendall(pos.data(), pos.size() * sizeof(double));
}

// ---------------------------------------------------------------------------
// recv_forceready — receive energy, forces, virial from worker
// ---------------------------------------------------------------------------
void FixRootstock::recv_forceready() {
  std::string msg = recvmsg();
  if (msg != "FORCEREADY")
    error->all(FLERR, "fix rootstock: expected FORCEREADY, got {}", msg);

  // Energy in Hartree
  double energy_hartree;
  recvall(&energy_hartree, sizeof(energy_hartree));
  energy_ = energy_hartree * HARTREE_TO_EV;

  // Number of atoms
  int32_t natoms;
  recvall(&natoms, sizeof(natoms));

  int nlocal = atom->nlocal;
  if (natoms != nlocal)
    error->all(FLERR, "fix rootstock: natoms mismatch ({} vs {})", natoms,
               nlocal);

  // Forces in Hartree/Bohr -> eV/Angstrom
  std::vector<double> forces_au(natoms * 3);
  recvall(forces_au.data(), forces_au.size() * sizeof(double));

  double force_conv = HARTREE_TO_EV / BOHR_TO_ANGSTROM;
  double **f = atom->f;
  for (int i = 0; i < natoms; i++) {
    f[i][0] += forces_au[3 * i + 0] * force_conv;
    f[i][1] += forces_au[3 * i + 1] * force_conv;
    f[i][2] += forces_au[3 * i + 2] * force_conv;
  }

  // Virial: 3x3 transposed in Hartree -> Voigt in eV
  // i-PI sends column-major 3x3 (virial.T), so after receiving as row-major
  // we have the transpose. For a symmetric tensor, indices are interchangeable.
  double virial_au[9];
  recvall(virial_au, sizeof(virial_au));

  // Convert Hartree -> eV, store in LAMMPS Voigt order: xx, yy, zz, yz, xz, xy
  virial[0] = virial_au[0] * HARTREE_TO_EV; // xx
  virial[1] = virial_au[4] * HARTREE_TO_EV; // yy
  virial[2] = virial_au[8] * HARTREE_TO_EV; // zz
  virial[3] = virial_au[5] * HARTREE_TO_EV; // yz
  virial[4] = virial_au[2] * HARTREE_TO_EV; // xz
  virial[5] = virial_au[1] * HARTREE_TO_EV; // xy

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
void FixRootstock::sendall(const void *buf, size_t len) {
  const char *p = (const char *)buf;
  size_t sent = 0;
  while (sent < len) {
    ssize_t n = ::send(client_fd_, p + sent, len - sent, 0);
    if (n <= 0)
      error->all(FLERR, "fix rootstock: send failed");
    sent += (size_t)n;
  }
}

void FixRootstock::recvall(void *buf, size_t len) {
  char *p = (char *)buf;
  size_t received = 0;
  while (received < len) {
    ssize_t n = ::recv(client_fd_, p + received, len - received, 0);
    if (n <= 0)
      error->all(FLERR, "fix rootstock: recv failed (connection closed?)");
    received += (size_t)n;
  }
}

void FixRootstock::sendmsg(const char *msg) {
  char buf[12];
  std::memset(buf, ' ', 12);
  size_t msglen = std::strlen(msg);
  if (msglen > 12)
    msglen = 12;
  std::memcpy(buf, msg, msglen);
  sendall(buf, 12);
}

std::string FixRootstock::recvmsg() {
  char buf[12];
  recvall(buf, 12);
  // Trim trailing spaces
  int end = 11;
  while (end >= 0 && buf[end] == ' ')
    end--;
  return std::string(buf, end + 1);
}

void FixRootstock::send_status() { sendmsg("STATUS"); }

std::string FixRootstock::recv_status() { return recvmsg(); }
