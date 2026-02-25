#!/bin/bash
  
VERSION=22Jul2025
wget https://github.com/lammps/lammps/archive/refs/tags/stable_${VERSION}.tar.gz
tar zvxf stable_${VERSION}.tar.gz
cd lammps-stable_${VERSION}
mkdir build && cd build

module purge
module load gcc-toolset/14
module load aocc/5.0.0
module load aocl/aocc/5.0.0
module load openmpi/aocc-5.0.0/4.1.6
FFTW3DIR=/opt/AMD/aocl/aocl-linux-aocc-5.0.0/aocc

cmake3 \
    -D CMAKE_INSTALL_PREFIX=$HOME/.local \
    -D LAMMPS_MACHINE=d9_double_aocc \
    -D ENABLE_TESTING=no \
    -D BUILD_MPI=yes \
    -D BUILD_OMP=yes \
    -D CMAKE_CXX_COMPILER=clang++ \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_CXX_FLAGS_RELEASE="-Ofast -march=native -DNDEBUG" \
    -D PKG_KSPACE=yes \
    -D FFT=FFTW3 \
    -D FFT_SINGLE=yes \
    -D FFTW3F_INCLUDE_DIR=${FFTW3DIR}/include_LP64 \
    -D FFTW3F_LIBRARY=${FFTW3DIR}/lib_LP64/libfftw3f.so \
    -D PKG_MOLECULE=yes \
    -D PKG_RIGID=yes \
    ../cmake

make -j 8
make install
