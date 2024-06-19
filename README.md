# datastates-ai

Distributed AI model repository with versioning, lineage, and incremental storage support

## Building and Installing DStates-AI

### Automated Installation with Spack

Be sure to follow the [spack documentation]() to configure spack for your platform.
Additionally mochi packages requires configuration of libfabric to use HPC fabrics.
The mochi project has documentation on [how to do configure it for many major super computers][mochi].
Depending on your system, you may want to re-use the system provided libfabric by using a spack external package.
Here we configure mochi to use standard Ethernet to for development on laptops.

```bash
git clone https://github.com/mochi-hpc/mochi-spack-packages.git mochi_packages
git clone https://github.com/robertu94/spack_packages robertu94_packages
spack repo add ./mochi_packages
spack repo add ./robertu94_packages

# [optional] configure mochi and libfabric for your fabric
spack config edit packages

spack install dstates-ai
```

### Manual Installation

**Installing Dependencies** dstates-ai requires: the nanobind python package, MPI, mochi-thallium, and the NVIDIA CUDA toolkit.  Please refer to these projects for how to install them.  [Supported versions are documented in our spack package][dstatesspack].

DStates-AI uses a CMake-based build system.
In the ideal case, the following command is sufficent to build DStates-AI.

```bash
git clone https://github.com/datastates/dstates-ai

cmake \
    -S ./dstates-ai \
    -B ./dstates-ai-build-dir \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/path/to/install/dir \
    -DBUILD_TESTING=ON

cmake --build -j ./dstates-ai-build-dir
cmake --install ./dstates-ai-build-dir
```

For more on CMake, please refer to [the official documentation][cmake].


[spack]: https://spack.readthedocs.io/en/latest/getting_started.html
[cmake]: https://cmake.org/cmake/help/book/mastering-cmake/chapter/Getting%20Started.html
[mochi]: https://github.com/mochi-hpc-experiments/platform-configurations/
[dstatesspack]:  https://github.com/robertu94/spack_packages/blob/master/packages/datastates-ai/package.py
