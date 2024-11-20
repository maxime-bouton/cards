# Python-MCMC

[![license](https://img.shields.io/badge/license-GPL--3.0-brightgreen.svg)](LICENSE)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
![Python](https://img.shields.io/badge/python-3670A0?style=flat&logo=python&logoColor=ffdd54)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Note: two environments may be temporarily needed during development, one with `numpy>=2.0`, another `pythorch>=2.5`.

The use of random number generators on GPU seems to be incompatible with `numpy`, and currently needs to be investigated.

## Setup

- To create an environment from the provided `.yml` file

  ```bash
  mamba env create --file gpu-linux-64.yml
  conda develop src
  ```

- Checking everything went well using a multi-GPU test (or Hello-world file)

  ```bash
  mpirun -x OMPI_MCA_pml=ucx \
      -x OMPI_MCA_osc=ucx \
      -x OMPI_MCA_opal_cuda_support=true \
      -x UCX_MEMTYPE_CACHE=n \
      -np 2 pytest --with-mpi test.py
  # mpiexec --mca pml ucx --mca osc ucx --mca coll_ucc_enable 1 --mca opal_cuda_support 1 -x UCX_MEMTYPE_CACHE=n -n 2 test.py...
  ```

- Manual installation using latest compatible versions of the packages is detailed below.

<details>

<summary>Ubuntu</summary>

### Ubuntu

Instructions tested on `epeautre`.

```bash
mamba create -n pymcmc python=3.12
mamba activate pymcmc
mamba install cuda # or nvidia::cuda
which nvcc # should be located in miniforge folder or so
mamba install cupy ipykernel ipyparallel
# mamba install openmpi ucx mpi4py  # currently, issues with mpi4py openmpi
mamba install mpich mpi4py ucx
mamba install matplotlib scikit-image

# choose the build of h5py including "mpi_openmpi_py\<>" with the right python version
# mamba search h5py
mamba install "h5py>=2.9=mpi*"

# mamba install pytorch
pip install torch  # ok for 2.5.1 version, not the case yet with mamba
mamba install numba  # issue with numba for now (need more recent numpy?)
mamba install pytest pre-commit ruff conda-build
mamba install pytest-cov
pip install docstr-coverage

# export manual configuration in a .yml file
mamba env export --name pymcmc --file gpu-mpich-linux-64.yml

conda develop src
```

</details>

<details>

<summary>OSX</summary>

### OSX

All features relying on `cupy` are not supported on MAC for now.

```bash
mamba create -n pymcmc
mamba activate pymcmc
mamba install mpich mpi4py  # openmpi
mamba install "h5py>=2.9=mpi*"
mamba install numpy numba pytorch
mamba install matplotlib scikit-image
mamba install pytest pre-commit ruff conda-build
# mamba install pytest-cov
pip install docstr-coverage

conda develop src

# export manual configuration in a .yml file
mamba env export --name pymcmc --file osx-64.yml
```

</details>

## Running tests

## Assessing code and docstring coverage

To test the code/docstring coverage locally, run the following commands

```bash
mamba activate pymcmc
python -m pytest --collect-only
export NUMBA_DISABLE_JIT=1 # need to disable jit compilation to check test coverage
coverage run -m pytest # check all tests
coverage report # generate a coverage report in the terminal
coverage html # HTML-based reports which let you visually see what lines of code were not tested
coverage xml -o reports/coverage/coverage.xml # produce xml file to generate the badge
genbadge coverage -o docs/coverage.svg
docstr-coverage . # check docstring coverage and generate the associated badge
```

To launch a single test, run a command of the form

```bash
mamba activate pymcmc
python -m pytest tests/test_module.py
pytest --markers  # check full list of markers availables
pytest -m "not mpi" --ignore-glob=**/floder_to_ignore/* # run all tests not marked as mpi + ignore files in any directory "/floder_to_ignore/"
mpiexec -n 2 python -m mpi4py -m pytest -m mpi  # run all tests marked mpi with 2 cores
mpiexec -n 2 python -m pytest tests/test_warmstart_rng_mpi.py
```
