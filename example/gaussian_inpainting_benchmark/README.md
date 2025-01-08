# Notes on environment setup for the benchmark

## Installing mamba as a user

On `zeus`, following the [mamba installation guide](https://github.com/conda-forge/miniforge)

```bash
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh

eval "$(/home/pierreantoine.thouvenin/miniforge3/bin/conda shell.YOUR_SHELL_NAME hook)"
mamba init
```

## Multi-GPU environment (pytorch + cupy)

- Manual installation on `zeus` used for benchmarks, working on `epeautre` as well

```bash
# https://stackoverflow.com/questions/62359175/pytorch-says-that-cuda-is-not-available-on-ubuntu
mamba create --name mcmc numba numpy openmpi ucx cuda cudatoolkit cupy pytorch torchvision -c pytorch -c conda-forge -c nvidia
mamba activate mcmc

# mamba install -c conda-forge cuda-nvcc cuda-nvrtc "cuda-version>=12.0"
# mamba install cuda-cudart cuda-version=12
# mamba install nccl

mamba install mpi4py openmpi "h5py>=2.9=mpi*" scipy imageio tqdm conda-build matplotlib
# mamba install scikit-image -c conda-forge  # ! this downgrades numpy for now and imposes cpu version of pytorch: using pip install instead for now
pip install scikit-image
# mamba install mpi4jax
# mamba install ipykernel ipyparallel # for notebooks

# optional install (contributing: code formatting and building documentation)
mamba install pytest isort coverage pre-commit furo sphinx sphinx_rtd_theme sphinxcontrib-bibtex sphinx-autoapi sphinx-design
pip install sphinxcontrib-apa sphinx_copybutton docstr-coverage genbadge wily

# installing package in development mode
conda develop src

# export
mamba env export > zeus_linux_`uname -m`_environment.yml
```

<!-- - Manual installation based on `mpich`, tested on `epeautre`

```bash
mamba create --name mcmc python=3.12 -y  # (3.12 required for cupy as of 15/12/2024)
mamba activate mcmc

# GPU packages
mamba install mpich ucx -y # openmpi ucx
# from mpi4py documentation
# installing mpi4py with mpich build (openmpi seems to create issues)
# mamba install mpi4py mpich -y
mamba install mpi4py mpich -y
mamba install cupy -y
mamba install pytorch -y
# mamba install mpi4jax
# mamba install ipykernel ipyparallel # for notebooks

mamba install "h5py>=2.9=mpi*" numpy numba scipy scikit-image imageio -y
mamba install tqdm furo conda-build -y

# optional install (contributing: code formatting and building documentation)
mamba install matplotlib -y
mamba install pytest isort coverage pre-commit sphinx sphinx_rtd_theme sphinxcontrib-bibtex sphinx-autoapi sphinx-design -y
pip install sphinxcontrib-apa sphinx_copybutton docstr-coverage genbadge wily

# installing package in development mode
conda develop src
``` -->

- exporting to `.yml` file

```bash
mamba env export > zeus_linux_`uname -m`_environment.yml
# mamba env export --from-history | grep -v "^prefix" > gpu_dsgs_linux_`uname -m`_environment.yml
# conda env export --no-builds > direct_environment.yml
```

- installing packages from existing `.yml` configuration file (to be tested on `zeus`)

```bash
cd examples/benchmark
mamba env create --name mcmc -f zeus_linux_`uname -m`_environment.yml
cd ../../
conda develop src
```

- remark: indications for pytorch with cuda support taken from this [post](https://stackoverflow.com/questions/76376486/how-to-install-pytorch-with-cuda-support-using-conda), a pre-requisite for multi-GPU `torch`

```bash
conda install pytorch-gpu torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

## Running a multi-GPU MPI script

- Multi-GPU scripts require the [Unified Communication X (UCX)](https://openucx.readthedocs.io/en/master/index.html), see the [documentation to activate cuda support](https://openucx.readthedocs.io/en/master/running.html) with either `openmpi` or `mpich`, see also [here](<https://github.com/pmodels/mpich/wiki/MPICH-CH4:UCX-with-CUDA-support-(3.3.x)>) and [there](https://docs.open-mpi.org/en/v5.0.x/tuning-apps/networking/cuda.html)

```bash
# with openmpi
mpirun -np 2 -mca pml ucx -mca btl ^uct -x UCX_NET_DEVICES=mlx5_0:1 ./app

# with mpich
mpirun -np 2 -env UCX_NET_DEVICES=mlx5_0:1 ./executable
```

- Running a multi-GPU test can be done in the terminal as follows

```bash
# to run the test, make sure the following are installed
# mamba install conda-build pytest ipykernel ipyparallel
# conda develop src

# https://github.com/openucx/ucx/issues/5284
# https://docs.open-mpi.org/en/v5.0.x/tuning-apps/networking/cuda.html

mpirun -x OMPI_MCA_pml=ucx \
       -x OMPI_MCA_osc=ucx \
       -x OMPI_MCA_opal_cuda_support=true \
       -x UCX_MEMTYPE_CACHE=n \
       -np 4 pytest --with-mpi test.py
# mpiexec --mca opal_cuda_support 1 ...
```

## Testing the library

To test the code/docstring coverage locally, run the following commands

```bash
mamba activate mcmc
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
mamba activate mcmc
python -m pytest tests/operators/test_crop.py
pytest --markers  # check full list of markers availables
pytest -m "not mpi" --ignore-glob=**/archive_unittest/* # run all tests not marked as mpi + ignore files in any directory "archive_unittest"
mpiexec -n 2 python -m mpi4py -m pytest -m mpi  # run all tests marked mpi with 2 cores
```

To measure code quality with `wily`

```bash
wily report mcmc -f HTML -o docs/build/wily_report.html
```

## Compiling and publishing the documentation

Add a `worktree` from the `master` branch to be able to push the documentation, once built locally.

```bash
# make sure the folder html does not exist before running the command
git worktree add docs/build/html gh-pages
cd docs/build/html
git add .
git commit -m "Build documentation as of $(git log '--format=format:%H' master -1)"
git push origin gh-pages
# delete the worktree
cd ../
git worktree remove html
```

## Slurm notes for zeus

- Full documentation [online](https://hpc-doc.univ-lille.fr/docs/accueil/)

- `gres` option, standing for `consumable generic resources`, mostly used to select nodes with GPUs, e.g.: `#SBATCH --gres=gpu:2` (booking nodes with 2 GPUs each)

- displaying the list of `gres`: `sbatch --gres=help`

- Canceling jobs: `scancel`

  - scancel JOBID permet d'annuler le job JOBID.
  - scancel -n toto annule tous vos jobs nommés toto.
  - scancel -n toto -t PENDING annule les jobs nommés toto en attente.
  - scancel -u user.login annule tous les jobs de l'utilisateur

- `sacct`: info on passed jobs, useful to estimate memory for future job if out of memory issue, using

  - `sacct -o jobid,jobname,reqnodes,reqcpus,reqmem,maxrss,averss,elapsed -j JOBID`
  - `sacct -o jobid,jobname,reqnodes,reqcpus,reqmem,maxrss,averss,elapsed -S YYYY-MM-DD`

- Checking resources (GRES, ...), see [reference](https://ask.cyberinfrastructure.org/t/how-do-i-get-the-list-of-features-and-resources-of-each-node-in-slurm/201)

```bash
sinfo -o "%20N  %10c  %10m  %25f  %10G "
```

Displaying options with `sinfo --help`. In particular `sinfo -o` specifies the format of the output, and the options above are short for

- N = node name
- c = number of cores
- m = memory
- f = features, often it will be the architecture or type of associated gpu
- G = gres type and number, e.g. gpu:2

The %20 means 20 characters for this field. For example, on may want to separate fields with |, and so the command would be
`sinfo -o “|%20N | %10c | %10m | %25f | %10G|”`.
