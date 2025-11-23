Setup
=====

Installation
------------

.. _environment-setup:

Environment setup
^^^^^^^^^^^^^^^^^

The library requires a functional installation of `conda <https://docs.anaconda.com/free/anaconda/install/index.html>`_ or `mamba <https://mamba.readthedocs.io/en/latest/installation.html>`_ (a faster drop-in replacement of `conda`, which only implies replacing commands of the form ``conda ...`` by ``mamba ...``). The instructions given below are given on the case of ``mamba``.

To install the library, issue the following commands in a terminal.

.. code-block:: bash

    # setup instruction
    # TODO: to complete

To avoid `file lock issue in h5py <https://github.com/h5py/h5py/issues/1101>`_, add the following line to the ``~/.zshrc`` file (or ``~/.bashrc``)

.. code-block:: bash

    export HDF5_USE_FILE_LOCKING='FALSE'

To test the installation went well, you can run the unit-tests provided in the package using the command below.

.. code-block:: bash

    conda activate jcgs-review
    pytest --collect-only
    export NUMBA_DISABLE_JIT=1  # need to disable jit compilation to check test coverage
    coverage run -m pytest      # run all the unit-tests (see Documentation section for more details)


Experiments
-----------

All the experiments reported in the paper can be reproduced from the ``.sh`` scripts provided in ``./examples/jcgs``, as detailed in the following paragraphs.

Memory requirements to save all the results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Running all the experiments produces a large volume of results / checkpoint data saved to the hard-drive in HDF5 (``.h5``) format, essentially for the experiments involving serial samplers (206GB in total).

All the samples generated with serial samplers are saved to disk to better assess the sampling quality, which explains such a large requirement. Ensure maximum 20GB hard-drive memory is available for each single run of one of the serial samplers on one of the datasets considered.

Users are strongly advised to progressively run a subset of the experiments at once, carefully monitoring the remaining space available on the hard-drive where results are saved. The checkpoint files corresponding to burn-in samples can be safely discarded once the experiment is finalized.

Each run of the distributed sampler on a dataset requires 300MB memory on the hard drive, as fewer elements are saved to disk (last state to restart the chain + average over the current batch to compute the MMSE estimator).

Running experiments in a detached tmux session
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The experiments can be run from a detached `tmux <https://github.com/tmux/tmux/wiki>`_ session running in the background. See the ``./examples/jcgs/run_from_tmux.sh`` script for further details.

A few basic instructions to interact with a ``tmux`` session are given below.

.. code-block:: bash

    # check name of the tmux session, called session_name in the following
    tmux list-session
    tmux a -t session_name # press crtl+b to kill the session once the work is done

    # to detach from a session (leave it running in the background, press ctrl+b, then d)
    # in the tmux session, press crtl+d to kill it once the work is done or, from a normal terminal
    tmux kill-session -t session_name


Running the experiments
^^^^^^^^^^^^^^^^^^^^^^^

The folder ``./examples/jcgs/configs`` contains ``.json`` files summarizing the list of parameters used for the different experiments/datasets. All the experiments can be reproduced using the following commands. Details about the location of the HDF5 files / data produced are included in each script.

.. code-block:: bash

    # from a terminal at the root of the archive

    mamba activate jcgs-review
    cd examples/jcgs

    # generate all the synthetic datasets used in the experiments (to be run only once)
    bash generate_data.sh

    # run all the experiments based on serial samplers (MYULA and proposed sampler)
    bash sampling_quality_experiment.sh

    # run strong scaling experiment
    bash strong_scaling_experiment.sh

    # run weak scaling experiment
    bash weak_scaling_experiment.sh

    # deactivating the conda environment (when no longer needed)
    mamba deactivate


The content of an ``.h5``  file can be quickly checked from the terminal (`reference <https://docs.h5py.org/en/stable/mpi.html?highlight=h5dump#using-parallel-hdf5-from-h5py>`_). Some examples are provided below.

.. code-block:: bash

    # replace <filename> by the name of your file
    h5dump --header <filename>.h5 # displays the name and size of all variables contained in the file
    conda activate jcgs-review

    # replace <filename> by the name of your file in the instructions below
    h5dump --header <filename>.h5 # displays the name and size of all variables contained in the file
    h5dump <filename>.h5 # diplays the value of all the variables saved in the file
    h5dump -d "/GroupFoo/databar[1,1;2,3;3,19;1,1]" <filename>.h5 # display part of a variable from a dataset within a given h5 file
    h5dump -d dset <filename>.h5 # displays content of a dataset dset
    h5dls -d <filename>.h5/dset  # displays content of a dataset dset
    du -sh .  # disk usage for the current user (in current directory)


Please check the ``h5py`` `documentation <https://docs.h5py.org/en/stable/quick.html>`_ for further details.


Development
-----------

Building the documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^

Most functionalities are fully documented using the ``numpy`` docstring style.

The documentation can be generated in ``.html`` format using the following commands issued from a terminal.

.. code-block:: bash

    conda activate jcgs-review
    cd build docs/build/html
    # compile documentation in html, latex or linkcheck
    make html


Assessing code and docstring coverage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To test the code/docstring coverage, run the following commands from a terminal.

.. tab-set::
    .. tab-item:: Running all tests

        .. code-block:: bash

            conda activate jcgs-review
            # need to disable jit compilation to check test coverage
            export NUMBA_DISABLE_JIT=1
            # check all tests
            coverage run -m unittest
            # generate a coverage report in the terminal
            coverage report
            # HTML-based reports showing what lines of code were not tested
            coverage html
            # produce xml file to generate the badge
            coverage xml -o reports/coverage/coverage.xml
            genbadge coverage -o docs/coverage.svg
            # check docstring coverage and generate the associated badge
            docstr-coverage .

    .. tab-item:: Running a single test

        .. code-block:: bash

            conda activate jcgs-review
            python -m pytest tests/operators/test_crop.py
            pytest --markers  # check full list of markers availables
            # run all tests not marked as mpi + ignore some files
            pytest -m "not mpi" --ignore-glob=**/archive_unittest/*
            # run all tests marked mpi with 2 cores
            mpiexec -n 2 python -m mpi4py -m pytest -m mpi
