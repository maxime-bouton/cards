Contributing
============

.. _environment-setup:

Environment setup
-----------------

Contributing to the library requires a functional installation of `pixi <https://pixi.prefix.dev/latest/>`_.

To configure the workspace after cloning the project from github, issue the following commands in a terminal from the root project directory.

.. code-block:: bash

    pixi self-update
    pixi clean
    pixi clean cache
    pixi install --environment full
    pixi shell --environment full


Testing
-------

Running tests
^^^^^^^^^^^^^

Before any commit or pull request to the master branch, verify all tests pass under the different configuration considered.

.. tab-set::
    .. tab-item:: Running all tests

        .. code-block:: bash

            pixi shell -e full

            # display available markers
            pytest --markers

            # list all tests available
            python -m pytest --collect-only

            # running all serial tests on CPU
            python -m pytest --mode serial --device cpu

            # running all serial tests on GPU
            python -m pytest --mode serial --device gpu

            # running all MPI tests on CPU
            mpiexec -n 2 python -m mpi4py -m pytest --mode mpi --device cpu

            # running all MPI tests on GPU
            mpiexec -x OMPI_MCA_pml=ucx -x OMPI_MCA_osc=ucx -x OMPI_MCA_opal_cuda_support=true -x UCX_MEMTYPE_CACHE=n -np 2 python -m pytest --mode mpi --device gpu

    .. tab-item:: Running a single test
        .. code-block:: bash

            pixi shell -e full

            # example serial test on CPU
            python -m pytest --mode serial --device cpu tests/operators/test_dft_convolution.py

            # example MPI test on GPU
            mpiexec -x OMPI_MCA_pml=ucx -x OMPI_MCA_osc=ucx -x OMPI_MCA_opal_cuda_support=true -x UCX_MEMTYPE_CACHE=n -np 2 python -m pytest --mode mpi --device gpu tests/operators/test_dft_convolution.py


Assessing code and docstring coverage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Code and docstring coverage can be evaluated locally with the following commands issued in a terminal from the root project directory.

.. code-block:: bash

    pixi shell -e full
    # need to disable jit compilation to check test coverage
    export NUMBA_DISABLE_JIT=1

    python -m pytest --collect-only
    # check all tests
    coverage run -m pytest
    # generate a coverage report in the terminal
    coverage report
    # HTML-based reports showing what lines of code were not tested
    coverage html
    # produce xml file to generate the badge
    coverage xml -o reports/coverage/coverage.xml
    genbadge coverage -o docs/coverage.svg
    # check docstring coverage and generate the associated badge
    docstr-coverage .


.. Continuous integration
.. ^^^^^^^^^^^^^^^^^^^^^^

.. To be completed


Packaging
---------

Building the documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^

Documentation follows the ``numpy`` docstring style.
Starting from scratch, the documentation can be built using `sphinx-autoapi <https://sphinx-autoapi.readthedocs.io/en/latest/tutorials.html>`_.

.. code-block:: bash

    cd doc
    # sphinx-build [OPTIONS] SOURCEDIR OUTPUTDIR [FILENAMES...]
    sphinx-build -b html . _build

To deactivate automatic API update and generation, update the field in ``doc/conf.py`` `as follows <https://sphinx-autoapi.readthedocs.io/en/latest/how_to.html#how-to-transition-to-manual-documentation>`_.

.. code-block:: python

    autoapi_generate_api_docs = False
    autoapi_keep_files = False


Building and publishing the conda package
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- The package can be simply built using the following command, producing a ``.conda`` package.

    .. code-block:: bash

        pixi shell -e full
        pixi build

- To locally verify the build step is successful, one may create a new test workspace with ``pixi`` (``pixi init . --format pyproject```), and modify the associated ``pyproject.toml`` as follows

    .. code-block:: toml

        [dependencies]
        cards = { path = "</PATH/TO/PACKAGE_NAME>.conda" }

- The package can be published to an `anaconda.org <https://www.anaconda.com/docs/tools/anaconda-org/maintainer-guide/upload-packages>`_ channel as follows

    .. code-block:: bash

        anaconda login
        anaconda upload <CHANNEL>::</PATH/TO/PACKAGE_NAME>.conda
