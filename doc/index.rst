cards documentation
===================

The ``cards`` (Composable Algorithms for Reproducible Distributed Sampling) Python library provides elementary operators, MPI communicators and Markov transition kernels to facilitate the design of custom distributed Plug-and-Play (PnP) Markov chain Monte Carlo (MCMC) algorithms for high-dimensional Bayesian inference.
The distributed functionalitites proposed in this library are primarily oriented towards SMPD algorithms running on multiple CPUs or GPUs.

The `associated github repository <https://github.com/maxime-bouton/cards/>`_ also contains codes to reproduce the image processing experiments reported in :cite:p:`Bouton2025`.


.. warning::

   This project is under active development, and the API my evolve significantly until version ``1.0``.


.. toctree::
   :maxdepth: 1
   :caption: Installation

   setup
   contributing
   biblio


.. toctree::
   :maxdepth: 1
   :caption: Development
   
   api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
