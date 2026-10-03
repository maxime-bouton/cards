# Changelog

## 1.0.0

Released on 2026-09-30.

### First major release

This release marks the transition to the first fully stable and "usable" version of the library. <br> We have focused heavily on simplifying the core API and making the examples much easier to understand and run.

### Major Changes
* **Core API simplification:** Completely rewrote and simplified the core library objects (e.g. models, transition kernels, estimators, etc.).
* **Improved examples:** Reworked all examples (e.g., Gaussian/Poisson deconvolution and inpainting) to be cleaner and easier to learn from.
* **Wider environment support:** Existing neural network can now be used in CPU and GPU contexts. Observations can be generated distributedly using MPI.

### Documentation & Testing
* Largely expanded documentation, including a new tutorial to help users get started.
* Expanded the PyTest suite to ensure maximal stability across both serial and MPI distributed modes for both CPU and GPU execution.
