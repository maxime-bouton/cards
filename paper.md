---
title: 'CARDS: Composable Algorithms for Reproducible Distributed Sampling'
tags:
  - Python
  - Markov chain Monte Carlo algorithms
  - Langevin algorithm
  - Plug-and-Play prior
  - distributed computing

authors:
  - name: Maxime Bouton
    orcid: 0009-0007-4480-3541
    equal-contrib: true
    affiliation: 1
  - name: Pierre-Antoine Thouvenin
    orcid: 0000-0003-1246-9458
    corresponding: true # (This is how to denote the corresponding author)
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 1
  - name: Stéphane Despierres
    affiliation: 1
  - name: Audrey Repetti
    orcid: 0000-0002-6296-6957
    affiliation: "2, 3" # (Multiple affiliations must be quoted)
  - name: Pierre Chainais
    orcid: 0000-0003-4377-7584
    affiliation: 1
affiliations:
 - name: Univ. Lille, CNRS, Centrale Lille, UMR 9189 CRIStAL, F-59000 Lille, France
   index: 1
 - name: School of Mathematical and Computer Sciences, Heriot-Watt University, Edinburgh EH14 4AS, UK
   index: 2
 - name: Maxwell Institute for Mathematical Sciences, Edinburgh EH8 9BT, UK
   index: 3
date: 12 September 2026
bibliography: paper.bib


# Optional fields for papers that are part of a joint submission.
# For example, submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing

# If you are not making a joint submission you should remove these lines.

# aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
# aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

<!-- Link to instructions : https://joss.readthedocs.io/en/latest/index.html -->

# Summary

...

# Statement of need

<!-- `Gala` is an Astropy-affiliated Python package for galactic dynamics. Python
enables wrapping low-level languages (e.g., C) for speed without losing
flexibility or ease-of-use in the user-interface. The API for `Gala` was
designed to provide a class-based and user-friendly interface to fast (C or
Cython-optimized) implementations of common operations such as gravitational
potential and force evaluation, orbit integration, dynamical transformations,
and chaos indicators for nonlinear dynamics. `Gala` also relies heavily on and
interfaces well with the implementations of physical units and astronomical
coordinate systems in the `Astropy` package [@astropy] (`astropy.units` and
`astropy.coordinates`).

`Gala` was designed to be used by both astronomical researchers and by
students in courses on gravitational dynamics or astronomy. It has already been
used in a number of scientific publications [@Pearson:2017] and has also been
used in graduate courses on Galactic dynamics to, e.g., provide interactive
visualizations of textbook material [@Binney:2008]. The combination of speed,
design, and support for Astropy functionality in `Gala` will enable exciting
scientific explorations of forthcoming data releases from the *Gaia* mission
[@gaia] by students and experts alike. -->

# State of the field

<!-- Several tools exist for galactic dynamics computations:
`galpy` [@Bovy:2015] is a Python package with similar goals,
providing orbit integration and potential classes for galactic dynamics.
`NEMO` [@Teuben:1995] is a well-established, comprehensive stellar dynamics
toolbox written primarily in C, offering extensive functionality but with a
steeper learning curve and less integration with modern Python workflows.
Other tools like `GalPot` provide specific Milky Way potential models but lack
the broader dynamical analysis capabilities.

`Gala` was built rather than contributing to existing projects for several
reasons. First, `Gala` was designed from the ground up to integrate seamlessly
with the Astropy ecosystem, using `astropy.units` and `astropy.coordinates`
as core dependencies rather than optional features. This tight integration
enables natural workflows for astronomers already using Astropy. Second,
`Gala`'s object-oriented API with consistent interfaces across subpackages
(potentials, integrators, dynamics) provides a more modular and extensible
design than alternatives available at the time. Third, `Gala` fills a specific
niche between simple demonstration codes and full N-body simulation packages
like `Gadget` [@Springel:2005] – it focuses on the common tasks in galactic
dynamics research (orbit integration, potential evaluation, coordinate
transformations) while maintaining both performance through C implementations
and usability through its Python interface.   -->

# Software design

<!-- `Gala`'s design philosophy is based on three core principles: (1) to provide a
user-friendly, modular, object-oriented API, (2) to use community tools and
standards (e.g., Astropy for coordinates and units handling), and (3) to use
low-level code (C/C++/Cython) for performance while keeping the user interface
in Python. Within each of the main subpackages in `gala` (`gala.potential`,
`gala.dynamics`, `gala.integrate`, etc.), we try to maintain a consistent API
for classes and functions. For example, all potential classes share a common
base class and implement methods for computing the potential, forces, density,
and other derived quantities at given positions. This also works for
compositions of potentials (i.e., multi-component potential models), which
share the potential base class but also act as a dictionary-like container for
different potential components. As another example, all integrators implement a
common interface for numerically integrating orbits. The integrators and core
potential functions are all implemented in C without support for units, but the
Python layer handles unit conversions and prepares data to dispatch to the C
layer appropriately.Within the coordinates subpackage, we extend Astropy's
coordinate classes to add more specialized coordinate frames and
transformations that are relevant for Galactic dynamics and Milky Way research. -->

# Research impact statement

<!-- `Gala` has demonstrated significant research impact and grown both its user base
and contributor community since its initial release. The package has evolved
through contributions from over 18 developers beyond the original core developer
(@adrn), with community members adding new features, reporting bugs, and
suggesting new features.

While `Gala` started as a tool primarily to support the core developer's
research, it has expanded organically to support a range of applications across
domains in astrophysics related to Milky Way and galactic dynamics. The package
has been used in over 400 publications (according to Google Scholar) spanning
topics in galactic dynamics such as modeling stellar streams [@Pearson:2017],
Milky Way mass modeling, and interpreting kinematic and stellar population
trends in the Galaxy. `Gala` is integrated within the Astropy ecosystem as an
affiliated package and has built functionality that extends the widely-used
`astropy.units` and `astropy.coordinates` subpackages. `Gala`'s impact extends
beyond citations in research: Because of its focus on usability and user
interface design, `Gala` has also been incorporated into graduate-level galactic
dynamics curricula at multiple institutions.

`Gala` has been downloaded over 100,000 times from PyPI and conda-forge yearly
(or ~2,000 downloads per week) over the past few years, demonstrating a broad
and active user community. Users span career stages from graduate students to
faculty and other established researchers and represent institutions around the
world. This broad adoption and active participation validate `Gala`'s role as
core community infrastructure for galactic dynamics research. -->

<!-- # Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% } -->

# AI usage disclosure

No generative AI tools were used in the development of this software, the writing
of this manuscript, or the preparation of supporting materials.

# Acknowledgements

This work was supported by the ANR project ``Chaire IA Sherlock'' ANR-20-CHIA-0031-01 hold by P. Chainais, the national support within the programme d’investissements d’avenir ANR-16-IDEX-0004 ULNE, Région HDF, and the CNRS IEA “DAISHI” hold by P.-A. Thouvenin and A. Repetti.
The project was provided with computing HPC and storage resources by GENCI at IDRIS on the supercomputer Jean Zay’s V100 partition thanks to the grant 2024-AD010615597.
The authors acknowledge engineering support from Damien Marchal and Emmanuel Leguy, research engineers at UMR 9189 CRIStAL lab, F-59000 Lille, France.
The library has been developed while M. Bouton and S. Despierres were affiliated with Centrale Lille Institut.

# References

<!-- Example paper.bib file:

@article{Pearson:2017,
  	url = {http://adsabs.harvard.edu/abs/2017arXiv170304627P},
  	Archiveprefix = {arXiv},
  	Author = {{Pearson}, S. and {Price-Whelan}, A.~M. and {Johnston}, K.~V.},
  	Eprint = {1703.04627},
  	Journal = {ArXiv e-prints},
  	Keywords = {Astrophysics - Astrophysics of Galaxies},
  	Month = mar,
  	Title = {{Gaps in Globular Cluster Streams: Pal 5 and the Galactic Bar}},
  	Year = 2017
}

@book{Binney:2008,
  	url = {http://adsabs.harvard.edu/abs/2008gady.book.....B},
  	Author = {{Binney}, J. and {Tremaine}, S.},
  	Booktitle = {Galactic Dynamics: Second Edition, by James Binney and Scott Tremaine.~ISBN 978-0-691-13026-2 (HB).~Published by Princeton University Press, Princeton, NJ USA, 2008.},
  	Publisher = {Princeton University Press},
  	Title = {{Galactic Dynamics: Second Edition}},
  	Year = 2008
}

@article{gaia,
    author = {{Gaia Collaboration}},
    title = "{The Gaia mission}",
    journal = {Astronomy and Astrophysics},
    archivePrefix = "arXiv",
    eprint = {1609.04153},
    primaryClass = "astro-ph.IM",
    keywords = {space vehicles: instruments, Galaxy: structure, astrometry, parallaxes, proper motions, telescopes},
    year = 2016,
    month = nov,
    volume = 595,
    doi = {10.1051/0004-6361/201629272},
    url = {http://adsabs.harvard.edu/abs/2016A%26A...595A...1G},
}

@article{astropy,
    author = {{Astropy Collaboration}},
    title = "{Astropy: A community Python package for astronomy}",
    journal = {Astronomy and Astrophysics},
    archivePrefix = "arXiv",
    eprint = {1307.6212},
    primaryClass = "astro-ph.IM",
    keywords = {methods: data analysis, methods: miscellaneous, virtual observatory tools},
    year = 2013,
    month = oct,
    volume = 558,
    doi = {10.1051/0004-6361/201322068},
    url = {http://adsabs.harvard.edu/abs/2013A%26A...558A..33A}
}

@article{Hunt:2025,
  author = {{Hunt}, Jason A.~S. and {Vasiliev}, Eugene},
  title = {Milky Way dynamics in light of Gaia},
  journal = {New Astronomy Reviews},
  year = 2025,
  volume = 98,
  doi = {10.1016/j.newar.2024.101721},
  url = {https://www.sciencedirect.com/science/article/pii/S1387647324000289}
}

@misc{fidgit,
  author = {A. M. Smith and K. Thaney and M. Hahnel},
  title = {Fidgit: An ungodly union of GitHub and Figshare},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  url = {https://github.com/arfon/fidgit}
} -->
