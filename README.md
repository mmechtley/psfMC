psfMC
=====

MCMC 2D surface brightness modeling for quasar host galaxies

Usage is provided in docstrings. My Ph.D. dissertation contains a more detailed discussion of the problem (simultaneous modeling of quasar point sources and host galaxies) and why I have made specific design decisions. My dissertation is freely available at the following URL:

[*Markov Chain Monte Carlo Modeling of High-Redshift Quasar Host Galaxies in Hubble Space Telescope Imaging*](http://biphenyl.org/papers/mmechtleydis.pdf)

The current master branch is an **alpha** release. There are still some known bugs and planned features for an inital release. These are noted in the code currently (FIXME/TODO/etc), but will be migrated to the github issue system.

Dependencies
------------

The software depends on the numpy, scipy, pymc, and pyfits modules.

Additionally, the pyregion module is optional but strongly recommended for ease of masking out foreground or background objects unrelated to the quasar being modeled.

The numexpr module is also optional, and is used to parallelize certain calculations (generating Sersic profiles), providing a marginal increase in speed.
