psfMC
=====

MCMC 2D surface brightness modeling for quasar host galaxies

Usage is provided in docstrings. My Ph.D. thesis contains a more detailed discussion of the problem (simultaneous modeling of quasar point sources and host galaxies) and why I have made specific design decisions. My dissertation is freely available at the following URL:

[*Markov Chain Monte Carlo Modeling of High-Redshift Quasar Host Galaxies in Hubble Space Telescope Imaging*](http://repository.asu.edu/attachments/126037/content/Mechtley_asu_0010E_13571.pdf)

A journal article is forthcoming, but if you use psfMC in the meantime, please cite it as:
Mechtley, M. 2014, PhD thesis, Arizona State Univ.

The current master branch is a **beta** release. No additional features are planned for the 1.0 release, and additional bugfixes are considered low priority (may make it in or may not).

Dependencies
------------

The software depends on the numpy, scipy, pymc, and pyfits modules.

Additionally, the pyregion module is optional but strongly recommended for ease of masking out foreground or background objects unrelated to the quasar being modeled.

The numexpr module is also optional, and is used to parallelize certain calculations (generating Sersic profiles), providing a marginal increase in speed.
