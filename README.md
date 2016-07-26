psfMC
=====
MCMC 2D surface brightness modeling for quasar / host galaxy decompositions

Usage is provided in docstrings, and an example model is provided in the 
`examples` directory. My Ph.D. thesis contains a more detailed discussion of 
the problem (simultaneous modeling of quasar point sources and host galaxies) 
and why I have made specific design decisions. My dissertation is available 
open-access at the following URL:

[*Markov Chain Monte Carlo Modeling of High-Redshift Quasar Host Galaxies in 
Hubble Space Telescope Imaging*]
(http://repository.asu.edu/attachments/126037/content/Mechtley_asu_0010E_13571.pdf)

A journal article is forthcoming, but if you use psfMC in the meantime, please 
cite:

[Mechtley, M. 2014, PhD Thesis, Arizona State Univ.]
(https://ui.adsabs.harvard.edu/#abs/2014PhDT.........1M)

Recent Changes and Notes
------------------------
The current master branch is a **beta** release (until I finalize formal unit 
tests). No additional features are planned for the 1.0 release, and additional 
bugfixes are considered low priority (may make it in or may not).

Beta release 1.0b5 adds a new analysis submodule. FITS image production has been 
moved there, and various statistical and plotting methods have been added for
analyzing posterior chains. This includes MCMC covariance corner plots via Dan 
Foreman-Mackey's [corner.py](https://github.com/dfm/corner.py) module. Also 
includes two new command line scripts: `corner_plot` and `plot_chain` to 
generate these plots from a terminal.

Beta release 1.0b4 migrated from pyfits to the astropy.io.fits module, since 
this is the package where future development of FITS support will continue.  

Beta release 1.0b3 changes the way that certain model parameters are supplied. 
Specifically, it adds a new `Configuration` component, as the only required 
component in the python model file. Parameters related to the model definition 
(measured galaxy images, psf images, magnitude zeropoint) are now specified in 
this component. Parameters related to MCMC sampling (`burn`, `iter`, etc.) are 
still passed directly to `model_galaxy_mcmc`. See the example model file in 
`examples/model_J0005-0006.py` for a usage example.

Dependencies
------------
The software depends on the `numpy`, `scipy`, `pymc`, and `astropy` modules. The 
`matplotlib` module and `corner` modules are required for certain tests, 
posterior statistics, and posterior analysis plots.

Additionally, the `pyregion` module is optional but strongly recommended for 
ease of masking out foreground or background objects unrelated to the quasar 
being modeled.

The `numexpr` module is also optional, and is used to parallelize certain 
calculations (generating Sersic profiles), providing a marginal increase in 
speed.
