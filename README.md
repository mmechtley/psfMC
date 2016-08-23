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
tests). I've used it myself extensively, but please be aware that there may 
still be bugs lurking around.
 
Beta release 2.0b1 is a **backwards-incompatible** release that switches to 
`emcee` for sampling (and is compatible with Python3). The `pymc` requirement 
has dropped entirely, as I now also store trace databases as a FITS table and 
handle the priors myself in a more lightweight manner. Note that because priors 
in `psfMC.distributions` now wrap `scipy.stats` distributions directly, some 
parameter names have changed (e.g., Normal now expects `loc=` and `scale=` 
instead of `mu=` and `tau=`). 

The rationale behind such a large change is that the Affine-Invariant ensemble 
sampler used by `emcee` is **much** more efficient for this particular problem 
(high-dimensional parameter space, reasonably convex single-mode posterior). 
Traditional Metropolis-Hastings chains required 50,000 samples or more of 
burn-in **each** to approach convergence, severely limiting the number of chains 
that could be used, and thus our confidence in the final result. The sampler now 
reliably approaches convergence with a similar number of total burn-in samples 
(e.g., `iterations=250 * nwalkers=200 = 50000`), but the ensemble sampler means 
this only has to be run once, rather than once for each chain. The large number 
of walkers replaces the primary purpose of multiple independent chains in the 
previous implementation: to improve our confidence in convergence. 

As a side effect, the sampling is now overall faster (takes less time to run the 
same number of samples), since `emcee` is a lightweight sampler only and doesn't
attempt clever caching / lazy evaluation like `pymc` (which the previous `pymc` 
implementation was not taking advtange of anyway).

Beta release 1.0b5 adds a new analysis submodule. FITS image production has been 
moved there, and various statistical and plotting methods have been added for
analyzing posterior chains. This includes MCMC covariance corner plots via Dan 
Foreman-Mackey's [corner.py](https://github.com/dfm/corner.py) module. Also 
includes two new command line scripts: `corner_plot` and `plot_chain` to 
generate these plots from a terminal.

Dependencies
------------
The software depends on the `numpy`, `scipy`, `emcee`, `astropy`, and `six` (for 
Python 2 backward compatibility) modules. Additionally the `matplotlib` and 
`corner` modules are required for posterior analysis plots.

The `pyregion` module is optional but strongly recommended for ease of masking 
out foreground or background objects unrelated to the quasar being modeled.

The `numexpr` module is also optional, and is used to parallelize certain 
calculations (generating Sersic profiles), providing a slight increase in 
overall speed.

Installation
------------
I only test/support Anaconda, use other Python distributions at your own risk.
Note: everything else is included with the default Anaconda distribution.
`conda install --channel astropy emcee pyregion corner`
`pip install git+https://github.com/mmechtley/psfMC.git`
