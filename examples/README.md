psfMC Example
=============
These files provide an example of using psfMC to simultaneously model a quasar 
and nearby galaxy in Hubble Space Telescope Wide Field Camera 3 IR imaging. The 
particular quasar is SDSS J0005-0006 (z=5.85, Fan et al. 2004), observed with
WFC3IR in June 2012 (PID: 12332, PI: Windhorst).

Data Calibration
----------------
The quasar and PSF star images (sci*.fits) are extracted from calibrated frames,
with individual exposures combined using the software Multidrizzle (Koekemoer 
et al 2002, 2011). The inverse variance maps (ivm*.fits) are created from the 
Multidrizzle ERR maps, which include noise from instrumental effects as well as 
count rate determination uncertainty from both the background sky and the 
individual sources. The images have been multiplied by a correction factor to 
account for correlated noise in the drizzled images, as in Guo et al. (2013).

Model Definition Files
----------------------
**model_J0005-0006.py:**
The quasar is modeled as a point source component with an accompanying Sersic 
profile component. The position prior for each is a uniform box centered at 
(64.5, 64.5) with width and height of 16 pixels. The total AB magnitude was 
measured to be 20.83 mag. The point source magnitude prior is uniform between 
0.2 mag brighter (overestimating the uncertainty) and 1.5 mag fainter (i.e. at 
least 25% of the observed flux is assumed to come from the point source). The 
Sersic profile magnitude prior is uniform between the total magnitude and 27.5 
mag (roughly the faintest galaxy that could be detected in the image). The 
semi-major and semi-minor axes (reff and reff_b) are allowed to vary between 2 
and 12 pixels (roughly 0.75 and 4.3 kpc in WMAP-era LCDM cosmology). The Sersic 
index prior (Weibull distribution with alpha=1.5 and beta=4) is based on the 
empirical distribution of Sersic indexes at lower redshift. 

A second Sersic profile also models a small galaxy to the upper left of the 
quasar, which blends with the quasar diffraction spike. The priors were selected
based on a rough Galfit model. 

**mask_J0005-0006.reg:** An SAOImage ds9 region file is included to define the 
fitting region. It defines a large circle centered on the quasar, with 3 
exclusion regions making out pixels belonging to other galaxies which are not 
blended with the quasar.

Example Run Script
------------------
The file **run_example.py** shows an example for setting up and running the 
MCMC sampler. This is the file that you actually run, using the terminal 
command:

    python run_example.py