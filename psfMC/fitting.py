from __future__ import division
import pyfits
import numpy as np
from warnings import warn
from pymc.MCMC import MCMC
from pymc.Matplot import plot
from .models import multicomponent_model
try:
    import pyregion
except ImportError:
    pyregion = None


# TODO: Friendlier interface for supplying components. File?
def model_psf_mcmc(obs_file, subIVM_file, psf_file, psfIVM_file,
                   fit_components=None, mag_zeropoint=0,
                   mask_file=None, db_name=None,
                   write_fits=True, **kwargs):
    if fit_components is None:
        raise ValueError('No fitting components specified. Please supply at ' +
                         'least one component.')
    if db_name is None:
        db_name = obs_file.replace('.fits','_db')

    # TODO: Set these based on total number of unknown components
    kwargs.setdefault('iter', 1000)
    kwargs.setdefault('burn', 500)
    kwargs.setdefault('tune_interval', 25)
    kwargs.setdefault('thin', 5)

    subData = pyfits.getdata(obs_file, ignore_missing_end=True)
    subDataIVM = pyfits.getdata(subIVM_file, ignore_missing_end=True)
    psfData = pyfits.getdata(psf_file, ignore_missing_end=True)
    psfDataIVM = pyfits.getdata(psfIVM_file, ignore_missing_end=True)

    if mask_file is not None:
        if pyregion is not None:
            hdr = pyfits.getheader(obs_file)
            regfilt = pyregion.open(mask_file).as_imagecoord(hdr).get_filter()
            mask = regfilt.mask(subData.shape)
            subData = np.ma.masked_array(subData, mask=~mask)
            subDataIVM[~mask] = 0
            # TODO: Use slice to fit only the masked area. But messes up xy pos.
        else:
            warn('pyregion could not be imported. mask_file will be ignored.')

    # Normalize the PSF kernel
    # TODO: Convert to float64 first?
    psfScale = 1 / psfData.sum()
    psfData *= psfScale
    psfDataIVM /= psfScale**2

    mc_model = multicomponent_model(subData, subDataIVM, psfData, psfDataIVM,
                                    components=fit_components,
                                    magZP=mag_zeropoint)
    sampler = MCMC(mc_model, db='pickle', name=db_name)
    sampler.sample(**kwargs)

    ## Saves out to pickle file
    sampler.db.close()

    # TODO: Remove debug output
    stats = sampler.stats()
    for stoch in sorted(stats):
        if stoch.startswith(tuple(str(i) for i in range(len(fit_components)))):
            print '{}: mean: {} std: {} final: {}'.format(
                stoch, stats[stoch]['mean'], stats[stoch]['standard deviation'],
                sampler.db.trace(stoch)[-1])

    if write_fits:
        # TODO: Add fit information to fits headers
        # TODO: Don't use replace here
        resid_file = obs_file.replace('sci', 'resid')
        model_file = obs_file.replace('sci', 'model')
        modelIVM_file = obs_file.replace('sci', 'residivm')
        with pyfits.open(obs_file) as f:
            f[0].data -= sampler.db.trace('convolved_model')[-1]
            f.writeto(resid_file, clobber=True, output_verify='fix')
            f[0].data = sampler.db.trace('convolved_model')[-1]
            f.writeto(model_file, clobber=True, output_verify='fix')

        with pyfits.open(subIVM_file) as f:
            f[0].data = sampler.db.trace('composite_IVM')[-1]
            f.writeto(modelIVM_file, clobber=True, output_verify='fix')

    # TODO: Return something? Maybe model, resid, IVM arrays?