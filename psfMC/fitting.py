import pyfits
import numpy as np
from pymc.MCMC import MCMC
from .models import multicomponent_model


# TODO: Better interface for supplying components. File? As param?
def model_psf_mcmc(subfile, subIVMfile, psffile, psfIVMfile,
                    mask_file=None, outfile=None, db_filename=None,
                    magZP=0, **kwargs):
    # TODO: Don't hardcode these!
    components = [('psf', 60, 70, 60, 70, 18, 23),
                  ('sersic', 60, 70, 60, 70, 22, 26,
                   1.5, 8.0, 0.5, 8.0, 0.1, 1.0, 0, 360)]

    if db_filename is None:
        db_filename = 'psffit'
    if outfile is None:
        outfile = subfile.replace('sci', 'resid')
    subData = pyfits.getdata(subfile, ignore_missing_end=True)
    subDataIVM = pyfits.getdata(subIVMfile, ignore_missing_end=True)
    psfData = pyfits.getdata(psffile, ignore_missing_end=True)
    psfDataIVM = pyfits.getdata(psfIVMfile, ignore_missing_end=True)

    if mask_file is not None:
        import pyregion
        hdr = pyfits.getheader(subfile)
        regfilt = pyregion.open(mask_file).as_imagecoord(hdr).get_filter()
        mask = regfilt.mask(subData.shape)
        subData = np.ma.masked_array(subData, mask=~mask)
        subDataIVM[~mask] = 0
        # TODO: Use slice to fit only the masked area. But messes up xy pos.

    # Normalize the PSF kernel
    # TODO: Convert to float64 first?
    psfScale = 1 / psfData.sum()
    psfData *= psfScale
    psfDataIVM /= psfScale**2

    sampler = MCMC(multicomponent_model(subData, subDataIVM,
                                        psfData, psfDataIVM,
                                        components=components,
                                        magZP=magZP),
                   db='pickle', name=db_filename)
    # TODO: Set these based on total number of unknown components
    kwargs.setdefault('iter', 6000)
    kwargs.setdefault('burn', 3000)
    kwargs.setdefault('tune_interval', 25)
    kwargs.setdefault('thin', 5)
    sampler.sample(**kwargs)

    ## Saves out to pickle file
    sampler.db.close()

    # TODO: Remove debug output
    stats = sampler.stats()
    for stoch in sorted(stats):
        if stoch.startswith(tuple(str(i) for i in range(len(components)))):
            print '{}: mean: {} std: {} final: {}'.format(
                stoch, stats[stoch]['mean'], stats[stoch]['standard deviation'],
                sampler.db.trace(stoch)[-1])

    # TODO: Flag for writing out files, add information to fits headers

    with pyfits.open(subfile) as f:
        f[0].data -= sampler.db.trace('convolved_model')[-1]
        f.writeto(outfile, clobber=True, output_verify='fix')
        f[0].data = sampler.db.trace('convolved_model')[-1]
        f.writeto(outfile.replace('resid', 'model'), clobber=True,
                  output_verify='fix')

    with pyfits.open(subIVMfile) as f:
        f[0].data = sampler.db.trace('composite_IVM')[-1]
        f.writeto(outfile.replace('resid','residerr'), clobber=True,
                  output_verify='fix')