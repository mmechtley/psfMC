import pyfits
import numpy as np
from pymc.MCMC import MCMC
from .models import multicomponent_model


# TODO: Better interface for supplying components. File? As param?
def model_psf_mcmc(subfile, subIVMfile, psffile, psfIVMfile,
                    maskRegionFile=None, outfile=None, **kwargs):
    # TODO: Don't hardcode these!
    components = [('psf', 60, 70, 60, 70, 18, 23),
                  ('sersic', 60, 70, 60, 70, 22, 26,
                   1.5, 8.0, 0.5, 8.0, 0.1, 1.0, 0, 360)]

    if outfile is None:
        outfile = subfile.replace('sci', 'resid')
    subData = pyfits.getdata(subfile, ignore_missing_end=True)
    subDataIVM = pyfits.getdata(subIVMfile, ignore_missing_end=True)
    psfData = pyfits.getdata(psffile, ignore_missing_end=True)
    psfDataIVM = pyfits.getdata(psfIVMfile, ignore_missing_end=True)

    if maskRegionFile is not None:
        import pyregion
        hdr = pyfits.getheader(subfile)
        regfilt = pyregion.open(maskRegionFile).as_imagecoord(hdr).get_filter()
        mask = regfilt.mask(subData.shape)
        subData = np.ma.masked_array(subData, mask=~mask)
        psfData = np.ma.masked_array(psfData, mask=~mask)
        subDataIVM[~mask] = 0
        psfDataIVM[~mask] = 0
        maskInd = np.where(mask)
        fitslice = (slice(maskInd[0].min(), maskInd[0].max() + 1),
                    slice(maskInd[1].min(), maskInd[1].max() + 1))
        subData = subData[fitslice]
        subDataIVM = subDataIVM[fitslice]
        psfData = psfData[fitslice]
        psfDataIVM = psfDataIVM[fitslice]

    # Normalize the PSF kernel
    psfScale = 1 / psfData.sum()
    psfData *= psfScale
    psfDataIVM /= psfScale**2

    sampler = MCMC(multicomponent_model(subData, subDataIVM,
                                        psfData, psfDataIVM,
                                        components=components))
    # TODO: Set these based on total number of unknown components
    kwargs.setdefault('iter', 2500)
    kwargs.setdefault('burn', 1800)
    kwargs.setdefault('tune_interval', 25)
    kwargs.setdefault('thin', 5)
    sampler.sample(**kwargs)

    ## Saves out to pickle file
    sampler.db.close()

    stats = sampler.stats()
    for stoch in stats:
        print '{}: mean: {} std: {}'.format(stoch, stats[stoch]['mean'],
                                            stats[stoch]['standard deviation'])

    # TODO: Write out copies of subtracted data file, and IVM to go with it

    # xyshift = stats['xyshift']['mean']
    # xyerr = stats['xyshift']['standard deviation']
    # scale = stats['scale']['mean']
    # scaleErr = stats['scale']['standard deviation']
    # sd = pyfits.getdata(subfile)
    # psfd = pyfits.getdata(psffile)
    # hdr = pyfits.getheader(subfile)
    # hdr.add_history('Used PSF {}'.format(psffile))
    # hdr.add_history('PSF Shift: {} Err: {}'.format(xyshift, xyerr))
    # hdr.add_history('PSF Scale: {} Err: {}'.format(scale, scaleErr))

    # pyfits.writeto(outfile,
    #                data=sd - scale * shift(psfd, xyshift, order=1, mode='wrap'),
    #                header=hdr, clobber=True, output_verify='fix')

    # pyfits.writeto(subfile.replace('sci','residerr'),
    #                data=shift(psfDataIVM, xyshift, order=1, mode='wrap')/scale,
    #                header=pyfits.getheader(subIVMfile),
    #                clobber=True)