from .ComponentBase import ComponentBase
from pymc.distributions import DiscreteUniform


class PSFSelector(ComponentBase):
    """
    Selects a PSF from a supplied list
    """
    def __init__(self, psflist=None, varlist=None, filenames=None):
        self.selected_index = DiscreteUniform('PSF_Index',
                                              lower=0,
                                              upper=len(psflist)-1)
        self.psflist = psflist
        self.varlist = varlist
        if filenames is None:
            self.filenames = ['']*len(psflist)
        else:
            self.filenames = filenames
        super(PSFSelector, self).__init__()

    def psf(self):
        return self.psflist[self.selected_index]

    def variance(self):
        return self.varlist[self.selected_index]

    def filename(self):
        return self.filenames[self.selected_index]
