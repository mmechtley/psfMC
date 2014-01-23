import pymc.distributions
from pymc.Container import Container
from pymc import CircularStochastic, stochastic_from_dist
from numpy import log, log10, exp, inf, sum
from numpy.random import random
from scipy.special import gammainccinv


def _magic_dist(class_name):
    """
    Magically import all pymc distributions, without requiring name argument
    """
    return lambda *args, **kwargs: \
        pymc.distributions.__dict__[class_name]('', *args, **kwargs)

_dists = [pymc.distributions.capitalize(_dist) for _dist
          in pymc.distributions.availabledistributions]
for _dist in _dists:
    locals()[_dist] = _magic_dist(_dist)


class CircUniform(CircularStochastic, pymc.distributions.Uniform):
    def __init__(self, lower, upper, *args, **kwargs):
        self.interval_parents = Container([upper, lower])
        pymc.distributions.Uniform.__init__(self, name='',
                                            lower=lower, upper=upper,
                                            *args, **kwargs)


def _schechter_like(x, alpha=-1.0, mstar=0.0, faint_cutoff=None):
    """
    Schechter (gamma) log-likelihood. Parameterized in magnitudes, ie.
    N(M) dM \propto Lrat^(alpha+1) * exp(-Lrat) dM
    where Lrat = 10^(-0.4*(M - mstar))
    """
    if faint_cutoff is None:
        faint_cutoff = mstar + 20
    lum_ratio = 10**(-0.4*(x-mstar))
    #faint_ratio = 10**(-0.4*(faint_cutoff - mstar))
    #tot_count, tot_err = _schechter_int(alpha, faint_ratio, inf)
    return sum((alpha+1)*log(lum_ratio) - lum_ratio)  #- log(tot_count)
    # + log(0.4*log(10)*phst)


def _rschechter(alpha=-1.0, mstar=0.0, faint_cutoff=None, size=None):
    """
    Schechter random variable. Schechter CDF is:
    N(>L) = nstar * gammainc(a+2, lratio)
    http://www.astro.virginia.edu/class/whittle/astr553/Topic04/Lecture_4.html
    where nstar = 0.4*log(10)*phst and +2 is because of +1 for mags
    """
    if faint_cutoff is None:
        faint_cutoff = mstar + 20
    # FIXME: Alpha +1 seems to match the pdf from above. Unsure what's up
    y_gamma = random(size)
    #y_gamma *= gammaincc(a+2,faint_cutoff)
    lratio = gammainccinv(alpha+2, y_gamma)
    return -2.5*log10(lratio) + mstar


#Schechter = stochastic_from_dist('Schechter', logp=_schechter_like,
#                                 random=_rschechter)