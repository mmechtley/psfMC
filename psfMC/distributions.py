import pymc.distributions
from pymc.Container import Container
from pymc import CircularStochastic, stochastic_from_dist
from numpy import log, log10
from numpy.random import random
from scipy.special import gammaincinv


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


def _schechter_like(x, alpha, mstar):
    lratio = 10**(-0.4*(x-mstar))
    return (alpha+1)*log(lratio) - lratio  # + log(0.4*log(10)*phst)


def _rschechter(alpha, mstar, size=None):
    """
    Schechter random variable. Schechter CDF is:
    N(>L) = nstar * gammainc(a+2, lratio)
    http://www.astro.virginia.edu/class/whittle/astr553/Topic04/Lecture_4.html
    where nstar = 0.4*log(10)*phst and +2 is because of +1 for mags
    """
    lratio = gammaincinv(alpha+2, random(size))
    return -2.5*log10(lratio) + mstar

Schechter = stochastic_from_dist('Schechter', logp=_schechter_like,
                                 random=_rschechter)