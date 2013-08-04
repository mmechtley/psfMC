from __future__ import division
import pymc.distributions
from pymc.Container import Container
from pymc import CircVonMises as _CircVonMises
from pymc import CircularStochastic

# FIXME: can't use 'className' here, uses global variable instead of creating
# function with literal
# for dist in pymc.distributions.availabledistributions:
#     className = pymc.distributions.capitalize(dist)
#     def distFactory(**kwargs):
#         return pymc.distributions.__dict__[className]('', **kwargs)
#     locals()[className] = distFactory

def Uniform(lower=None, upper=None, **kwargs):
    return pymc.distributions.Uniform('', lower=lower, upper=upper, **kwargs)


def Normal(mu=None, tau=None, **kwargs):
    return pymc.distributions.Normal('', mu=mu, tau=tau, **kwargs)

def TruncatedNormal(mu=None, tau=None, a=None, b=None, **kwargs):
    return pymc.distributions.TruncatedNormal('', mu=mu, tau=tau, a=a, b=b,
                                              **kwargs)


def VonMises(mu=None, kappa=None, **kwargs):
    return pymc.distributions.VonMises('', mu=mu, kappa=kappa, **kwargs)


def CircVonMises(mu=None, kappa=None, **kwargs):
    return _CircVonMises('', mu=mu, kappa=kappa, **kwargs)


class CircUniform(CircularStochastic, pymc.distributions.Uniform):
     def __init__(self, lower, upper, *args, **kwargs):
         self.interval_parents = Container([upper, lower])
         pymc.distributions.Uniform.__init__(self, name='',
                                             lower=lower, upper=upper,
                                             *args, **kwargs)