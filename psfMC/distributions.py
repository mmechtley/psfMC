from __future__ import division
import pymc.distributions

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


def VonMises(mu=None, kappa=None, **kwargs):
    return pymc.distributions.VonMises('', mu=mu, kappa=kappa **kwargs)