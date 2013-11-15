from __future__ import division
import pymc.distributions
from pymc.Container import Container
from pymc import CircularStochastic

# Magically import all pymc distributions, without requiring name argument
def _magic_dist(className):
    return lambda *args, **kwargs: \
        pymc.distributions.__dict__[className]('', *args, **kwargs)

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
