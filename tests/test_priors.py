import numpy as np
import matplotlib.pyplot as pp
from psfMC.distributions import Schechter


def test_schechter():
    pp.figure()
    for alpha in (-0.5, -1.0, -1.5):
        schech = Schechter('test', alpha=alpha, mstar=0.0, faint_cutoff=10.0,
                           value=np.ones(10000))
        #schech.shape = (10000,)
        # generate some random samples
        mag_samples = schech.random()
        pp.hist(mag_samples, log=True, histtype='step', bins=100)
        xs = np.linspace(-5, 20, 100)
        schechs = [Schechter('test', alpha=alpha, mstar=0.0, faint_cutoff=10.0,
                           value=x) for x in xs]
        probs = np.exp([schech.logp for schech in schechs])
        pp.plot(xs, probs)
    pp.show()

if __name__ == '__main__':
    test_schechter()
