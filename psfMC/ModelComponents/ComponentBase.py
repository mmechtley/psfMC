from numpy import sum, concatenate, array, ravel


class ComponentBase(object):
    """
    Base class which other components inherit from
    """
    _fits_abbrs = []

    def __init__(self):
        super(ComponentBase, self).__init__()
        self._priors = dict()
        self._constants = dict()

    def get_stochastic_val(self, name):
        """
        Get the value of a stochastic variable from appropriate dict. For priors
        this is prior.value. For constants, it is just the supplied value.
        """
        if name in self._priors:
            return self._priors[name].value
        else:
            return self._constants[name]

    def assign_stochastic(self, name, value):
        """
        Assign a stochastic, tracking it in the appropriate dict. Prior
        distributions go into priors store, constant values go into constants
        """
        if hasattr(value, 'value'):
            self._priors[name] = value
        else:
            self._constants[name] = value

    def set_stochastic_values(self, param_values=None):
        """
        Set values for all stochastic variables given a vector of values.

        :param param_values: Vector of values for all stochastic variables, in
            canonical (alphabetically sorted) order. If None, values for each
            will be drawn from the corresponding prior distribution first.
        :returns: Vector of values that were set for all stochastic variables
        """
        sorted_prior_names = sorted(self._priors.keys())

        if param_values is None:
            param_values = array([])
            for prior_name in sorted_prior_names:
                prior_val = ravel(self._priors[prior_name].random())
                param_values = concatenate((param_values, prior_val))

        start_index = 0
        for stoch in sorted_prior_names:
            stoch_size = ravel(self._priors[stoch].value).size
            new_value = array(param_values[start_index:start_index+stoch_size])
            self._priors[stoch].value = new_value
            start_index += stoch_size

        return param_values

    def num_stochastics(self):
        """
        Count the total number of stochastic variables on this component
        """
        return sum(self.stochastic_lens(), dtype=int)

    def stochastic_lens(self):
        """
        Get a list of the lengths of each stochastic variable (i.e. number of
        separate free parameters per stochastic). For most this is 1, but e.g.
        xy is 2.
        """
        return [array(prior.value).size for key, prior
                in sorted(self._priors.items())]

    def stochastic_names(self):
        """
        Get a list of names of each stochastic variable, in canonical (alpha
        sorted) order
        """
        return [prior.name for key, prior in sorted(self._priors.items())]

    def update_stochastic_names(self, count=None):
        """
        Set trace names based on component number, type, and attribute name
        Also add abbreviated 'fitsname' attribute to individual stochastics
        """
        comptype = self.__class__.__name__
        for attr in self._priors:
            newname = '{}_{}'.format(comptype, attr)
            fitsname = newname
            for longname, abbr in self.__class__._fits_abbrs:
                fitsname = fitsname.replace(longname, abbr)

            if count is not None:
                newname = '{:d}_{}'.format(count, newname)
                fitsname = '{:d}{}'.format(count, fitsname)

            try:
                self._priors[attr].name = newname
                self._priors[attr].fitsname = fitsname
            except AttributeError:
                pass

    def log_priors(self):
        """
        Get the joint log-probablity of all priors, based on their current
        values
        """
        logp = 0
        for prior in self._priors.values():
            logp += sum(prior.logp(prior.value))
        return logp


class StochasticProperty(object):
    """
    A property-like descriptor for stochastic variables. Accepts a key name
    in the constructor and, for property getter and setter, calls
    get_stochastic_val(key) and assign_stochastic(key, value) on a ComponentBase
    instance object. See e.g. implementation of Sky component:

    class Sky(ComponentBase):
        adu = StochasticProperty('adu')
        ...
    """
    def __init__(self, key):
        self.key = key

    def __get__(self, instance, owner=None):
        return instance.get_stochastic_val(self.key)

    def __set__(self, instance, value):
        instance.assign_stochastic(self.key, value)

    def __delete__(self, instance):
        raise NotImplementedError('Cannot delete stochastics')
