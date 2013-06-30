from copy import copy
from pymc.Container import DictContainer, ContainerBase, file_items
from pymc.Container_values import OCValue


class ComponentBase(ContainerBase):
    """
    Base class which other components inherit from
    """
    def __init__(self):
        # Significant portions of this adapted from
        # pymc.Container.ObjectContainer.
        dictpop = copy(self.__dict__)
        if 'self' in dictpop:
            dictpop.pop('self')
        self._dict_container = DictContainer(dictpop)
        file_items(self, dictpop)

        self._value = copy(self)
        ContainerBase.__init__(self, self)
        self.OCValue = OCValue(self)

    def update_trace_names(self, count=None):
        """
        Set trace names based on component number, type, and attribute name
        """
        comptype = self.__class__.__name__
        for attr in self.__dict__:
            newname = '{}_{}'.format(comptype, attr)
            if count is not None:
                newname = str(count) + '_' + newname
            try:
                self.__dict__[attr].__name__ = newname
            except AttributeError:
                pass

    def replace(self, item, new_container, key):
        dict.__setitem__(self.__dict__, key, new_container)

    def _get_value(self):
        self.OCValue.run()
        return self._value

    value = property(fget=_get_value,
                     doc='Copy of object with stochastics replaced by values')

    @staticmethod
    def from_tuple(tup):
        return __class__.__init__(*tup)