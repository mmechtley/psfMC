from .ComponentBase import ComponentBase, StochasticProperty


class Sky(ComponentBase):
    """
    Sky component
    """
    adu = StochasticProperty('adu')

    def __init__(self, adu=None):
        super(Sky, self).__init__()
        self.adu = adu

    def add_to_array(self, arr, **kwargs):
        arr += self.adu
        return arr
