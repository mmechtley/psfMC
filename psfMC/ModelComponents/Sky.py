from .ComponentBase import ComponentBase

class Sky(ComponentBase):
    """
    Sky component
    """
    def __init__(self, adu=None):
        self.adu = adu
        super(Sky, self).__init__()