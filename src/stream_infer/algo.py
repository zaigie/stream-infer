class BaseAlgo:
    def __init__(self, name=None) -> None:
        if not name:
            name = self.__class__.__name__
        self.name = name

    def init(self):
        """
        Initialize the algo model or other resources.
        """
        raise NotImplementedError

    def run(self):
        """
        Run inference and return the inference result.
        """
        raise NotImplementedError
