class BaseAlgo:
    def __init__(self, name=None) -> None:
        if not name:
            name = self.__class__.__name__
        self.name = name

    def init(self):
        """
        初始化算法模型或其他资源。
        """
        raise NotImplementedError

    def run(self):
        """
        运行推理并返回推理结果。
        """
        raise NotImplementedError
