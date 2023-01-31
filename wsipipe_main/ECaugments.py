class RandomCropSpecifyOffset(object):
    def __init__(self, size_diff: int) -> None:
        super().__init__()
        self.size_diff = size_diff