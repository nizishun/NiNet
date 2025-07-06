import pathlib
import random
import numpy as np


class InfImageIterator:
    """
    Infinite iterator of original image from a specified directory.
    """

    def __init__(self, image_dir, batch_size=1, shuffle=False):
        self.batch_size = batch_size
        self.shuffle = shuffle

        image_dir = pathlib.Path(image_dir)
        assert image_dir.exists(), f'Image directory [{image_dir}] does not exist.'
        self.files = sorted(image_dir.glob('*'))

    def __iter__(self):
        self.i = 0
        self.n = len(self.files)

        return self

    def __next__(self):
        batch = []
        for _ in range(self.batch_size):
            if self.shuffle and self.i == 0:
                random.shuffle(self.files)
            filepath = self.files[self.i]
            f = open(filepath, 'rb')
            batch.append(np.frombuffer(f.read(), dtype=np.uint8))
            self.i = (self.i + 1) % self.n

        return batch,
