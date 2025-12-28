from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


class MotionBlurKernel:
    """
    Generate realistic motion blur kernels with configurable size and intensity.

    Parameters
    ----------
    size : tuple of int, default (100, 100)
        Size of the kernel in pixels (width, height).
    intensity : float, default 0.0
        Intensity of motion blur between 0 and 1.
        0 = linear motion, 1 = highly non-linear motion.
    """

    def __init__(
        self,
        size=(100, 100),
        intensity=0.0,
        dtype: np.dtype | None = None,
        rng: np.random.Generator | None = None,
    ):
        if not isinstance(size, tuple) or len(size) != 2:
            raise ValueError("Size must be a tuple of 2 positive integers")
        if not all(isinstance(s, int) and s > 0 for s in size):
            raise ValueError("Size must contain positive integers")
        if not 0 <= intensity <= 1:
            raise ValueError("Intensity must be between 0 and 1")

        self.size = size
        self.intensity = intensity
        self.diagonal = np.sqrt(size[0] ** 2 + size[1] ** 2)
        self.dtype = dtype
        self.rng = rng or np.random.default_rng()
        self._kernel = self.generate()

    def _generate_path(self):
        """
        Generate a random motion blur path based on intensity.

        Returns
        -------
        list of tuple
            List of (x, y) coordinates representing the blur path.
        """
        max_path_len = (
            0.75
            * self.diagonal
            * (self.rng.uniform() + self.rng.uniform(0, self.intensity**2))
        )

        steps = []
        while sum(steps) < max_path_len:
            step = self.rng.beta(1, 30) * (1 - self.intensity + 0.1) * self.diagonal
            if step < max_path_len:
                steps.append(step)

        max_angle = self.rng.uniform(0, self.intensity * np.pi)
        jitter = self.rng.beta(2, 20)

        angles = [self.rng.uniform(-max_angle, max_angle)]
        for _ in range(len(steps) - 1):
            angle = self.rng.triangular(0, self.intensity * max_angle, max_angle + 0.1)
            if self.rng.uniform() < jitter:
                angle *= -np.sign(angles[-1])
            else:
                angle *= np.sign(angles[-1])
            angles.append(angle)

        complex_increments = np.array(steps) * np.exp(1j * np.array(angles))
        path_complex = np.cumsum(complex_increments)

        center = (self.size[0] + 1j * self.size[1]) / 2
        path_complex -= np.mean(path_complex)
        path_complex *= np.exp(1j * self.rng.uniform(0, np.pi))
        path_complex += center

        return [(z.real, z.imag) for z in path_complex]

    def generate(self):
        """
        Generate the motion blur kernel as a normalized numpy array.

        Returns
        -------
        numpy.ndarray
            2D array representing the motion blur kernel, normalized to sum to 1.
        """
        path = self._generate_path()

        size_2x = (self.size[0] * 2, self.size[1] * 2)
        img = Image.new("L", size_2x, 0)
        draw = ImageDraw.Draw(img)

        path_2x = [(x * 2, y * 2) for x, y in path]

        line_width = max(1, int(self.diagonal / 150))
        draw.line(path_2x, fill=255, width=line_width)

        blur_radius = max(1, int(self.diagonal * 0.01))
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        img = img.resize(self.size, Image.LANCZOS)

        kernel = np.array(img, dtype=self.dtype)
        kernel = kernel / np.sum(kernel) if np.sum(kernel) > 0 else kernel

        self._kernel = kernel
        return kernel

    def save(self, filepath):
        """
        Save the kernel as a PNG image.

        Parameters
        ----------
        filepath : str or Path
            Path where to save the kernel image.
        """

        filepath = Path(filepath).with_suffix(".h5")

        with h5py.File(filepath, "w") as f:
            f.create_dataset("kernel", data=self._kernel)

    @property
    def kernel(self):
        """
        Get the generated kernel as a numpy array.

        Returns
        -------
        numpy.ndarray
            The motion blur kernel.
        """
        return self._kernel
