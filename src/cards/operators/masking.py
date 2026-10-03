"""Implementation of the masking operator involved in inpainting problems."""

# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais. A Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse Problems. IEEE Transactions on Computational Imaging, 2026, 12, pp.839-849. (https://dx.doi.org/10.1109/TCI.2026.3685151)

import cards.backend as xp
from cards.operators.linear_operator import LinearOperator


class Masking(LinearOperator):
    r"""Implementation of a masking operator as involved in inpainting problems.

    Parameters
    ----------
    mask : xp.ndarray
        Mask tensor, with 0 corresponding to masked entries, 1 to observed entries.

    Attributes
    ----------
    mask : xp.ndarray
        Mask tensor, with 0 corresponding to masked entries, 1 to observed entries.

    Note
    ----
    Masking is implemented as a Hadamard product, and not as a scropping operator (i.e., retaining only non-masked entries from an input tensor).
    """

    def __init__(self, mask: xp.ndarray):
        super().__init__(mask.shape, mask.shape)
        self.mask = mask

    def forward(self, image: xp.ndarray, op=None) -> xp.ndarray:
        return self.mask * image

    def adjoint(self, data: xp.ndarray, adjoint_op=None) -> xp.ndarray:
        return self.mask * data
