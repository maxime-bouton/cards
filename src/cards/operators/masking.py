"""Implementation of the loss operator for an inpainting problem."""

# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais. A Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse Problems. IEEE Transactions on Computational Imaging, 2026, 12, pp.839-849. (https://dx.doi.org/10.1109/TCI.2026.3685151)

# TODO: documentation

from cards.operators.linear_operator import LinearOperator


class Masking(LinearOperator):
    def __init__(self, mask):
        self.mask = mask

    def forward(self, input):
        return self.mask * input

    def adjoint(self, input):
        return self.mask * input
