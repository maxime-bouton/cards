"""Implementation of the loss operator for an inpainting problem."""

# TODO: documentation

from cards.operators.linear_operator import LinearOperator


class Masking(LinearOperator):
    def __init__(self, mask):
        self.mask = mask

    def forward(self, input):
        return self.mask * input

    def adjoint(self, input):
        return self.mask * input
