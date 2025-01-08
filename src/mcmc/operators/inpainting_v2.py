from mcmc.operators.linear_operator import LinearOperator


class SerialInpainting(LinearOperator):
    def __init__(self, mask):
        self.mask = mask

    def forward(self, input_image):
        return self.mask * input_image

    def adjoint(self, input_data):
        return self.mask * input_data
