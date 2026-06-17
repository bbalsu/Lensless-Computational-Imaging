import torch.nn as nn


class ModularLeADMM(nn.Module):
    """
    Modular LeADMM model with optional pre- and post-processors.

    Pipeline:
        measurement -> pre -> LeADMM -> post -> reconstruction
    """

    def __init__(self, reconstructor, pre=None, post=None):
        super().__init__()

        self.pre = pre
        self.reconstructor = reconstructor
        self.post = post

    def forward(self, measurement, psf):
        if self.pre is not None:
            measurement = self.pre(measurement)

        reconstruction = self.reconstructor(measurement, psf)

        if self.post is not None:
            reconstruction = self.post(reconstruction)

        return reconstruction
