"""
This file contains a simple autoencoder model

Authors
 * Pradnya Kandarkar, 2022
"""


import torch
import torch.nn as nn
import speechbrain as sb
from speechbrain.nnet.CNN import Conv1d, ConvTranspose1d
from speechbrain.nnet.containers import ModuleList
from math import floor


class AudioAutoencoder(torch.nn.Module):
    """This class implements the autoencoder

    Arguments
    ---------
    in_channels : int
      Number of input channels

    Example
    -------
    >>> input_features = torch.rand([5, 10, 40])
    >>> sample_encoder = AudioAutoencoder()
    >>> output = sample_encoder(input_features)
    >>> output.shape
    torch.Size([5, 10, 40])
    """

    def __init__(
        self,
        in_channels=1,
    ):
        super(AudioAutoencoder, self).__init__()

        self.in_channels = in_channels
        self.kernel_size = 7
        self.stride = 3

        # Encoder block - This block has 2 convolutional layers and compresses
        # the provided input into its latent representation
        self.encoder = ModuleList()
        self.encoder.extend([
            Conv1d(
                in_channels=self.in_channels,
                out_channels=self.in_channels * 2,
                kernel_size=self.kernel_size,
                stride=self.stride),
            torch.nn.LeakyReLU(),
            Conv1d(
                in_channels=self.in_channels * 2,
                out_channels=self.in_channels * 4,
                kernel_size=self.kernel_size,
                stride=self.stride)

        ])

        # Decoder block - This block has 2 transposed convolutional layers. They
        # are used to progressively unsample the latent representations to match
        # the dimentionality of the provided input
        self.decoder = ModuleList()
        self.decoder.extend([
            ConvTranspose1d(
                in_channels=self.in_channels * 4,
                out_channels=self.in_channels * 2,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=floor(self.kernel_size / 2)),
            torch.nn.LeakyReLU(),
            ConvTranspose1d(
                in_channels=self.in_channels * 2,
                out_channels=self.in_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=floor(self.kernel_size / 2))
        ])

    def forward(self, x, lens=None):
        """This function processes the input sequence by passing it through an
        encoder and a decoder block to reconstruct the provided input.

        Arguments
        ---------
        x : torch.Tensor
          Input tensor for the audio with shape (batch, time, channel)

        Returns
        ---------
        out : torch.Tensor
          Reconstructed tensor for the audio with shape (batch, time, channel)
        """

        # import pdb; pdb.set_trace()
        # print("\nx.shape before encoder: ", x.shape)
        x = self.encoder(x)
        # print("x.shape after encoder (c1): ", x.shape)
        x = self.decoder(x)
        # print("x.shape after decoder: ", x.shape)

        return x
