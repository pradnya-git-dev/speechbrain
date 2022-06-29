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
    num_layers : int
      Number of layers for the encoder and the decoder
    in_channels : int
      Number of input channels
    kernel_size : int
      Kernel size
    stride : int
      Stride
    """

    def __init__(
        self,
        num_layers=2,
        in_channels=1,
        kernel_size=5,
        stride=3
    ):
        super(AudioAutoencoder, self).__init__()

        # Initalization
        self.num_layers = num_layers
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride

        # Defines the channels to be used in the encoder and decoder blocks
        self.channels = [self.in_channels
                         * (2 ** i) for i in range(self.num_layers + 1)]

        # Encoder block
        self.encoder = ModuleList()

        # Appends convolutional layers and leaky ReLU layers to the encoder
        # block
        for i in range(len(self.channels) - 2):
            self.encoder.extend([
                Conv1d(
                    in_channels=self.channels[i],
                    out_channels=self.channels[i + 1],
                    kernel_size=self.kernel_size,
                    stride=self.stride),
                torch.nn.LeakyReLU()
            ])

        # Appends the last convolutional layer to the encoder block
        self.encoder.append(
            Conv1d(
                in_channels=self.channels[-2],
                out_channels=self.channels[-1],
                kernel_size=self.kernel_size,
                stride=self.stride)
        )

        # Decoder block
        self.decoder = ModuleList()

        # Appends the transposed convolutional layers and leaky ReLU layers to
        # the decoder block
        for i in range(len(self.channels) - 1, 1, -1):
            self.decoder.extend([
                ConvTranspose1d(
                    in_channels=self.channels[i],
                    out_channels=self.channels[i - 1],
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=floor(self.kernel_size / 2)),
                torch.nn.LeakyReLU()
            ])

        # Appends the last transposed convolution layer to the decoder block
        self.decoder.append(
            ConvTranspose1d(
                in_channels=self.channels[1],
                out_channels=self.channels[0],
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=floor(self.kernel_size / 2)),
        )

    def forward(self, x, lens=None):
        """This function processes the input sequence by passing it through an
        encoder and a decoder block to reconstruct the provided input.

        Arguments
        ---------
        x : torch.Tensor
          Input tensor for the audio with shape (batch, time, channel)

        Returns
        ---------
        x : torch.Tensor
          Reconstructed tensor for the audio with shape (batch, time, channel)
        """

        x = self.encoder(x)
        x = self.decoder(x)
        return x
