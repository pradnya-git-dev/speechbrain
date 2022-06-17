"""
This file contains the definitions for a ResNet model (used for computing the
embeddings) and a classifier (applied on top of the embeddings to get the
classification)

Authors
 * Nauman Dawalatabad, 2020
 * Mirco Ravanelli, 2020
 * Pradnya Kandarkar, 2022
"""


import torch  # noqa: F401
import torch.nn as nn
import speechbrain as sb
from speechbrain.nnet.pooling import StatisticsPooling, Pooling1d, AdaptivePool
from speechbrain.nnet.CNN import Conv1d
from speechbrain.nnet.linear import Linear
from speechbrain.nnet.containers import Sequential
from speechbrain.nnet.normalization import BatchNorm1d


class ResNetBlock(torch.nn.Module):
    """This class defines a ResNet block. The ResNet block has a predefined
    number of of convolutional layers (here 3), and is used to construct the
    overall ResNet model as required.

    Arguments
    ---------
    in_channels : int
        Expected number of input channels
    intermediate_channels : int
        Expected number of output channels for the corresponding in_channels
    identity_downsample : torch.nn.modules.container.Sequential or NoneType
        A convolutional layer, required to connect layers in case input size or
        the number of input channels changes. If not required, the default value
        is None.
    stride : int
        Expected stride
    """

    def __init__(
        self,
        in_channels,
        intermediate_channels,
        identity_downsample=None,
        stride=1
    ):

        super(ResNetBlock, self).__init__()
        # The expansion size is 4 for ResNet 50, 101, and 150
        self.expansion = 4

        self.conv1 = Conv1d(
            in_channels=in_channels,
            out_channels=intermediate_channels,
            kernel_size=1,
            stride=1)
        self.bn1 = BatchNorm1d(input_size=intermediate_channels)

        self.conv2 = Conv1d(
            in_channels=intermediate_channels,
            out_channels=intermediate_channels,
            kernel_size=3,
            stride=stride)
        self.bn2 = BatchNorm1d(input_size=intermediate_channels)

        self.conv3 = Conv1d(
            in_channels=intermediate_channels,
            out_channels=intermediate_channels * self.expansion,
            kernel_size=1,
            stride=1)
        self.bn3 = BatchNorm1d(
            input_size=intermediate_channels
            * self.expansion)

        self.relu = torch.nn.LeakyReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        """This function processes the input sequence by passing it through a
        ResNet block and applying identity downsample if required. Depending on
        the layer position and the overall ResNet architecture, the number of
        output channels may or may not be the same as the number of input
        channels.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor with shape (batch, time, channel)

        Returns
        ---------
        out : torch.Tensor
            Output tensor with shape (batch, time, channel)
        """
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        out = self.relu(x)

        return out


class ResNet(torch.nn.Module):
    """This class defines the overall ResNet model architecture. The model has
    some fixed and some configurable layers. The "layers parameter is used to
    control the number of configurable layers of the model. It has a list of int
    values and each value indicates the number of  ResNetBlocks for the
    corresponding layer.

    Arguments
    ---------
    ResNetBlock : ResNetBlock
        A class for building blocks of the ResNet layers
    layers : list of ints
        A list indicating the number of ResNetBlocks in each configurable layer
    in_channels : int
        Expected number of input channels
    lin_neurons : int
        Expected number of neurons in the linear layers
    """

    def __init__(
        self,
        ResNetBlock,
        layers,
        in_channels,
        lin_neurons
    ):
        super(ResNet, self).__init__()

        # Predetermined layers
        self.in_channels = 64
        self.conv1 = Conv1d(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=7,
            stride=2
        )
        self.bn1 = BatchNorm1d(input_size=64)
        self.relu = torch.nn.LeakyReLU()

        self.maxpool = Pooling1d(
            pool_type="max",
            kernel_size=3,
            stride=2)

        # Configurable layers
        self.layer1 = self._make_layer(
            ResNetBlock, layers[0], intermediate_channels=64, stride=1
        )

        self.layer2 = self._make_layer(
            ResNetBlock, layers[1], intermediate_channels=128, stride=1
        )

        self.layer3 = self._make_layer(
            ResNetBlock, layers[2], intermediate_channels=256, stride=1
        )

        self.layer4 = self._make_layer(
            ResNetBlock, layers[3], intermediate_channels=512, stride=1
        )

        # To manage variable lengths, average all embeddings over time.
        # Here, we end up with a single vector and can apply a classifier on top
        self.avgpool = AdaptivePool(1)
        self.fc = Linear(
            input_size=512 * 4,
            n_neurons=lin_neurons
        )

    def forward(self, x):
        """This function processes the input sequence by passing it through a
        ResNet model.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor with shape (batch, time, channel)

        Returns
        ---------
        out : torch.Tensor
            Output tensor with shape (batch, lin_neurons)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        out = self.fc(x)

        return out

    def _make_layer(
            self,
            ResNetBlock,
            num_residual_blocks,
            intermediate_channels,
            stride):
        """This function constructs the configurable layers of the ResNet model.

        Arguments
        ---------
        ResNetBlock : ResNetBlock
            A class for building blocks of the ResNet layers
        num_residual_blocks : int
            The number of ResNetBlocks in the configurable layer
        intermediate_channels : int
            Used to adjust the number of input and output channels of the layers
        stride : int
            Stride value

        Returns
        ---------
        speechbrain.nnet.containers.Sequential
            A sequence of ResNetBlocks for the layer
        """
        identity_downsample = None

        # Holds ResNetBlocks for the layer
        rb_layers = []

        # If the stride (input size) changes or the number of channels changes,
        # updates the definition of identity_downsample. This is useful because
        # we need to update the identity value when we add it to the downstream
        # convolutional layers which may have a different shape
        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(
                Conv1d(
                    in_channels=self.in_channels,
                    out_channels=intermediate_channels * 4,
                    kernel_size=1,
                    stride=stride,
                ),
                BatchNorm1d(input_size=intermediate_channels * 4),
            )

        # Adds the first block to the layer
        rb_layers.append(
            ResNetBlock(
                self.in_channels,
                intermediate_channels,
                identity_downsample=identity_downsample,
                stride=stride
            )
        )

        # Updates the number of input channels
        self.in_channels = intermediate_channels * 4

        # Adds the remaining blocks to the layer
        for i in range(num_residual_blocks - 1):
            rb_layers.append(
                ResNetBlock(
                    self.in_channels,
                    intermediate_channels
                )
            )

        return Sequential(*rb_layers)


class ResNetModel(torch.nn.Module):
    """This class can be used to construct a ResNet model with different number
    of layers.

    Arguments
    ---------
    device : str
        The device to be used for the model
    layers : list of ints
        A list indicating the number of ResNetBlocks in each configurable layer
        of the ResNet model
    in_channels : int
        Expected number of input channels
    lin_neurons : int
        Expected number of neurons in the linear layers (embedding dimensions)
    """

    def __init__(
        self,
        device="cpu",
        resnet_layers=[3, 4, 6, 3],
        in_channels=40,
        lin_neurons=512
    ):
        super(ResNetModel, self).__init__()
        self.resnet = ResNet(
            ResNetBlock,
            resnet_layers,
            in_channels,
            lin_neurons)

    def forward(self, x, lens=None):
        """This function processes the input sequence by passing it through a
        ResNet model.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor with shape (batch, time, channel)

        Returns
        ---------
        out : torch.Tensor
            Output tensor with shape (batch, lin_neurons)
        """
        out = self.resnet(x)
        return out


class Classifier(sb.nnet.containers.Sequential):
    """This class implements the last MLP on the top of xvector features.
    Arguments
    ---------
    input_shape : tuple
        Expected shape of an example input.
    activation : torch class
        A class for constructing the activation layers.
    lin_blocks : int
        Number of linear layers.
    lin_neurons : int
        Number of neurons in linear layers.
    out_neurons : int
        Number of output neurons.

    Example
    -------
    >>> input_feats = torch.rand([5, 10, 40])
    >>> compute_xvect = Xvector()
    >>> xvects = compute_xvect(input_feats)
    >>> classify = Classifier(input_shape=xvects.shape)
    >>> output = classify(xvects)
    >>> output.shape
    torch.Size([5, 1, 1211])
    """

    def __init__(
        self,
        input_shape,
        activation=torch.nn.LeakyReLU,
        lin_blocks=1,
        lin_neurons=512,
        out_neurons=1211,
    ):
        super().__init__(input_shape=input_shape)

        self.append(activation(), layer_name="act")
        self.append(sb.nnet.normalization.BatchNorm1d, layer_name="norm")

        if lin_blocks > 0:
            self.append(sb.nnet.containers.Sequential, layer_name="DNN")

        # Adding fully-connected layers
        for block_index in range(lin_blocks):
            block_name = f"block_{block_index}"
            self.DNN.append(
                sb.nnet.containers.Sequential, layer_name=block_name
            )
            self.DNN[block_name].append(
                sb.nnet.linear.Linear,
                n_neurons=lin_neurons,
                bias=True,
                layer_name="linear",
            )
            self.DNN[block_name].append(activation(), layer_name="act")
            self.DNN[block_name].append(
                sb.nnet.normalization.BatchNorm1d, layer_name="norm"
            )

        # Final Softmax classifier
        self.append(
            sb.nnet.linear.Linear, n_neurons=out_neurons, layer_name="out"
        )
        self.append(
            sb.nnet.activations.Softmax(apply_log=True), layer_name="softmax"
        )
