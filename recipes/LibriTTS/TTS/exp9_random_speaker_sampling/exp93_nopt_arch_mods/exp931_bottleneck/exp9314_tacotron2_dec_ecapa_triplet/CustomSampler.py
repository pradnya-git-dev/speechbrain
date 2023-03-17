from math import sqrt
from speechbrain.nnet.loss.guidedattn_loss import GuidedAttentionLoss
import torch
from torch import nn
from torch.nn import functional as F
from collections import namedtuple
import speechbrain as sb
from speechbrain.pretrained import EncoderClassifier
from speechbrain.nnet.linear import Linear
from speechbrain.nnet.CNN import Conv1d
from speechbrain.processing.speech_augmentation import Resample
from speechbrain.utils.data_utils import batch_pad_right
import pickle
import random
from speechbrain.nnet.normalization import LayerNorm


class LinearNorm(torch.nn.Module):
    """A linear layer with Xavier initialization

    Arguments
    ---------
    in_dim: int
        the input dimension
    out_dim: int
        the output dimension
    bias: bool
        whether or not to use a bias
    w_init_gain: linear
        the weight initialization gain type (see torch.nn.init.calculate_gain)

    Example
    -------
    >>> import torch
    >>> from speechbrain.lobes.models.Tacotron2 import Tacotron2
    >>> layer = LinearNorm(in_dim=5, out_dim=3)
    >>> x = torch.randn(3, 5)
    >>> y = layer(x)
    >>> y.shape
    torch.Size([3, 3])
    """

    def __init__(self, in_dim, out_dim, bias=True, w_init_gain="linear"):
        super().__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain),
        )

    def forward(self, x):
        """Computes the forward pass

        Arguments
        ---------
        x: torch.Tensor
            a (batch, features) input tensor


        Returns
        -------
        output: torch.Tensor
            the linear layer output

        """
        return self.linear_layer(x)


class Sampler(nn.Module):
  """
  This module takes a speaker embedding as an input. A transformation is applied
  to the speaker embedding to map it to a latent space thant should be Gaussian.
  The output of this module is used as the speaker embedding for the TTS.
  """
  def __init__(
    self,
    spk_emb_size,
    z_spk_emb_size,
  ):
    super().__init__()
    self.spk_emb_size = spk_emb_size
    self.z_spk_emb_size = z_spk_emb_size

    # import pdb; pdb.set_trace()
    self.linear1_size = int(self.spk_emb_size * 2/3)
    self.linear2_size = int(self.z_spk_emb_size * 4)
    self.linear3_size = int(self.z_spk_emb_size * 3)
    self.linear4_size = int(self.z_spk_emb_size * 2)

    self.linear1 = LinearNorm(self.spk_emb_size, self.linear1_size)
    self.lnorm1 = LayerNorm(input_size=self.linear1_size)
    self.linear2 = LinearNorm(self.linear1_size, self.linear2_size)
    self.lnorm2 = LayerNorm(input_size=self.linear2_size)
    self.linear3 = LinearNorm(self.linear2_size, self.linear3_size)
    self.lnorm3 = LayerNorm(input_size=self.linear3_size)
    self.linear4 = LinearNorm(self.linear3_size, self.linear4_size)
    self.lnorm4 = LayerNorm(input_size=self.linear4_size)
    self.mean = LinearNorm(self.linear4_size, self.z_spk_emb_size)
    self.log_var = LinearNorm(self.linear4_size, self.z_spk_emb_size)
    
    self.normal = torch.distributions.Normal(0,1)

  def forward(self, spk_embs):

    
    out = self.linear1(spk_embs)
    out = self.lnorm1(out)
    out = F.relu(out)
    out = self.linear2(out)
    out = self.lnorm2(out)
    out = F.relu(out)
    out = self.linear3(out)
    out = self.lnorm3(out)
    out = F.relu(out)
    out = self.linear4(out)
    out = self.lnorm4(out)
    mlp_out = F.relu(out)
    z_mean = self.mean(mlp_out)
    z_log_var = self.log_var(mlp_out)

    # ToDo: Move these to GPU if available
    self.normal.loc = self.normal.loc.to(spk_embs.device)
    self.normal.scale = self.normal.scale.to(spk_embs.device)
    
    random_sample = self.normal.sample([spk_embs.shape[0], self.z_spk_emb_size])

    z_spk_embs = z_mean + torch.exp(0.5 * z_log_var) * random_sample

    return z_spk_embs, z_mean, z_log_var, mlp_out
    
  @torch.jit.export
  def infer(self, spk_embs=None):

    if spk_embs != None:
      z_spk_embs, z_mean, z_log_var, mlp_out = self.forward(spk_embs)
      return z_spk_embs.squeeze(), z_mean.squeeze(), mlp_out.squeeze()
    else:
      z_spk_embs = self.normal.sample([self.z_spk_emb_size])
      return z_spk_embs