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

    self.linear1_size = int(self.spk_emb_size * 2/3)
    
    # VAE Encoder
    self.enc_lin1 = LinearNorm(self.spk_emb_size, self.linear1_size)
    self.enc_lnorm1 = LayerNorm(input_size=self.linear1_size)

    self.enc_lin2 = LinearNorm(self.linear1_size, self.linear1_size)
    self.enc_lnorm2 = LayerNorm(input_size=self.linear1_size)

    self.enc_lin3 = LinearNorm(self.linear1_size, self.z_spk_emb_size)
    self.enc_lnorm3 = LayerNorm(input_size=self.z_spk_emb_size)

    self.enc_lin4 = LinearNorm(self.z_spk_emb_size, self.z_spk_emb_size)
    self.enc_lnorm4 = LayerNorm(input_size=self.z_spk_emb_size)

    # VAE sampling
    self.mean = LinearNorm(self.z_spk_emb_size, self.z_spk_emb_size)
    self.log_var = LinearNorm(self.z_spk_emb_size, self.z_spk_emb_size)

    self.normal = torch.distributions.Normal(0,1)

    # VAE Decoder
    self.dec_lin1 = LinearNorm(self.z_spk_emb_size, self.z_spk_emb_size)
    self.dec_lnorm1 = LayerNorm(input_size=self.z_spk_emb_size)

    self.dec_lin2 = LinearNorm(self.z_spk_emb_size, self.linear1_size)
    self.dec_lnorm2 = LayerNorm(input_size=self.linear1_size)

    self.dec_lin3 = LinearNorm(self.linear1_size, self.linear1_size)
    self.dec_lnorm3 = LayerNorm(input_size=self.linear1_size)

    self.dec_lin4 = LinearNorm(self.linear1_size, self.spk_emb_size)




  def encode(self, spk_emb):

    out = self.enc_lin1(spk_emb)
    out = self.enc_lnorm1(out)
    out = F.relu(out)
    out = self.enc_lin2(out)
    out = self.enc_lnorm2(out)
    out = F.relu(out)
    out = self.enc_lin3(out)
    out = self.enc_lnorm3(out)
    out = F.relu(out)
    out = self.enc_lin4(out)
    out = self.enc_lnorm4(out)
    out = F.relu(out)
    z_mean = self.mean(out)
    z_log_var = self.log_var(out)

    return z_mean, z_log_var


  def reparameterize(self, z_mean, z_log_var):

    self.normal.loc = self.normal.loc.to(z_mean.device)
    self.normal.scale = self.normal.scale.to(z_mean.device)
    
    random_sample = self.normal.sample([z_mean.shape[0], self.z_spk_emb_size])

    z_spk_emb = z_mean + torch.exp(0.5 * z_log_var) * random_sample

    return z_spk_emb


  def decode(self, z_spk_emb):

    out = self.dec_lin1(z_spk_emb)
    out = self.dec_lnorm1(out)
    out = F.relu(out)

    out = self.dec_lin2(out)
    out = self.dec_lnorm2(out)
    out = F.relu(out)

    out = self.dec_lin3(out)
    out = self.dec_lnorm3(out)
    out = F.relu(out)

    spk_emb_rec = self.dec_lin4(out) 

    return spk_emb_rec
      

  def forward(self, spk_emb):
    z_mean, z_log_var = self.encode(spk_emb)
    z_spk_emb = self.reparameterize(z_mean, z_log_var)
    spk_emb_rec = self.decode(z_spk_emb)
    return spk_emb_rec, z_mean, z_log_var, z_spk_emb
    
  @torch.jit.export
  def infer(self, spk_emb=None):

    if spk_emb != None:
      spk_emb_rec, z_mean, z_log_var, z_spk_emb = self.forward(spk_emb)
      return spk_emb_rec.squeeze(), z_mean.squeeze(), z_spk_emb.squeeze()
    else:
      z_spk_emb = self.normal.sample([self.z_spk_emb_size])
      spk_emb_rec = self.decode(z_spk_emb)
      return spk_emb_rec