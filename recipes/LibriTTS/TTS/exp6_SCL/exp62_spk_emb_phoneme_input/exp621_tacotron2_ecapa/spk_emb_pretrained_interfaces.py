from speechbrain.pretrained import Pretrained
import torch

class MelSpectrogramEncoder(Pretrained):
    """A ready-to-use MelSpectrogramEncoder model
    
    """

    MODULES_NEEDED = ["normalizer", "embedding_model"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def encode_mel_spectrogram(self, mel_spec):
        
        # Fake a batch:
        batch = mel_spec
        if len(mel_spec.shape) == 2:
          batch = mel_spec.unsqueeze(0)
        rel_length = torch.tensor([1.0])
        results = self.encode_batch(batch, rel_length)
        return results

    def encode_batch(self, mel_specs, lens=None):

        # Assign full length if lens is not assigned
        if lens is None:
            lens = torch.ones(mel_specs.shape[0], device=self.device)
        
        mel_specs, lens = mel_specs.to(self.device), lens.to(self.device)
        mel_specs = torch.transpose(mel_specs, 1, 2)
        feats = self.hparams.normalizer(mel_specs, lens)
        encoder_out = self.hparams.embedding_model(feats)
        return encoder_out

    def forward(self, mel_specs, lens):
        """Runs the encoder"""
        return self.encode_batch(mel_specs, lens)