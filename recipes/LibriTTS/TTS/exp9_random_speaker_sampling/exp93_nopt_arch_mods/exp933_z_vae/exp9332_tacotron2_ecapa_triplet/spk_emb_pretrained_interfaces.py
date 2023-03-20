import speechbrain
from speechbrain.pretrained import Pretrained
from speechbrain.utils.text_to_sequence import text_to_sequence
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


class MSTacotron2(Pretrained):
    """
    A ready-to-use wrapper for Tacotron2 (text -> mel_spec).
    Arguments
    ---------
    hparams
        Hyperparameters (from HyperPyYAML)
    Example
    -------
    >>> tmpdir_vocoder = getfixture('tmpdir') / "vocoder"
    >>> tacotron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir=tmpdir_vocoder)
    >>> mel_output, mel_length, alignment = tacotron2.encode_text("Mary had a little lamb")
    >>> items = [
    ...   "A quick brown fox jumped over the lazy dog",
    ...   "How much wood would a woodchuck chuck?",
    ...   "Never odd or even"
    ... ]
    >>> mel_outputs, mel_lengths, alignments = tacotron2.encode_batch(items)
    >>> # One can combine the TTS model with a vocoder (that generates the final waveform)
    >>> # Intialize the Vocoder (HiFIGAN)
    >>> tmpdir_tts = getfixture('tmpdir') / "tts"
    >>> hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir=tmpdir_tts)
    >>> # Running the TTS
    >>> mel_output, mel_length, alignment = tacotron2.encode_text("Mary had a little lamb")
    >>> # Running Vocoder (spectrogram-to-waveform)
    >>> waveforms = hifi_gan.decode_batch(mel_output)
    """

    HPARAMS_NEEDED = ["model", "random_sampler"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_cleaners = ["english_cleaners"]
        self.infer = self.hparams.model.infer

    def text_to_seq(self, txt):
        """Encodes raw text into a tensor with a customer text-to-equence fuction
        """
        sequence = text_to_sequence(txt, self.text_cleaners)
        return sequence, len(sequence)

    def encode_batch(self, texts, spk_embs=None):
        """Computes mel-spectrogram for a list of texts
        Texts must be sorted in decreasing order on their lengths
        Arguments
        ---------
        text: List[str]
            texts to be encoded into spectrogram
        Returns
        -------
        tensors of output spectrograms, output lengths and alignments
        """
        # import pdb; pdb.set_trace()
        texts = sorted(texts, key=lambda x: (-len(x), x))

        with torch.no_grad():
            inputs = [
                {
                    "text_sequences": torch.tensor(
                        self.text_to_seq(item)[0], device=self.device
                    )
                }
                for item in texts
            ]
            inputs = speechbrain.dataio.batch.PaddedBatch(inputs)

            lens = [self.text_to_seq(item)[1] for item in texts]
            assert lens == sorted(
                lens, reverse=True
            ), "ipnut lengths must be sorted in decreasing order"
            input_lengths = torch.tensor(lens, device=self.device)

            if spk_embs != None:
              spk_embs = spk_embs.to(self.device)

            spk_emb_rec, z_mean, z_spk_emb = self.hparams.random_sampler.infer(spk_embs)
            spk_emb_rec = spk_emb_rec.to(self.device)

            spk_emb_rec = [spk_emb_rec for i in range(len(texts))]
            z_mean = [z_mean for i in range(len(texts))]
            z_spk_emb = [z_spk_emb for i in range(len(texts))]

            spk_emb_rec = torch.stack(spk_emb_rec)
            z_mean = torch.stack(z_mean)
            z_spk_emb = torch.stack(z_spk_emb)


            mel_outputs_postnet, mel_lengths, alignments = self.infer(
                inputs.text_sequences.data, spk_emb_rec, input_lengths
            )
        return mel_outputs_postnet, mel_lengths, alignments, spk_emb_rec, z_mean, z_spk_emb

    def encode_text(self, text, spk_embs=None):
        """Runs inference for a single text str"""
        return self.encode_batch([text], spk_embs)

    def forward(self, texts, spk_embs=None):
        "Encodes the input texts."

        # import pdb; pdb.set_trace()
        return self.encode_batch(texts, spk_embs)