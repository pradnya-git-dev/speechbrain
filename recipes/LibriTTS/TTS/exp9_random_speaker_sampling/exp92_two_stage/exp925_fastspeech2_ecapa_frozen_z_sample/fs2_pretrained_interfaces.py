import speechbrain
from speechbrain.pretrained import Pretrained
from speechbrain.utils.text_to_sequence import text_to_sequence
import torch
from speechbrain.dataio.dataio import length_to_mask
import os
from speechbrain.pretrained import GraphemeToPhoneme

class HIFIGAN(Pretrained):
    """
    A ready-to-use wrapper for HiFiGAN (mel_spec -> waveform).
    Arguments
    ---------
    hparams
        Hyperparameters (from HyperPyYAML)
    Example
    -------
    >>> tmpdir_vocoder = getfixture('tmpdir') / "vocoder"
    >>> hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir=tmpdir_vocoder)
    >>> mel_specs = torch.rand(2, 80,298)
    >>> waveforms = hifi_gan.decode_batch(mel_specs)
    >>> # You can use the vocoder coupled with a TTS system
    >>>	# Intialize TTS (tacotron2)
    >>> tmpdir_tts = getfixture('tmpdir') / "tts"
    >>>	tacotron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir=tmpdir_tts)
    >>>	# Running the TTS
    >>>	mel_output, mel_length, alignment = tacotron2.encode_text("Mary had a little lamb")
    >>>	# Running Vocoder (spectrogram-to-waveform)
    >>>	waveforms = hifi_gan.decode_batch(mel_output)
    """

    HPARAMS_NEEDED = ["generator"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.infer = self.hparams.generator.inference
        self.first_call = True

    def decode_batch(self, spectrogram, mel_lens=None, hop_len=None):
        """Computes waveforms from a batch of mel-spectrograms
        Arguments
        ---------
        spectrogram: torch.tensor
            Batch of mel-spectrograms [batch, mels, time]
        mel_lens: torch.tensor
            A list of lengths of mel-spectrograms for the batch
            Can be obtained from the output of Tacotron/FastSpeech
        hop_len: int
            hop length used for mel-spectrogram extraction
            should be the same value as in the .yaml file
        Returns
        -------
        waveforms: torch.tensor
            Batch of mel-waveforms [batch, 1, time]
        """
        # Prepare for inference by removing the weight norm
        if self.first_call:
            self.hparams.generator.remove_weight_norm()
            self.first_call = False
        with torch.no_grad():
            waveform = self.infer(spectrogram.to(self.device))

        # Mask the noise caused by padding during batch inference
        if mel_lens is not None and hop_len is not None:
            waveform = self.mask_noise(waveform, mel_lens, hop_len)

        return waveform

    def mask_noise(self, waveform, mel_lens, hop_len):
        """Mask the noise caused by padding during batch inference
        Arguments
        ---------
        wavform: torch.tensor
            Batch of generated waveforms [batch, 1, time]
        mel_lens: torch.tensor
            A list of lengths of mel-spectrograms for the batch
            Can be obtained from the output of Tacotron/FastSpeech
        hop_len: int
            hop length used for mel-spectrogram extraction
            same value as in the .yaml file
        Returns
        -------
        waveform: torch.tensor
            Batch of waveforms without padded noise [batch, 1, time]
        """
        waveform = waveform.squeeze(1)
        # the correct audio length should be hop_len * mel_len
        mask = length_to_mask(
            mel_lens * hop_len, waveform.shape[1], device=waveform.device
        ).bool()
        waveform.masked_fill_(~mask, 0.0)
        return waveform.unsqueeze(1)

    def decode_spectrogram(self, spectrogram):
        """Computes waveforms from a single mel-spectrogram
        Arguments
        ---------
        spectrogram: torch.tensor
            mel-spectrogram [mels, time]
        Returns
        -------
        waveform: torch.tensor
            waveform [1, time]
        audio can be saved by:
        >>> waveform = torch.rand(1, 666666)
        >>> sample_rate = 22050
        >>> torchaudio.save(str(getfixture('tmpdir') / "test.wav"), waveform, sample_rate)
        """
        if self.first_call:
            self.hparams.generator.remove_weight_norm()
            self.first_call = False
        with torch.no_grad():
            waveform = self.infer(spectrogram.unsqueeze(0).to(self.device))
        return waveform.squeeze(0)

    def forward(self, spectrogram):
        "Decodes the input spectrograms"
        return self.decode_batch(spectrogram)


class FastSpeech2(Pretrained):
    """
    A ready-to-use wrapper for Fastspeech2 (text -> mel_spec).
    Arguments
    ---------
    hparams
        Hyperparameters (from HyperPyYAML)
    Example
    -------
    >>> tmpdir_tts = getfixture('tmpdir') / "tts"
    >>> fastspeech2 = Fastspeech2.from_hparams(source="speechbrain/tts-fastspeecg2-ljspeech", savedir=tmpdir_tts)
    >>> mel_outputs, durations, pitch, energy = fastspeech2.encode_text("Mary had a little lamb")
    >>> items = [
    ...   "A quick brown fox jumped over the lazy dog",
    ...   "How much wood would a woodchuck chuck?",
    ...   "Never odd or even"
    ... ]
    >>> mel_outputs, durations, pitch, energy = fastspeech2.encode_batch(items)
    >>>
    >>> # One can combine the TTS model with a vocoder (that generates the final waveform)
    >>> # Intialize the Vocoder (HiFIGAN)
    >>> tmpdir_vocoder = getfixture('tmpdir') / "vocoder"
    >>> hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir=tmpdir_vocoder)
    >>> # Running the TTS
    >>> mel_output, mel_length, alignment = fastspeech2.encode_text("Mary had a little lamb")
    >>> # Running Vocoder (spectrogram-to-waveform)
    >>> waveforms = hifi_gan.decode_batch(mel_output)
    """

    HPARAMS_NEEDED = ["model", "input_encoder"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        lexicon = self.hparams.lexicon
        lexicon = ["@@"] + lexicon
        self.input_encoder = self.hparams.input_encoder
        self.input_encoder.update_from_iterable(lexicon, sequence_input=False)
        self.input_encoder.add_unk()
        self.g2p = GraphemeToPhoneme.from_hparams("speechbrain/soundchoice-g2p")

    def encode_batch(self, texts, pace=1.1):
        """Computes mel-spectrogram for a list of texts

        Arguments
        ---------
        text: List[str]
            texts to be encoded into spectrogram
        pace: float
            pace for the speech synthesis
            
        Returns
        -------
        tensors of output spectrograms, output lengths and alignments
        """

        # Converts texts to their respective phoneme sequences
        phoneme_seqs = list()
        for text in texts:
          phoneme_seq = self.g2p(text)
          phoneme_seq = " ".join(phoneme_seq)
          phoneme_seqs.append(phoneme_seq)

        # Sorts phoneme sequences in descending order of length
        phoneme_seqs = sorted(phoneme_seqs, key=lambda x: (-len(x), x))

        with torch.no_grad():
            inputs = [
                {
                    "phoneme_sequences": self.input_encoder.encode_sequence_torch(
                        item.split()
                    ).int()
                }
                for item in phoneme_seqs
            ]
            inputs = speechbrain.dataio.batch.PaddedBatch(inputs).to(self.device)
            mel_outputs, _, durations, pitch, energy, _ = self.hparams.model(
                inputs.phoneme_sequences.data, pace=pace
            )

            # Transposes to make in compliant with HiFI GAN expected format
            mel_outputs = mel_outputs.transpose(-1, 1)

        return mel_outputs, durations, pitch, energy

    def encode_text(self, text, pace=1.1):
        """Runs inference for a single text str
        Arguments
        ---------
        text: str
            text to be encoded into spectrogram
        pace: float
            pace for the speech synthesis
        """
        return self.encode_batch([text], pace=pace)

    def forward(self, texts, pace=1.1):
        """Encodes the input texts.
        Arguments
        ---------
        text: List[str]
            texts to be encoded into spectrogram
        pace: float
            pace for the speech synthesis
        """
        return self.encode_batch(texts, pace=pace)

class MSFastSpeech2(Pretrained):
    """
    A ready-to-use wrapper for Fastspeech2 (text -> mel_spec).
    Arguments
    ---------
    hparams
        Hyperparameters (from HyperPyYAML)
    Example
    -------
    >>> tmpdir_tts = getfixture('tmpdir') / "tts"
    >>> fastspeech2 = Fastspeech2.from_hparams(source="speechbrain/tts-fastspeecg2-ljspeech", savedir=tmpdir_tts)
    >>> mel_outputs, durations, pitch, energy = fastspeech2.encode_text("Mary had a little lamb")
    >>> items = [
    ...   "A quick brown fox jumped over the lazy dog",
    ...   "How much wood would a woodchuck chuck?",
    ...   "Never odd or even"
    ... ]
    >>> mel_outputs, durations, pitch, energy = fastspeech2.encode_batch(items)
    >>>
    >>> # One can combine the TTS model with a vocoder (that generates the final waveform)
    >>> # Intialize the Vocoder (HiFIGAN)
    >>> tmpdir_vocoder = getfixture('tmpdir') / "vocoder"
    >>> hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir=tmpdir_vocoder)
    >>> # Running the TTS
    >>> mel_output, mel_length, alignment = fastspeech2.encode_text("Mary had a little lamb")
    >>> # Running Vocoder (spectrogram-to-waveform)
    >>> waveforms = hifi_gan.decode_batch(mel_output)
    """

    HPARAMS_NEEDED = ["model", "input_encoder", "random_sampler"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        lexicon = self.hparams.lexicon
        lexicon = ["@@"] + lexicon
        self.input_encoder = self.hparams.input_encoder
        self.input_encoder.update_from_iterable(lexicon, sequence_input=False)
        self.input_encoder.add_unk()
        self.g2p = GraphemeToPhoneme.from_hparams("speechbrain/soundchoice-g2p")

    def encode_batch(self, texts, spk_embs=None, pace=1.1):
        """Computes mel-spectrogram for a list of texts

        Arguments
        ---------
        text: List[str]
            texts to be encoded into spectrogram
        pace: float
            pace for the speech synthesis
            
        Returns
        -------
        tensors of output spectrograms, output lengths and alignments
        """

        # Converts texts to their respective phoneme sequences
        phoneme_seqs = list()
        for text in texts:
          phoneme_seq = self.g2p(text)
          phoneme_seq = " ".join(phoneme_seq)
          phoneme_seqs.append(phoneme_seq)

        # Sorts phoneme sequences in descending order of length
        phoneme_seqs = sorted(phoneme_seqs, key=lambda x: (-len(x), x))

        with torch.no_grad():
            inputs = [
                {
                    "phoneme_sequences": self.input_encoder.encode_sequence_torch(
                        item.split()
                    ).int()
                }
                for item in phoneme_seqs
            ]
            inputs = speechbrain.dataio.batch.PaddedBatch(inputs).to(self.device)

            if spk_embs != None:
              spk_embs = spk_embs.to(self.device)

            z_spk_embs, z_mean, mlp_out = self.hparams.random_sampler.infer(spk_embs)
            z_spk_embs = z_spk_embs.to(self.device)

            z_spk_embs = [z_spk_embs for i in range(len(texts))]
            z_mean = [z_mean for i in range(len(texts))]
            mlp_out = [mlp_out for i in range(len(texts))]

            z_spk_embs = torch.stack(z_spk_embs)
            z_mean = torch.stack(z_mean)
            mlp_out = torch.stack(mlp_out)

            mel_outputs, _, durations, pitch, energy, _ = self.hparams.model(
                inputs.phoneme_sequences.data, z_spk_embs, pace=pace
            )

            # Transposes to make in compliant with HiFI GAN expected format
            mel_outputs = mel_outputs.transpose(-1, 1)

        return mel_outputs, durations, pitch, energy, z_spk_embs, z_mean, mlp_out

    def encode_text(self, text, spk_embs=None, pace=1.1):
        """Runs inference for a single text str
        Arguments
        ---------
        text: str
            text to be encoded into spectrogram
        pace: float
            pace for the speech synthesis
        """
        return self.encode_batch([text], spk_embs, pace=pace)

    def forward(self, texts, spk_embs=None, pace=1.1):
        """Encodes the input texts.
        Arguments
        ---------
        text: List[str]
            texts to be encoded into spectrogram
        pace: float
            pace for the speech synthesis
        """
        return self.encode_batch(texts, spk_embs, pace=pace)