import speechbrain
from speechbrain.pretrained import Pretrained
from speechbrain.utils.text_to_sequence import text_to_sequence
import torch
from speechbrain.dataio.dataio import length_to_mask
import os
from speechbrain.pretrained import GraphemeToPhoneme
import re, string

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
    >>> fastspeech2 = FastSpeech2.from_hparams(source="speechbrain/tts-fastspeech2-ljspeech", savedir=tmpdir_tts)   # doctest: +SKIP
    >>> mel_outputs, durations, pitch, energy = fastspeech2.encode_text("Mary had a little lamb")   # doctest: +SKIP
    >>> items = [
    ...   "A quick brown fox jumped over the lazy dog",
    ...   "How much wood would a woodchuck chuck?",
    ...   "Never odd or even"
    ... ]
    >>> mel_outputs, durations, pitch, energy = fastspeech2.encode_batch(items) # doctest: +SKIP
    >>>
    >>> # One can combine the TTS model with a vocoder (that generates the final waveform)
    >>> # Intialize the Vocoder (HiFIGAN)
    >>> tmpdir_vocoder = getfixture('tmpdir') / "vocoder"
    >>> hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-libritts-16kHz", savedir=tmpdir_vocoder)
    >>> # Running the TTS
    >>> mel_output, durations, pitch, energy = fastspeech2.encode_text("Mary had a little lamb")    # doctest: +SKIP
    >>> # Running Vocoder (spectrogram-to-waveform)
    >>> waveforms = hifi_gan.decode_batch(mel_output)   # doctest: +SKIP
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

    def encode_batch(self, texts, pace=1.0, pitch_rate=1.0, energy_rate=1.0):
        """Computes mel-spectrogram for a list of texts
        Arguments
        ---------
        texts : List[str]
            texts to be encoded into spectrogram
        pace : float
            pace for the speech synthesis
        pitch_rate : float
            scaling factor for phoneme pitches
        energy_rate : float
            scaling factor for phoneme energies
        Returns
        -------
        tensors of output spectrograms, output lengths, pitches and energies.
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
            inputs = speechbrain.dataio.batch.PaddedBatch(inputs).to(
                self.device
            )
            (
                mel_outputs,
                _,
                durations,
                pitch,
                _,
                energy,
                _,
                _,
            ) = self.hparams.model(
                inputs.phoneme_sequences.data,
                pace=pace,
                pitch_rate=pitch_rate,
                energy_rate=energy_rate,
            )

            # Transposes to make in compliant with HiFI GAN expected format
            mel_outputs = mel_outputs.transpose(-1, 1)

        return mel_outputs, durations, pitch, energy

    def encode_text(self, text, pace=1.0, pitch_rate=1.0, energy_rate=1.0):
        """Runs inference for a single text str
        Arguments
        ---------
        text : str
            text to be encoded into spectrogram
        pace : float
            pace for the speech synthesis
        pitch_rate : float
            scaling factor for phoneme pitches
        energy_rate : float
            scaling factor for phoneme energies
        """
        return self.encode_batch(
            [text], pace=pace, pitch_rate=pitch_rate, energy_rate=energy_rate
        )

    def forward(self, texts, pace=1.0, pitch_rate=1.0, energy_rate=1.0):
        """Encodes the input texts.
        Arguments
        ---------
        texts : List[str]
            texts to be encoded into spectrogram
        pace : float
            pace for the speech synthesis
        pitch_rate : float
            scaling factor for phoneme pitches
        energy_rate : float
            scaling factor for phoneme energies
        """
        return self.encode_batch(
            texts, pace=pace, pitch_rate=pitch_rate, energy_rate=energy_rate
        )


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
    >>>	# Initialize TTS (tacotron2)
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
        spectrogram: torch.Tensor
            Batch of mel-spectrograms [batch, mels, time]
        mel_lens: torch.tensor
            A list of lengths of mel-spectrograms for the batch
            Can be obtained from the output of Tacotron/FastSpeech
        hop_len: int
            hop length used for mel-spectrogram extraction
            should be the same value as in the .yaml file
        Returns
        -------
        waveforms: torch.Tensor
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
        spectrogram: torch.Tensor
            mel-spectrogram [mels, time]
        Returns
        -------
        waveform: torch.Tensor
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