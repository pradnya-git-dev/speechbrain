from speechbrain.pretrained import GraphemeToPhoneme
from speechbrain.lobes.models.g2p.dataio import phoneme_pipeline

g2p = GraphemeToPhoneme.from_hparams("speechbrain/soundchoice-g2p")
text = "To be or not to be, that is the question"
phonemes = g2p(text)

phn, phn_encoded_list, phn_encoded = phoneme_pipeline(phonemes)

print(text)
print(phonemes)
print(phn)
print(phn_encoded_list)
print(phn_encoded)
