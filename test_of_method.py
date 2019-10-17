import librosa as lr
import numpy as np

SR = 16000


def process_audio(audio_name):
    audio, _ = lr.load(audio_name, sr=SR)

    afs = lr.feature.mfcc(audio,
                          sr=SR,
                          n_mfcc=34,
                          n_fft=2048)

    afss = np.sum(afs[2:], axis=-1)
    afss = afss / np.max(np.abs(afss))

    return afss


def confidence(x, y):
    return np.sum((x - y) ** 2)


woman21 = process_audio("woman2.1.wav")
print('1')
woman22 = process_audio("woman2.2.wav")
print('2')
woman11 = process_audio("woman1.1.wav")
print('3')
woman12 = process_audio("woman1.2.wav")
print('4')

print('same:', confidence(woman11, woman11))
print('same:', confidence(woman21, woman22))
print('diff:', confidence(woman11, woman21))
print('diff:', confidence(woman11, woman22))
print('diff:', confidence(woman12, woman21))
print('diff:', confidence(woman12, woman22))
