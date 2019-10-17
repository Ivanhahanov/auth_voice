import librosa as lr
import numpy as np
from keras.layers import Dense, LSTM, Activation
from keras.models import Sequential
from keras.optimizers import Adam

SR = 16000
LENGTH = 16
OVERLAP = 8
FFT = 1024
def filter_audio(audio):
    apower = lr.amplitude_to_db(np.abs(lr.stft(audio, n_fft=2048)), ref=np.max)
    apsums = np.sum(apower, axis=0)**2
    apsums -= np.min(apsums)
    apsums /= np.max(apsums)

    apsums = np.convolve(apsums, np.ones((9,)), 'same')

    apsums -= np.min(apsums)
    apsums /= np.max(apsums)

    apsums = np.array(apsums > 0.35, dtype=bool)

    apsums = np.repeat(apsums, np.ceil(len(audio)/len(apsums)))[:len(audio)]

    return audio[apsums]

def prepare_audio(aname, target=False):
    print('loading %s' % aname)
    audio, _ = lr.load(aname, sr=SR)
    audio = filter_audio(audio)
    data = lr.stft(audio, n_fft=FFT).swapaxes(0,1)
    samples = []

    for i in range(0, len(data) -LENGTH, OVERLAP):
        samples.append(np.abs(data[i:i + LENGTH]))

    results_shape = (len(samples), 1)
    results = np.ones(results_shape) if target else np.zeros(results_shape)
    return np.array(samples), results

voices = [()]

X, Y = prepare_audio(*voices[0])
for voice in voices[1:]:
    dx, dy = prepare_audio(*voice)
    X = np.concatenate((X,dx), axis=0)
    Y = np.concatenate((Y, dy), axis=0)
    del dx, dy

perm = np.random.permutation(len(X))
X = X[perm]
Y = Y[perm]

model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=X.shape[1:]))
model.add(LSTM(64))
model.add(Dense(64))
model.add(Activation('tanh'))
model.add(Dense(16))
model.add(Activation('sigmoid'))
model.add(Dense(1))
model.add(Activation('hard_sigmoid'))

model.compile(Adam(lr=0.005), loss='binary_crossentropy', metrics=['accurancy'])
model.fit(X, Y, epochs=15, batch_size=32, validation_split=0.2)

print(model.evaluate(X, Y))
model.save('model.hdf5')
