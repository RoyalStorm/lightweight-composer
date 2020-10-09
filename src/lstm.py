import numpy
from keras.utils import np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization as BatchNorm
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.python.keras.callbacks import TensorBoard

from ds_utils import load_dataset
from midi_utils import generate_notes, convert_to_midi


def prepare_sequences(notes, pitch_names, latent_dim):
    """ Prepare the sequences used by the Neural Network """

    # Create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitch_names))

    network_input = []
    network_output = []

    sequence_length = 100
    # Create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # Reshape the input into a format compatible with LSTM layers
    # network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    normalized_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    # Normalize input
    # network_input = network_input / float(latent_dim)
    normalized_input = normalized_input / float(latent_dim)

    network_output = np_utils.to_categorical(network_output)

    return network_input, normalized_input, network_output


def build_net(input, latent_dim):
    """ Create the structure of the neural network """
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(input.shape[1], input.shape[2]),
        recurrent_dropout=0.3,
        return_sequences=True
    ))
    model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3, ))
    model.add(LSTM(512))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(latent_dim))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model


def train(model, x, y, epochs, batch_size, save_period):
    """ Train the neural network """
    filepath = './weights/model-{epoch:02d}-{loss:.3f}.h5'
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min',
        period=save_period
    )
    tensorboard = TensorBoard(
        log_dir='./logs',
        histogram_freq=0,
        write_graph=True,
        write_images=False
    )
    callbacks_list = [checkpoint, tensorboard]

    model.fit(x, y, epochs=epochs, batch_size=batch_size, callbacks=callbacks_list)


def plot_model_architecture(model):
    plot_model(model, 'model_architecture.png', show_shapes=True)


if __name__ == '__main__':
    # First of all, we need to prepare dataset and dumps it on disk, only one time!
    # notes = dump_dataset('kaggle_ds_dump.notes')

    # Or, if dataset already created
    notes = load_dataset('kaggle_ds_dump.notes')
    pitch_names = sorted(set(item for item in notes))
    latent_dim = len(set(notes))
    x, x_normalized, y = prepare_sequences(notes, pitch_names, latent_dim)

    # Build model
    model = build_net(x_normalized, latent_dim)

    # If you want contain training from current weights
    model.load_weights('./best/best.h5')

    # Train model
    # train(model, x_normalized, y, epochs=4500, batch_size=128, save_period=250)

    # And finally generate sample
    raw_notes = generate_notes(model, x, pitch_names, latent_dim, generated_notes_number=500)
    convert_to_midi(raw_notes)
