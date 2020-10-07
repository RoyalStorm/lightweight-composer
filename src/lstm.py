import glob
import os
import pickle
import random as rnd
import string

import numpy
from keras.utils import np_utils
from music21 import converter, instrument, note, chord, stream
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization as BatchNorm
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.python.keras.callbacks import TensorBoard


def dump_dataset(dump_name):
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    notes = []

    for file in glob.glob("../midi_songs/*.midi"):
        midi = converter.parse(file)

        print(f"Parsing {file}")

        notes_to_parse = None

        try:  # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse()
        except:  # file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    with open(f'../dumps/{dump_name}', 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes


def load_dataset(dump_name):
    with open(f'../dumps/{dump_name}', 'rb') as filepath:
        return pickle.load(filepath)


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
    filepath = "./weights/model-{epoch:02d}-{loss:.3f}.h5"
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


def generate_notes(model, input, pitch_names, latent_dim, generated_notes_number):
    """ Generate notes from the neural network based on a sequence of notes """
    # Pick a random sequence from the input as a starting point for the prediction
    start = numpy.random.randint(0, len(input) - 1)

    int_to_note = dict((number, note) for number, note in enumerate(pitch_names))

    pattern = input[start]
    prediction_output = []

    for note_index in range(generated_notes_number):
        prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(latent_dim)

        prediction = model.predict(prediction_input, verbose=0)

        index = numpy.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)

        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    return prediction_output


def generate_sample_name(min_chars: int = 6, max_chars: int = 10):
    return os.path.join('./samples',
                        ''.join(rnd.choices(string.ascii_lowercase, k=rnd.randint(min_chars, max_chars))) + '.mid')


def convert_to_midi(prediction_output):
    """ Convert the output from the prediction to notes and create a midi file
        from the notes """
    offset = 0
    output_notes = []

    # Create note and chord objects based on the values generated by the model
    for pattern in prediction_output:
        # Pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # Pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # Increase offset each iteration so that notes do not stack
        offset += 0.5

    midi_stream = stream.Stream(output_notes)

    midi_stream.write('midi', fp=generate_sample_name(6, 10))


if __name__ == '__main__':
    # First of all, we need to prepare dataset and dumps it on disk, only one time!
    # notes = dump_dataset('magenta_ds_dump.notes')

    # Or, if dataset already created
    notes = load_dataset('kaggle_ds_dump.notes')
    pitch_names = sorted(set(item for item in notes))
    latent_dim = len(set(notes))
    x, x_normalized, y = prepare_sequences(notes, pitch_names, latent_dim)

    # Build model
    model = build_net(x_normalized, latent_dim)

    # You can plot model architecture
    # plot_model_architecture(model)

    # If you want contain training from current weights
    model.load_weights('./best/best.h5')

    # Train model
    train(model, x_normalized, y, epochs=4500, batch_size=128, save_period=250)

    # And finally generate sample
    # raw_notes = generate_notes(model, x, pitch_names, latent_dim, generated_notes_number=500)
    # convert_to_midi(raw_notes)
