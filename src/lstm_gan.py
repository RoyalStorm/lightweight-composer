from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
from keras.layers import BatchNormalization
from keras.layers import Dense
from keras.layers import Input, Reshape, CuDNNLSTM, Bidirectional
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import np_utils

from ds_utils import load_dataset
from midi_utils import convert_to_midi


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
    normalized_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    # Normalize input
    # network_input = network_input / float(latent_dim)
    normalized_input = normalized_input / float(latent_dim)

    network_output = np_utils.to_categorical(network_output)

    return network_input, normalized_input, network_output


class ComposerGAN:
    def __init__(self):
        self.notes = load_dataset('kaggle_ds_dump.notes')
        self.latent_dim = len(set(self.notes))

        self.pitch_names = sorted(set(item for item in self.notes))
        self.x, self.x_normalized, self.y = prepare_sequences(self.notes, self.pitch_names, self.latent_dim)

        self.seq_shape = (self.x_normalized.shape[1], self.x_normalized.shape[2])
        self.disc_loss = []
        self.gen_loss = []

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates note sequences
        z = Input(shape=(self.latent_dim,))
        generated_seq = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(generated_seq)

        # The combined model (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_discriminator(self):
        model = Sequential()
        model.add(CuDNNLSTM(512, input_shape=self.seq_shape, return_sequences=True))
        model.add(Bidirectional(CuDNNLSTM(512)))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        seq = Input(shape=self.seq_shape)
        validity = model(seq)

        return Model(seq, validity)

    def build_generator(self):
        model = Sequential()
        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.seq_shape), activation='tanh'))
        model.add(Reshape(self.seq_shape))
        model.summary()

        noise = Input(shape=(self.latent_dim,))
        seq = model(noise)

        return Model(noise, seq)

    def train(self, epochs, batch_size, save_period):
        # Adversarial ground truths
        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        # Training the model
        for epoch in range(1, epochs + 1):
            # Training the discriminator
            # Select a random batch of note sequences
            idx = np.random.randint(0, self.x_normalized.shape[0], batch_size)
            real_seqs = self.x_normalized[idx]

            # noise = np.random.choice(range(484), (batch_size, self.latent_dim))
            # noise = (noise-242)/242
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Generate a batch of new note sequences
            gen_seqs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(real_seqs, real)
            d_loss_fake = self.discriminator.train_on_batch(gen_seqs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            #  Training the Generator
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Train the generator (to have the discriminator label samples as real)
            g_loss = self.combined.train_on_batch(noise, real)

            # Print the progress and save into loss lists
            if epoch % save_period == 0:
                print(f'Epoch {epoch}/{epochs} [D loss: {d_loss[0]}, accuracy: {100 * d_loss[1]}] [G loss: {g_loss}]')
                self.disc_loss.append(d_loss[0])
                self.gen_loss.append(g_loss)

        self.discriminator.save(f'./weights/disc-{epochs}.h5')
        self.generator.save(f'./weights/gen-{epochs}.h5')
        self.plot_loss()

    def generate(self, input_notes):
        # Get pitch names and store in a dictionary
        notes = input_notes
        pitchnames = sorted(set(item for item in notes))
        int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

        # Use random noise to generate sequences
        noise = np.random.normal(0, 1, (1, self.latent_dim))
        predictions = self.generator.predict(noise)

        pred_notes = [x * 242 + 242 for x in predictions[0]]
        pred_notes = [int_to_note[int(x)] for x in pred_notes]

        convert_to_midi(pred_notes)

    def plot_loss(self):
        plt.plot(self.disc_loss, c='red')
        plt.plot(self.gen_loss, c='blue')
        plt.title("GAN Loss per Epoch")
        plt.legend(['Discriminator', 'Generator'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig('GAN_Loss_per_Epoch_final.png', transparent=True)
        plt.close()


if __name__ == '__main__':
    composer_gan = ComposerGAN()
    composer_gan.train(epochs=5_000, batch_size=256, save_period=100)
