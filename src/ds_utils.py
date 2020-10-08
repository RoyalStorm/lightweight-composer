import glob
import pickle

from music21 import converter, instrument, note, chord


def dump_dataset(dump_name: str):
    """ Get all the notes and chords from the midi files in the ../midi_songs directory """
    notes = []

    for file in glob.glob('../midi_songs/*.midi'):
        print(f'Parsing {file}')

        midi = converter.parse(file)
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


def load_dataset(dump_name: str):
    """ Load all the notes and chords from the dump in the ../dumps directory """
    with open(f'../dumps/{dump_name}', 'rb') as filepath:
        return pickle.load(filepath)
