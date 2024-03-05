# From https://github.com/SeanNaren/deepspeech.pytorch, relying on https://github.com/parlance/ctcdecode for Beam Search decoder.

from typing import List

import torch
from six.moves import xrange


class Decoder:
    """
    Basic decoder class from which all other decoders inherit. Implements several
    helper functions. Subclasses should implement the decode() method.

    Arguments:
        classes (list): mapping from integers to characters.
        blank_index (int, optional): index for the blank '_' character. Defaults to 0.
    """

    def __init__(self, classes, blank_index=0):
        self.classes = classes
        self.indices_to_classes = dict([(i, c) for (i, c) in enumerate(classes)])
        self.blank_index = blank_index
        space_index = len(classes)  # To prevent errors in decode, we add an out of bounds index for the space
        if ' ' in classes:
            space_index = classes.index(' ')
        self.space_index = space_index

    def decode(self, probs, sizes=None):
        """
        Given a matrix of character probabilities, returns the decoder's
        best guess of the transcription

        Arguments:
            probs: Tensor of character probabilities, where probs[c,t]
                            is the probability of character c at time t
            sizes(optional): Size of each sequence in the mini-batch
        Returns:
            string: sequence of the model's best guess for the transcription
        """
        raise NotImplementedError


class BeamCTCDecoder(Decoder):
    def __init__(self,
                 classes,
                 lm_path=None,
                 alpha=0,
                 beta=0,
                 cutoff_top_n=40,
                 cutoff_prob=1.0,
                 beam_width=100,
                 num_processes=4,
                 blank_index=0):
        super(BeamCTCDecoder, self).__init__(classes)
        try:
            from ctcdecode import CTCBeamDecoder
        except ImportError:
            raise ImportError("BeamCTCDecoder requires paddledecoder package.")
        classes = list(classes)  # Ensure classes are a list before passing to decoder
        self._decoder = CTCBeamDecoder(classes, lm_path, alpha, beta, cutoff_top_n, cutoff_prob, beam_width,
                                       num_processes, blank_index)

    def convert_to_strings(self, out, seq_len):
        results = []
        for b, batch in enumerate(out):
            utterances = []
            for p, utt in enumerate(batch):
                size = seq_len[b][p]
                if size > 0:
                    transcript = ''.join(map(lambda x: self.indices_to_classes[x.item()], utt[0:size]))
                else:
                    transcript = ''
                utterances.append(transcript)
            results.append(utterances)
        return results

    def convert_tensor(self, offsets, sizes):
        results = []
        for b, batch in enumerate(offsets):
            utterances = []
            for p, utt in enumerate(batch):
                size = sizes[b][p]
                if sizes[b][p] > 0:
                    utterances.append(utt[0:size])
                else:
                    utterances.append(torch.tensor([], dtype=torch.int))
            results.append(utterances)
        return results

    def decode(self, probs, sizes=None):
        """Decodes probability output using ctcdecode package.

        Args:
            probs: Tensor of character probabilities, where probs[c,t] is the probability of character c at time t
            sizes: Size of each sequence in the mini-batch

        Returns:
            string: sequences of the model's best guess for the transcription
        """
        probs = probs.cpu()
        out, scores, offsets, seq_lens = self._decoder.decode(probs, sizes)

        strings = self.convert_to_strings(out, seq_lens)
        offsets = self.convert_tensor(offsets, seq_lens)
        return strings, offsets


class GreedyDecoder(Decoder):
    def __init__(self, classes, blank_index=0):
        super(GreedyDecoder, self).__init__(classes, blank_index)


    #@profile
    def convert_to_strings(self,
                           sequences,
                           sizes=None,
                           remove_repetitions=False) -> (List[str], List[List[int]]):
        """Given a list of numeric sequences, returns the corresponding strings"""
        strings = []
        offsets = []
        for x in xrange(len(sequences)):
            seq_len = sizes[x] if sizes is not None else len(sequences[x])
            string, string_offsets = self.process_string(sequences[x], seq_len, remove_repetitions)
            strings.append(string)  # We only return one path
            offsets.append(string_offsets)

        return strings, offsets

    #@profile
    def process_string(self,
                       sequence: torch.tensor,
                       size: int,
                       remove_repetitions=False) -> (str, torch.tensor):
        string = ''
        offsets = []
        for i in range(size):
            char = self.indices_to_classes[sequence[i].item()]
            if char != self.indices_to_classes[self.blank_index]:
                # if this char is a repetition and remove_repetitions=true, then skip
                if remove_repetitions and i != 0 and char == self.indices_to_classes[sequence[i - 1].item()]:
                    pass
                elif char == self.classes[self.space_index]:
                    string += ' '
                    offsets.append(i)
                else:
                    string += char
                    offsets.append(i)
        return string, offsets

    #@profile
    def decode(self, probs, sizes=None, remove_repetitions: bool = True) -> (List[str], List[torch.tensor]):
        """
        Returns the argmax decoding given the probability matrix. Removes
        repeated elements in the sequence, as well as blanks.

        Arguments:
            probs: Tensor of character probabilities from the network. Expected shape of batch x seq_length x output_dim
            sizes(optional): Size of each sequence in the mini-batch
        Returns:
            strings: sequences of the model's best guess for the transcription on inputs
            offsets: time step per character predicted
        """
        max_probs = torch.argmax(probs, 2)

        return self.convert_to_strings(sequences=max_probs.view(max_probs.size(0), max_probs.size(1)),
                                       sizes=sizes,
                                       remove_repetitions=remove_repetitions)
