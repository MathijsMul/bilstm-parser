from arc_hybrid import Stack, Buffer, Arcs, Configuration
import numpy as np
import pprint
from random import shuffle
from collections import Counter

class ConllLoader:
    def __init__(self, input_file, oracle, alpha=0.25, features='default', max_num_sentences=np.inf):
        self.file = input_file
        self.compute_gold = oracle
        self.alpha = alpha # parameter for word dropout
        self.sentences = []
        self.vocab = []
        self.pos_tags = []
        self.arc_labels = []

        if features == 'default':
            # def features:  top 3 items on the stack and the first item on the buffer
            self.features = {'stack' : [-3, -2, -1],
                             'buffer' : [0]}
        self.max_sentence_idx = max_num_sentences

    def load_file(self):
        print('Loading data from %s' % self.file)
        with open(self.file, 'r') as f:
            lines = f.readlines()
            num_lines = len(lines)
            sentence = []
            for idx, line in enumerate(lines):
                if line == '\n' or idx == num_lines - 1:
                    if idx == num_lines - 1:
                        sentence.append(line.split('\t'))
                    sentence_info = self.analyse_sentence(sentence)
                    if self.compute_gold:
                        oracle = self.infer_gold_transitions(sentence_info)
                        sentence_info['oracle'] = {'features' : oracle[0],
                                                   'transitions' : oracle[1]}

                    self.sentences.append(sentence_info)
                    self.vocab.extend(sentence_info['words'])
                    self.pos_tags.extend(sentence_info['pos_tags'])
                    self.arc_labels.extend([arc[2] for arc in sentence_info['arcs']])

                    sentence = []

                    if len(self.sentences) >= self.max_sentence_idx:
                        break
                else:
                    sentence.append(line.split('\t'))

        self.num_samples = len(self.sentences)
        print('Total: %i sentences' % self.num_samples)

        self.num_tokens = len(self.vocab)
        self.avg_sentence_length = self.num_tokens / self.num_samples
        print('Avg. sentence length: %.3f' % self.avg_sentence_length)

        # compute word dropout probabilities (for replacement with UNK-WORD
        counter = Counter(self.vocab)
        self.word_dropout_probabilities = counter
        for key in self.word_dropout_probabilities:
            self.word_dropout_probabilities[key] = self.alpha / (self.word_dropout_probabilities[key] + self.alpha)

        self.vocab = list(set(self.vocab)) + ['UNK-WORD']
        self.pos_tags = list(set(self.pos_tags))
        self.arc_labels = list(set(self.arc_labels))
        self.sentences_unshuffled = list(self.sentences)

    def analyse_sentence(self, sentence):
        words = ['ROOT'] + [s[1] for s in sentence] # add ROOT
        word_to_idx = {word : idx for idx, word in enumerate(words)}
        pos_tags = ['ROOT'] + [s[4] for s in sentence]
        arcs = [(s[6], s[0], s[7]) for s in sentence]
        info_dict = {'words' : words,
                     'word_to_idx' : word_to_idx,
                     'pos_tags' : pos_tags,
                     'arcs' : arcs,
                     'length' : len(words)}
        return(info_dict)

    def infer_gold_transitions(self, sentence_info, sanity_check=True):
        c = Configuration([str(i) for i in range(sentence_info['length'])])
        arcs_given = Arcs()
        arcs_given.load(list(sentence_info['arcs']))
        if sanity_check:
            arcs_given_original = list(sentence_info['arcs'])

        features = [c.extract_features(self.features)]
        # always start with shift
        c.shift()
        transitions = ['shift']

        while ((len(c.stack) + len(c.buffer)) > 1):
            features += [c.extract_features(self.features)]

            if not c.stack[-1] == '0': # never reduce ROOT
                if arcs_given.contains(c.stack[-2], c.stack[-1]):
                    if not arcs_given.child_still_has_children(c.stack[-1]):
                        label = arcs_given.get_label(c.stack[-2], c.stack[-1])
                        transitions += [('right', label)]
                        arcs_given.contents.remove((c.stack[-2], c.stack[-1], label))
                        c.right_arc(label)
                    elif len(c.buffer) > 0:
                        transitions += ['shift']
                        c.shift()

                elif arcs_given.contains(c.buffer[0], c.stack[-1]):
                    if not arcs_given.child_still_has_children(c.stack[-1]):
                        label = arcs_given.get_label(c.buffer[0], c.stack[-1])
                        transitions += [('left', label)]
                        arcs_given.contents.remove((c.buffer[0], c.stack[-1], label))
                        c.left_arc(label)
                    elif len(c.buffer) > 0:
                        transitions += ['shift']
                        c.shift()
                elif len(c.buffer) > 0:
                    transitions += ['shift']
                    c.shift()
            elif len(c.buffer) > 0:
                transitions += ['shift']
                c.shift()
            else:
                # final left-reduction
                label = arcs_given.get_label(c.stack[-2], c.stack[-1])
                transitions += [('right', label)]
                c.right_arc(label)

        if sanity_check:
            arcs_given_original.sort()
            c.arcs.contents.sort()
            assert c.arcs.contents == arcs_given_original, 'Oracle arcs do not match given arcs.'

        return(features, transitions)

    def print_first_n_entries(self, n):
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(self.sentences[:n])

    def shuffle(self):
        shuffle(self.sentences)