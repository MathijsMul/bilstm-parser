from arc_hybrid import Configuration
from utils import transition_from_code
import torch
import torch.nn as nn
from torch.autograd import Variable
import pprint

pp = pprint.PrettyPrinter(indent=4)

class BiLSTMParser(nn.Module):
    def __init__(self, name, vocab, pos_tags, word_dim, pos_dim, num_layers_lstm, hidden_units_lstm, hidden_units_mlp, arc_labels, features):
        super(BiLSTMParser, self).__init__()

        self.name = name
        self.word_dim = word_dim # dimensionality of word embeddings
        self.pos_dim = pos_dim # dimensionality of POS tag embeddings
        self.input_size = self.word_dim + self.pos_dim
        self.num_layers_lstm = num_layers_lstm # number of LSTM layers
        self.hidden_units_lstm = hidden_units_lstm # dimensionality of LSTM hidden units
        self.hidden_units_mlp = hidden_units_mlp # dimensionality of MLP hidden units

        self.arc_labels = arc_labels
        self.arc_label_to_idx = {arc_label: idx + 1 for idx, arc_label in enumerate(self.arc_labels)}
        self.arc_idx_to_label = {idx + 1 : arc_label for idx, arc_label in enumerate(self.arc_labels)}
        self.num_transitions = 2 * len(arc_labels) + 1 # number of different shift/reduce actions (the labels)

        self.vocab = vocab + ['UNK-WORD']
        self.pos_tags = pos_tags + ['UNK-POS']
        self.word_dict = {word: i for i, word in enumerate(self.vocab)} # dictionary of vocabulary words
        self.pos_dict = {pos_tag: i for i, pos_tag in enumerate(self.pos_tags)}  # dictionary of POS tags
        self.voc_size = len(self.vocab)  # vocabulary size
        self.pos_size = len(self.pos_tags)  # number of POS tags
        self.word_emb = nn.Embedding(self.voc_size, self.word_dim) # word embedding matrix
        self.pos_emb = nn.Embedding(self.pos_size, self.pos_dim) # POS embedding matrix

        self.bilstm = nn.LSTM(input_size = self.input_size,
                              hidden_size= self.hidden_units_lstm,
                              num_layers= self.num_layers_lstm,
                              bidirectional= True) # bidirectional LSTM unit

        self.mlp_in = nn.Linear(in_features = 2 * 4 * self.hidden_units_lstm, out_features = self.hidden_units_mlp) # MLP to-hidden matrix, assuming 4 bidirectional features
        self.tanh = nn.Tanh() # MLP nonlinearity
        self.mlp_out = nn.Linear(in_features = self.hidden_units_mlp, out_features = self.num_transitions) # MLP to-output matrix

        # not in Goldberg & Kipperwasser:
        # self.softmax = nn.Softmax()

        self.bilstm_representations_batch = []

        if features == 'default':
            # def features:  top 3 items on the stack and the first item on the buffer
            self.features = {'stack' : [-3, -2, -1],
                             'buffer' : [0]}

    def sentence_inputs(self, sentence):
        """

        :param sentence:  [(word, pos_tag), ..., (word, pos_tag)]
        :return: (sentence_length x (word_dim + pos_dim)) tensor containing concatenated word and POS embeddings
        """

        words = [pair[0] for pair in sentence]
        pos_tags = [pair[1] for pair in sentence]
        word_idxs = Variable(torch.LongTensor([self.word_dict[word] for word in words]))
        pos_idxs = Variable(torch.LongTensor([self.pos_dict[pos] for pos in pos_tags]))
        word_embeddings = self.word_emb(word_idxs)
        pos_embeddings = self.pos_emb(pos_idxs)
        word_pos_cat = torch.cat((word_embeddings, pos_embeddings), 1)
        return (word_pos_cat)

    def sort_inputs(self, inputs):
        inputs_ordered = sorted(enumerate(inputs), key=lambda k: len(k[1]), reverse=True)
        order = [item[0] for item in inputs_ordered]
        inputs_ordered = [item[1] for item in inputs_ordered]
        restore_order = [idx for idx, new_idx in sorted(enumerate(order), key=lambda k: k[1])]
        return(inputs_ordered, order, restore_order)

    def batch_features(self, sentences):
        seq_lengths = [len(sequence) for sequence in sentences]  # list of integers holding information about the batch size at each sequence step
        batch_in = torch.nn.utils.rnn.pad_sequence([self.sentence_inputs(s) for s in sentences]) # pad short inputs
        pack = torch.nn.utils.rnn.pack_padded_sequence(batch_in, seq_lengths, batch_first=False)
        return (pack)

    def bilstm_representations(self, sentences_tagged):
        """

        :param sentences: list of sentences
        :return:
        """
        batch_size = len(sentences_tagged)
        sentences_ordered, order, restore_order = self.sort_inputs(sentences_tagged) # order from longest to shortest
        input_batch = self.batch_features(sentences_ordered) # (seq_len, batch, input_size)
        lstm_output, states = self.bilstm(input_batch)
        unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(lstm_output) # unpack
        outputs = [unpacked[:unpacked_len[i],i,:] for i in range(batch_size)] # select LSTM outputs
        outputs_reordered = [outputs[restore_order[i]] for i in range(batch_size)] # reorder LSTM outputs according to original order
        return(outputs_reordered)

    def forward(self, sentences, pos_tags, features=None, output_to_conll=False):
        """

        :param input: (sentences, features)
        :return:
        """

        words_pos_zipped = [list(zip(item[0], item[1])) for item in list(zip(sentences, pos_tags))]

        # precompute intermediate bidirectional representations of sentence words
        bilstm_representations = self.bilstm_representations(words_pos_zipped)
        self.bilstm_representations_batch = bilstm_representations

        if output_to_conll:
            conll_output = ''

        if features is None:
            # do configutation - transition - configuration one by one (sequentially)
            outputs = []

            for idx_sentence, sentence in enumerate(sentences):
                outputs_sentence = []
                c = Configuration([str(i) for i in range(len(sentence))])

                while not c.is_empty():
                    configuration = c.extract_features(self.features)
                    configuration_tensor = torch.cat([self.get_bilstm_representation(idx_sentence, word_idx) for word_idx in configuration], 1)
                    mlp_output = self.classification_layers(configuration_tensor)

                    top_indices = torch.topk(mlp_output, self.num_transitions)[1][0]
                    for entry in top_indices.split(1):
                        transition = transition_from_code(entry.item(), self.arc_idx_to_label)
                        if c.transition_admissible(transition):
                            outputs_sentence += [entry.item()]
                            c.apply_transition(transition)
                            break

                if output_to_conll:
                    conll_fragment = self.arcs_to_conll(c.arcs)
                    conll_output += conll_fragment

                outputs += [outputs_sentence]

        else:
            # during training features (sequences of configurations) are given
            num_inputs = sum([len(feature_list) for feature_list in features])

            all_input_features = torch.zeros((num_inputs, 2 * 4 * self.hidden_units_lstm)) # initialize container
            for idx_sentence, feature_list in enumerate(features):
                for idx_conf, configuration in enumerate(feature_list):
                    configuration_tensor = torch.cat([self.get_bilstm_representation(idx_sentence, word_idx) for word_idx in configuration], 1)
                    all_input_features[idx_conf,:] = configuration_tensor

            outputs = self.classification_layers(all_input_features)

        if output_to_conll:
            return(outputs, conll_output)
        else:
            return(outputs)

    def classification_layers(self, lstm_features):
        mlp_hidden = self.mlp_in(lstm_features)
        mlp_hidden_activated = self.tanh(mlp_hidden)
        mlp_output = self.mlp_out(mlp_hidden_activated)
        #mlp_output_softmax = self.softmax(mlp_output)
        #return (mlp_output_softmax)
        return(mlp_output)

    def get_bilstm_representation(self, sentence_idx, word_idx):
        if word_idx is None:
            # return zero tensor for emtpy feature positions
            return(torch.zeros((1, 2 * self.hidden_units_lstm)))
        else:
            return(self.bilstm_representations_batch[int(sentence_idx)][int(word_idx)].unsqueeze(0))

    def arcs_to_conll(self, arcs):
        conll_output = ''
        all_labeled_arcs = arcs.contents
        number_words = len(all_labeled_arcs)
        indices = [i + 1 for i in range(number_words)]

        for word_index in indices:
            for arc in all_labeled_arcs:
                if arc[1] == str(word_index):
                    conll_output += str(word_index) + '\t' + 'WORD' + '\t' + '_' + '\t' + 'TAG' + '\t' + 'TAG' + '\t' + '_' + '\t' + str(arc[0]) + '\t' + str(arc[2]) + '\t' + '_' + '\t' + '_' + '\n'
        #conll_output += '\n'
        return(conll_output)
