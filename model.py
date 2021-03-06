from arc_hybrid import Configuration
from utils import transition_from_code
import torch
import torch.nn as nn
from torch.autograd import Variable

class BiLSTMParser(nn.Module):
    """
    Bidirectional LSTM parser inspired by Kiperwasser & Goldberg (2016): https://aclweb.org/anthology/Q16-1023.

    Attributes:
        name: name of the model
        word_dim: word embedding dimensionality
        pos_dim: POS tag embedding dimensionality
        input_size: dimension of LSTM input
        num_layers_lstm: nr of LSTM layers
        hidden_units_lstm: dimensionality of hidden LSTM layer
        hidden_units_mlp: dimensionality of hidden MLP (classification) layer
        arc_labels: list of arc labels
        arc_label_to_idx: dictionary from arc labels to indices
        arc_idx_to_label: inverse of the above dictionary
        num_transitions: number of different transitions (classes)
        vocab: vocabulary
        pos_tags: POS tags
        word_dict: dictionary from vocabulary words to indices
        pos_dict: dictionary from POS tags to indices
        voc_size: vocabulary size
        pos_size: number of POS tags
        word_emb: word embedding matrix
        pos_emb: POS tag embedding matrix
        bilstm: bidirectional LSTM cell
        mlp_in: MLP input layer
        tanh: Tanh nonlinearity for MLP
        mlp_out: MLP output layer
        bilstm_representations_batch: stored bidirectional LSTM representations for input batch (sentence)
        features: locations of feature items from arc hybrid configuration (stack & buffer)
    """

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

        self.vocab = vocab
        self.pos_tags = pos_tags
        self.word_dict = {word: i for i, word in enumerate(self.vocab)} # dictionary of vocabulary words
        self.pos_dict = {pos_tag: i for i, pos_tag in enumerate(self.pos_tags)}  # dictionary of POS tags
        self.voc_size = len(self.vocab)  # vocabulary size
        self.pos_size = len(self.pos_tags)  # number of POS tags
        self.word_emb = nn.Embedding(self.voc_size, self.word_dim) # word embedding matrix
        self.pos_emb = nn.Embedding(self.pos_size, self.pos_dim) # POS embedding matrix

        self.bilstm = nn.LSTM(input_size=self.input_size,
                              hidden_size=self.hidden_units_lstm,
                              num_layers=self.num_layers_lstm,
                              bidirectional=True) # bidirectional LSTM unit

        self.mlp_in = nn.Linear(in_features = 2 * 4 * self.hidden_units_lstm, out_features = self.hidden_units_mlp) # MLP to-hidden matrix, assuming 4 bidirectional features
        self.tanh = nn.Tanh() # MLP nonlinearity
        self.mlp_out = nn.Linear(in_features = self.hidden_units_mlp, out_features = self.num_transitions) # MLP to-output matrix

        self.bilstm_representations_batch = []

        if features == 'default':
            # def features:  top 3 items on the stack and the first item on the buffer
            self.features = {'stack' : [-3, -2, -1],
                             'buffer' : [0]}

    def forward(self, words, pos_tags, features=None, output_to_conll=False):

        # Precompute intermediate bidirectional representations of sentence words
        length_sentence = len(words)
        words = [word if word in self.vocab else 'UNK-WORD' for word in words]

        bilstm_representations = self.bilstm_representations(words, pos_tags)
        self.bilstm_representations_batch = bilstm_representations

        if output_to_conll:
            conll_output = ''

        if features is None:
            # Do configutation - transition - configuration one by one (sequentially)
            outputs = []
            c = Configuration([str(i) for i in range(length_sentence)])

            while not c.is_empty():
                configuration = c.extract_features(self.features)
                configuration_tensor = torch.cat([self.get_bilstm_representation(word_idx) for word_idx in configuration], 1)
                mlp_output = self.classification_layers(configuration_tensor)

                top_indices = torch.topk(mlp_output, self.num_transitions)[1][0]
                for entry in top_indices.split(1):
                    transition = transition_from_code(entry.item(), self.arc_idx_to_label)
                    if c.transition_admissible(transition):
                        outputs += [entry.item()]
                        c.apply_transition(transition)
                        break

            if output_to_conll:
                conll_fragment = self.arcs_to_conll(c.arcs)
                conll_output += conll_fragment

        else:
            # During training the features (sequences of configurations) are given
            num_configurations = len(features)
            all_input_features = torch.zeros((num_configurations, 2 * 4 * self.hidden_units_lstm)) # initialize container
            for idx_conf, configuration in enumerate(features):
                bilstm_tensors = [self.get_bilstm_representation(word_idx) for word_idx in configuration]
                configuration_tensor = torch.cat(bilstm_tensors, 1)
                all_input_features[idx_conf,:] = configuration_tensor

            outputs = self.classification_layers(all_input_features)

        if output_to_conll:
            return(outputs, conll_output)
        else:
            return(outputs)

    def sentence_inputs(self, words, pos_tags):
        """
        Translate sentence to sequence of LSTM inputs (concatenations of word embeddings and POS tags).

        :param sentence:  [(word, pos_tag), ..., (word, pos_tag)]
        :return: (sentence_length x (word_dim + pos_dim)) tensor containing concatenated word and POS embeddings
        """

        word_idxs = Variable(torch.LongTensor([self.word_dict[word] for word in words]))
        pos_idxs = Variable(torch.LongTensor([self.pos_dict[pos] for pos in pos_tags]))
        word_embeddings = self.word_emb(word_idxs)
        pos_embeddings = self.pos_emb(pos_idxs)
        word_pos_cat = torch.cat((word_embeddings, pos_embeddings), 1).unsqueeze(1)
        return (word_pos_cat)

    def classification_layers(self, lstm_features):
        mlp_hidden = self.mlp_in(lstm_features)
        mlp_hidden_activated = self.tanh(mlp_hidden)
        mlp_output = self.mlp_out(mlp_hidden_activated)
        return(mlp_output)

    def bilstm_representations(self, words, pos_tags):
        sentence_input = self.sentence_inputs(words, pos_tags)
        lstm_output, _ = self.bilstm(sentence_input) # output: (seq_len, batch, hidden_size * num_directions)
        return(lstm_output)

    def get_bilstm_representation(self, word_idx):
        if word_idx is None:
            # Return zero tensor for emtpy feature positions
            return(torch.zeros((1, 2 * self.hidden_units_lstm)))
        else:
            return(self.bilstm_representations_batch[int(word_idx)])

    def arcs_to_conll(self, arcs):
        """
        Translate arcs for a sentence to CONLL fragment.

        :param arcs: Arcs object
        :return: CONLL fragment
        """

        conll_output = ''
        all_labeled_arcs = arcs.contents
        number_words = len(all_labeled_arcs)
        indices = [i + 1 for i in range(number_words)]

        for word_index in indices:
            for arc in all_labeled_arcs:
                if arc[1] == str(word_index):
                    conll_output += str(word_index) + '\t' + 'WORD' + '\t' + '_' + '\t' + 'TAG' + '\t' + 'TAG' + '\t' + '_' + '\t' + str(arc[0]) + '\t' + str(arc[2]) + '\t' + '_' + '\t' + '_' + '\n'
        return(conll_output)
