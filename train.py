from utils import transition_code, test
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import time
import os
import warnings
from random import random

class ModelTrainer:
    """
    Class for model training.

    Attributes:
        model: BiLSTMParser object to train
        datamanager_train_file: ConllLoader object for train data
        datamanager_test_file: ConllLoader object for test data
        epochs: number of epochs to train
        run: training run
        criterion: loss function to use
        optimizer: optimization algorithm
        show_loss: print loss statistics during training
        training_time: time elapsed during training
        model_path: path to saved model
        output_path_train: path to output CONLL files for train data
        output_path_test: path to output CONLL files for test data
    """

    def __init__(self, model, datamanager_train_file, datamanager_test_file, epochs, criterion, optimizer, l2_penalty, run, show_loss=True):
        self.model = model
        self.datamanager_train_file = datamanager_train_file
        self.datamanager_test_file = datamanager_test_file
        self.epochs = epochs
        self.run = str(run)

        if criterion == 'CrossEntropy':
            self.criterion = nn.CrossEntropyLoss()

        if optimizer == 'Adam':
            self.optimizer = optim.Adam(model.parameters(), weight_decay = l2_penalty)
        elif optimizer == 'SGD':
            self.optimizer = optim.SGD(model.parameters(), lr=0.1)

        self.show_loss = show_loss
        self.training_time = 0

        self.initialize_output_dir() # initialize results directory

    def initialize_output_dir(self):
        """
        Initialize output directories.

        :return:
        """
        results_path = 'results/' + self.model.name + '_run' + self.run
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        else:
            warnings.warn('Output directory already exists.')

        self.model_path = results_path + '/models/'
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        output_path = results_path + '/output_conlls/'
        self.output_path_train = output_path + 'train/'
        if not os.path.exists(self.output_path_train):
            os.makedirs(self.output_path_train)
            #TODO: provide appropriate summary of file name
            self.output_path_train += self.datamanager_train_file.file.split('/')[-1].split('.')[-2]

        if not self.datamanager_test_file is None:
            self.output_path_test = output_path + 'test/'
            if not os.path.exists(self.output_path_test):
                os.makedirs(self.output_path_test)
                # TODO: provide appropriate summary of file name
                self.output_path_test += self.datamanager_test_file.file.split('/')[-1].split('.')[-2]

    def train(self, test_each_epoch):
        start_time_training = time.time()

        for epoch in range(self.epochs):
            print('Training epoch %i / %i' % (epoch + 1, self.epochs))
            running_loss = 0.0
            self.datamanager_train_file.shuffle()

            for idx, sentence in enumerate(self.datamanager_train_file.sentences):
                words = sentence['words']
                pos_tags = sentence['pos_tags']

                for word_idx, word in enumerate(words):
                    if random() < self.datamanager_train_file.word_dropout_probabilities[word]:
                        words[word_idx] = 'UNK-WORD'

                features = sentence['oracle']['features']
                gold_transitions = [transition_code(transition, self.model.arc_label_to_idx) for transition in sentence['oracle']['transitions']]
                targets = Variable(torch.LongTensor([transition for transition in gold_transitions]))

                self.optimizer.zero_grad()
                outputs = self.model(words, pos_tags, features)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                # Show loss statistics
                if self.show_loss:
                    running_loss += loss.item()
                    if (idx + 1) % 100 == 0:  # print every 10 sentences
                        print('[%d, %5d] loss: %.3f' %
                              (epoch + 1, idx + 1, running_loss / 100))
                        running_loss = 0.0

            if test_each_epoch:
                # Evaluate on train set
                output_conll_train = self.output_path_train + 'epoch%i.conll' % (epoch + 1)
                train_results = test(self.model, self.datamanager_train_file, output_conll_train)

                # print('Training results:')
                print(train_results)

                if not self.datamanager_test_file is None:
                    # evaluate on test set
                    output_conll_test = self.output_path_test + 'epoch%i.conll' % (epoch + 1)
                    test_results = test(self.model, self.datamanager_test_file, output_conll_test)

                    # print('Testing results:')
                    print(test_results)

        # Save model
        torch.save(self.model.state_dict(), self.model_path + 'model.pt')

        total_training_time = time.time() - start_time_training
        self.training_time += total_training_time