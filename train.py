from data_loader import ConllLoader
from model import BiLSTMParser
from test import test
from utils import transition_code
import pprint
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch
import time
from random import random

class ModelTrainer:
    def __init__(self, model, datamanager_train_file, datamanager_test_file, epochs, criterion, optimizer, l2_penalty, show_loss=True):
        self.model = model
        self.datamanager_train_file = datamanager_train_file
        self.datamanager_test_file = datamanager_test_file
        self.epochs = epochs

        #TODO: implement hinge loss
        if criterion == 'CrossEntropy':
            self.criterion = nn.CrossEntropyLoss()
        elif criterion == 'Hinge':
            self.criterion = nn.MultiMarginLoss()

        if optimizer == 'Adam':
            self.optimizer = optim.Adam(model.parameters(), weight_decay = l2_penalty)
        elif optimizer == 'SGD':
            self.optimizer = optim.SGD(model.parameters(), lr=0.1)

        self.show_loss = show_loss
        self.training_time = 0

    def train(self, test_each_epoch):
        start_time_training = time.time()

        for epoch in range(self.epochs):
            print('Training epoch %i / %i' % (epoch + 1, self.epochs))
            running_loss = 0.0
            self.datamanager_train_file.shuffle()

            for idx, sentence in enumerate(self.datamanager_train_file.sentences):
                #print('Training sentence %i / %i' % (idx + 1, self.datamanager_train_file.num_samples))

                words = sentence['words']
                pos_tags = sentence['pos_tags']

                for word_idx, word in enumerate(words):
                    if random() < self.datamanager_train_file.word_dropout_probabilities[word]:
                        words[word_idx] = 'UNK-WORD'
                        #pos_tags[idx] = 'UNK-POS'

                features = sentence['oracle']['features']
                gold_transitions = [transition_code(transition, self.model.arc_label_to_idx) for transition in sentence['oracle']['transitions']]
                targets = Variable(torch.LongTensor([transition for transition in gold_transitions]))

                self.optimizer.zero_grad()
                outputs = self.model(words, pos_tags, features)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                # show loss statistics
                if self.show_loss:
                    running_loss += loss.item()
                    if (idx + 1) % 10 == 0:  # print every 10 sentences
                        print('[%d, %5d] loss: %.3f' %
                              (epoch + 1, idx, running_loss / 10))
                        running_loss = 0.0

            if test_each_epoch:
                # evaluate on train set
                output_conll_train = self.datamanager_train_file.file.split('/')[-1].split('.')[-2] + 'epoch%i.conll' % (epoch + 1)
                train_results = test(self.model, self.datamanager_train_file, output_conll_train)

                # print('Training results:')
                print(train_results)

                if not self.datamanager_test_file is None:
                    # evaluate on test set
                    output_conll_test = self.datamanager_test_file.file.split('/')[-1].split('.')[-2] + 'epoch%i.conll' % (epoch + 1)
                    test_results = test(self.model, self.datamanager_test_file, output_conll_test)
                    # print('Testing results:')
                    print(test_results)

        # save model
        torch.save(self.model.state_dict(), 'models/' + self.model.name + '.pt')

        total_training_time = time.time() - start_time_training
        self.training_time += total_training_time