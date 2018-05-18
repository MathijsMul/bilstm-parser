import pprint

class Stack:
    def __init__(self):
        self.contents = []

    def __getitem__(self, item):
        if abs(item) > len(self):
            return(None)
        else:
            return self.contents[item]

    def __len__(self):
        return len(self.contents)

    def reduce(self):
        self.contents.pop()

    def add(self, word):
        self.contents.append(word)

class Buffer:
    def __init__(self, sentence):
        self.contents = [word for word in sentence]

    def __getitem__(self, item):
        if item >= len(self):
            return(None)
        else:
            return self.contents[item]

    def __len__(self):
        return len(self.contents)

    def shift(self):
        self.contents.pop(0)

class Arcs:
    def __init__(self):
        self.contents = []

    def __repr__(self):
        return(str(self.contents))

    def add(self, arc):
        """
        add arc
        :param arc: (head, modifier, label)
        :return:
        """
        self.contents.append(arc)

    def load(self, arcs_given):
        self.contents = arcs_given

    def unlabeled_arcs(self):
        return(list(map(lambda triple: (triple[0], triple[1]), self.contents)))

    def contains(self, head, dependent):
        # check if (head, dependent, label) in arcs for any label
        unlabeled_arcs = self.unlabeled_arcs()
        return((head, dependent) in unlabeled_arcs)

    def child_still_has_children(self, child):
        # check if word has no dependents of its own before being reduced
        unlabeled_arcs = self.unlabeled_arcs()
        (parents, children) = zip(*unlabeled_arcs)
        return(child in parents)

    def get_label(self, head, dependent):
        index_arc = self.unlabeled_arcs().index((head, dependent))
        label = self.contents[index_arc][2]
        return(label)

class Configuration:
    def __init__(self, sentence):
        self.stack = Stack()
        self.buffer = Buffer(sentence)
        self.arcs = Arcs()
        self.contents = {'stack': self.stack.contents,
                         'buffer': self.buffer.contents,
                         'arcs': self.arcs.contents}

    def pretty_print(self):
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(self.contents)

    def shift(self):
        """
        top word of buffer shifted to stack

        :return:
        """
        self.stack.add(self.buffer[0])
        self.buffer.shift()

    def left_arc(self, label):
        """
        top word of stack depends on top word of buffer

        :return:
        """
        self.arcs.add((self.buffer[0], self.stack[-1], label))
        self.stack.reduce()

    def right_arc(self, label):
        """
        top word of stack depends on second word of stack

        :return:
        """
        self.arcs.add((self.stack[-2], self.stack[-1], label))
        self.stack.reduce()

    def apply_transition(self, transition):
        if transition == 'shift':
            self.shift()
        elif transition[0] == 'left':
            self.left_arc(transition[1])
        elif transition[0] == 'right':
            self.right_arc(transition[1])

    def extract_features(self, feature_dict):
        buffer_features = [self.buffer[idx] for idx in feature_dict['buffer']]
        stack_features = [self.stack[idx] for idx in feature_dict['stack']]
        return(stack_features + buffer_features)

    def transition_admissible(self, transition):
        if transition == 'shift':
            return(len(self.buffer) > 0)
        elif transition[0] == 'left':
            return(len(self.buffer) > 0 and len(self.stack) > 0 and self.stack[-1] != '0')
        elif transition[0] == 'right':
            return(len(self.stack) > 1 and self.stack[-1] != '0')

    def is_empty(self):
        return(len(self.stack) == 1 and len(self.buffer) == 0)