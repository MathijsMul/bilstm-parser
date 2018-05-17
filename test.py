from data_loader import ConllLoader
from model import BiLSTMParser
import subprocess

def test(model, datamanager_test_file, output_conll, batch_size=1):
    """
    always take batch size 1, because consecutive predictions depend on each other so they cannot be parallelized

    datamanager_test_file : ConllLoader object
    """

    test_file = datamanager_test_file.file
    output_conll = 'conlls/' + output_conll
    output_conll_file = open(output_conll, 'w')

    for idx, sentence in enumerate(datamanager_test_file.sentences_unshuffled):
        #if idx % 99 == 0:
        #    print('Testing batch (sentence) %i / %i' % (idx + 1, num_test_batches))
        words = sentence['words']
        pos_tags = sentence['pos_tags']
        _, conll_fragment = model(words, pos_tags, output_to_conll=True)
        output_conll_file.write(conll_fragment)
        if idx != datamanager_test_file.num_samples - 1:
            output_conll_file.write('\n')

    output_conll_file.close()
    eval_results = eval(test_file, output_conll)
    return(eval_results)

def eval(conll_gold, conll_predicted):
    eval_output = subprocess.check_output(['perl',
                                 'eval.pl',
                                 '-q',
                                 '-g',
                                 conll_gold,
                                 '-s',
                                 conll_predicted]).decode("utf-8")
    s = eval_output.split('\n')
    labeled_attachment_score = float(s[0].split()[-2])
    unlabeled_attachment_score = float(s[1].split()[-2])
    label_accuracy_score = float(s[2].split()[-2])
    return(labeled_attachment_score, unlabeled_attachment_score, label_accuracy_score)