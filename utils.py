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

def transition_code(transition, arc_label_to_idx):
    if transition == 'shift':
        return 0
    elif transition[0] == 'left':
        return(arc_label_to_idx[transition[1]])
    elif transition[0] == 'right':
        return (arc_label_to_idx[transition[1]] + len(arc_label_to_idx))

def transition_from_code(code, arc_idx_to_label):
    if code == 0:
        return('shift')
    else:
        if code in arc_idx_to_label:
            return(('left', arc_idx_to_label[code]))
        else:
            return (('right', arc_idx_to_label[code - len(arc_idx_to_label)]))

def hinge_loss(outputs, target):
    raise(NotImplementedError)

def crop_file(file_in, nr_sentences):
    file_out = file_in.split('.')[-2] + str(nr_sentences) + '.conll'
    with open(file_in, 'r') as f_in:
        with open(file_out, 'w') as f_out:
            sentence_count = 0
            for idx, line in enumerate(f_in):
                if line == '\n':
                    sentence_count += 1
                    if sentence_count == nr_sentences:
                        break
                    else:
                        f_out.write(line)
                else:
                    f_out.write(line)