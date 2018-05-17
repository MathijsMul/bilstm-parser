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