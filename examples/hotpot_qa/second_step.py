import json
import argparse
from tqdm import tqdm
import os

def read_json(input_file):
    data = []
    with open(input_file, 'r') as f:
        data = json.load(f)
    return data

def write_json(data, file):
    print('write json format data to {}'.format(file))
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(data, f)

def write_line_json(examples, output_file):
    print('write {} examples to {}'.format(len(examples), output_file))
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example))
            f.write('\n')

def read_line_json(file):
    print('read line json from {}'.format(file))
    data = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:

            data.append(json.loads(line.strip('\n')))
    return data


def add_para_example(firt_clue, potential_clue, example, mode='train'):
    para_examples = []
    _id = example['_id']
    question = example['_id']
    paratext1 = "".join(firt_clue[3])
    if mode != 'test':
        sf = example['supporting_facts']
    else:
        sf = [["", 0]]
    sf_dict = {x[0]: x for x in sf}
    for x in potential_clue:
        title = x[2]
        para_text = "".join(x[3])
        label = "0"
        if title in sf_dict:
            label = '1'
        para_example = {'id': title, 'squad_id': _id, 'question': question + " " + paratext1, 'document': para_text,
                        'label': label, "title_para": [x[2], x[3]]}
        para_examples.append(para_example)
    return para_examples

def convert2nd(datalist, mode='train'):
    print('start to convert {} examples'.format(len(datalist)))
    para_examples = []
    second_num = 0
    for example in tqdm(datalist):
        confidence_num = 0
        ranked_paras = [x for x in example['ranked'] if x[0] == 'ranked']
        ranked_paras = sorted(ranked_paras, key=lambda x:x[1], reverse=True)
        for x in ranked_paras[:10]:
            if x[1] > 0.0:
                confidence_num += 1
        if confidence_num >= 2:
            continue
        elif confidence_num == 1:
            if len(ranked_paras) <=1:
                print('ranked paras num is too small')
                continue
            second_num += 1
            firt_clue = ranked_paras[0]
            potential_clue = ranked_paras[1:]
            para_examples.extend(add_para_example(firt_clue, potential_clue, example, mode=mode))
    print('second num is {}'.format(second_num))
    return para_examples


def combine_1_2step(ranked_list, cls_input, cls_logits):
    print('combine sf results')
    if len(cls_input) != len(cls_logits['preds']):
        print('sf input {}, sf result {}'.format(len(cls_input), len(cls_logits['preds'])))
    assert len(cls_input) == len(cls_logits['preds'])
    preds = cls_logits['preds']
    labels = cls_logits['labels']
    logits = cls_logits['logits']
    label_dict = {1: '1', 0: '0'}
    pred_2nd_dict = {}
    print('len preds: {}, len labels: {}'.format(len(preds), len(labels)))
    for example, pred, label, logit in zip(cls_input, preds, labels, logits):
        label = label_dict[label]
        assert label == example['label']
        _id = example['squad_id']
        if not pred_2nd_dict.get(_id, []):
            pred_2nd_dict[_id] = [['ranked2nd', logit[1], example['id'], example['title_para'][1]]]
        else:
            pred_2nd_dict[_id].append(['ranked2nd', logit[1], example['id'], example['title_para'][1]])

    ranked_list_new = []
    for example in ranked_list:
        if not pred_2nd_dict.get(example['_id'], None):
            example['ranked2nd'] = []
            ranked_list_new.append(example)
            continue
        ranked2nd = pred_2nd_dict.get(example['_id'])
        ranked2nd = sorted(ranked2nd, key=lambda x:x[1], reverse=True)
        example['ranked2nd'] = ranked2nd[:]
        ranked_list_new.append(example)
    return ranked_list_new


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--option', default='convert', type=str)
    parser.add_argument('--ranked_file', default='', type=str)
    parser.add_argument('--output_dir', default='second_step', type=str)
    parser.add_argument('--mode', default='train', type=str)

    parser.add_argument('--cls_input', default='', type=str)
    parser.add_argument('--cls_result', default='', type=str)
    args = parser.parse_args()
    data = read_json(args.ranked_file)
    dir = os.path.dirname(os.path.abspath(args.ranked_file))
    output_dir = os.path.join(dir, args.output_dir)
    if args.option == 'convert':
        second_examples = convert2nd(data, mode=args.mode)
        if not os.path.exists(output_dir):
            print('mkdir {}'.format(output_dir))
            os.mkdir(output_dir)
        else:
            print('overwrite {}'.format(output_dir))
        output_file = os.path.join(output_dir, 'dev.json.para.2nd')
        write_line_json(second_examples, output_file)
    else:
        logits = read_json(args.cls_result)
        cls_inputs = read_line_json(args.cls_input)
        data_combined = combine_1_2step(data, cls_inputs, logits)
        output_file = os.path.join(output_dir, 'dev.json')
        write_json(data_combined, output_file)

