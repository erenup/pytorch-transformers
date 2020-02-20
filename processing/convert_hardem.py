#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""A script to convert the default SQuAD dataset to the format:

'{"question": "q1", "answer": ["a11", ..., "a1i"]}'
...
'{"question": "qN", "answer": ["aN1", ..., "aNi"]}'

"""

import argparse
import json
from tqdm import tqdm
import numpy as np
import os
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str)
parser.add_argument('--output', type=str, help='')
parser.add_argument('--tfidf_top_n', default=0, type=int, help='top n paras in tifidf retrivel data')
parser.add_argument('--cls_top_n', default=0, type=int, help='top n paras by cls scores')

parser.add_argument('--convert_option', default='qa_to_txt', help='covert_option: qa_to_txt or txt_to_qa, combine_cls_input_output')
parser.add_argument('--eval_result', default='', help='it is only needed in convert_option, the output of cls model json file')

args = parser.parse_args()

def read_para_input(input_file):
    dataset = {}
    with open(input_file, 'r', encoding='utf-8') as fin:
        for line in tqdm(fin, desc='read input'):
            line = json.loads(line.strip('\n'))
            if line['squad_id'] not in dataset:
                dataset[line['squad_id']] = {'question': line['question'],
                                       'contexts': [line['document']],
                                       'gold_context': line['squad_context'],
                                       'answers': line['squad_answers'],
                                       'id': line['squad_id'],
                                       'document_tfidf_scores': [line.get('tfidf_score', None)],
                                       'document_cls_scores': [line.get('cls_score', None)],
                                        'labels': [line.get('label', None)],
                                       }
            else:
                dataset[line['squad_id']]['contexts'].append(line['document'])
                dataset[line['squad_id']]['document_tfidf_scores'].append((line.get('tfidf_score', None)))
                dataset[line['squad_id']]['document_cls_scores'].append((line.get('cls_score', None)))
                dataset[line['squad_id']]['labels'].append(line.get('label', None))
    print('data set num with squad id as key is {}'.format(len(dataset)))
    return dataset



if args.convert_option == 'qa_to_txt':

    # Read dataset
    with open(args.input) as f:
        dataset = json.load(f)

    # Iterate and write question-answer pairs
    qa_pair_cnt = 0
    qa_pair_cnt_original = 0
    new_dataset = {'data': [], 'version': 'remove long answers'}
    with open(args.output, 'w', encoding='utf-8') as f:
        for article in tqdm(dataset['data'], total=len(dataset['data'])):
            for paragraph in article['paragraphs']:
                if paragraph.get('gold_context', None):
                    squad_context = paragraph['gold_context']
                else:
                    squad_context = paragraph['context']
                for qa in paragraph['qas']:
                    question = qa['question']
                    qa_pair_cnt_original += 1
                    is_impossible = qa.get("is_impossible", None)
                    if is_impossible:
                        continue
                    if qa['answers']:
                        text = qa['answers'][0]['text']
                        # if len(text.split()) > 5:
                        #     continue

                    qa_pair_cnt += 1
                    answer = [a['text'] for a in qa['answers']]
                    squad_answers = qa['answers']
                    squad_qid = qa['id']
                    f.write(json.dumps({'question': question, 'answer': answer, 'squad_answers': squad_answers,
                                        'squad_context': squad_context, 'squad_qid': squad_qid}))
                    f.write('\n')
                    qas = [qa]
                    paragraph = {'context': squad_context, 'qas': qas}
                    article = {'paragraphs': [paragraph]}
                    new_dataset['data'].append(article)
    with open(args.output + '.without_long_answer_qa', 'w', encoding='utf-8') as f:
        json.dump(new_dataset, f)

    print('convert {} qa pair in json to {} pairs in txt'.format(qa_pair_cnt_original, qa_pair_cnt))

elif args.convert_option == 'txt_to_qa':
    dataset = read_para_input(args.input)
    has_answer = 0
    has_answer_tfidf = 0
    has_answer_not_tfidf = 0
    has_answer_not_cls = 0
    total_cnt = 0
    squad_qa = {'data': [], 'version': 'converted from txt'}
    top_n = 5
    for squad_id, squad_question_content in tqdm(dataset.items(), desc='convert to qa format', total=len(dataset.keys())):
        total_cnt += 1
        article = {}
        top_n = args.tfidf_top_n or args.cls_top_n
        if args.tfidf_top_n:
            sorted_indices = np.argsort(squad_question_content['document_tfidf_scores'])[::-1].tolist()
        elif args.cls_top_n:
            squad_question_content_positive_scores = [score[1] for score in squad_question_content['document_cls_scores']]
            sorted_indices = np.argsort(squad_question_content_positive_scores)[::-1].tolist()
            # contexts = select_paras(squad_question_content['contexts'], squad_question_content['document_cls_scores'],
            #                         args.cls_top_n, options='cls')
        assert len(squad_question_content['labels']) == len(squad_question_content['document_cls_scores'])
        topn_indices = sorted_indices[:top_n]
        topn_positives = [index for index in topn_indices if squad_question_content['labels'][index] == '1']
        topn_tfidf_positives = [index for index in list(range(len(squad_question_content['labels'])))[:top_n]
                                    if squad_question_content['labels'][index] == '1']

        if not topn_tfidf_positives and topn_positives:
            has_answer_not_tfidf += 1
        if not topn_positives and topn_tfidf_positives:
            has_answer_not_cls += 1
        if topn_tfidf_positives:
            has_answer_tfidf += 1
        if topn_positives:
            has_answer += 1
        contexts = [squad_question_content['contexts'][index] for index in topn_indices]
        paragraph = {'context': ' '.join(contexts)}
        qa = {'question': squad_question_content['question'],
              'answers': squad_question_content['answers'],
              'id': squad_question_content['id']}
        paragraph['gold_context'] = squad_question_content['gold_context']
        paragraph['qas'] = [qa]
        paragraphs = [paragraph]
        article['paragraphs'] = paragraphs
        squad_qa['data'].append(article)
    print('topn {} tfidf recall {}'.format(top_n, has_answer_tfidf * 1.0 /total_cnt))
    print('top {} recall: {}'.format(top_n, has_answer * 1.0 / total_cnt))
    print('top {} recall byond tfidf: {}'.format(top_n, has_answer_not_tfidf * 1.0 / total_cnt))
    print('top tfidf {} recall byond cls: {}'.format(top_n, has_answer_not_cls * 1.0 / total_cnt))
    squad_qa['cls_recall'] = has_answer * 1.0 / total_cnt
    squad_qa['tfidf_recall'] = has_answer_not_tfidf * 1.0 / total_cnt
    squad_qa['tfidf_not_in_cls_recall'] = has_answer_not_cls * 1.0 / total_cnt
    squad_qa['cls_not_in_tfidf_recall'] = has_answer_not_tfidf * 1.0 / total_cnt
    input_dir = os.path.dirname(os.path.abspath(args.input))
    output_file = "{}/qa.top_{}".format(input_dir, top_n)
    print('write squad data to {}'.format(output_file))
    with open(output_file, 'w', encoding='utf-8') as fout:
        json.dump(squad_qa, fout)

elif args.convert_option == 'combine_cls_input_output':
    input = args.input
    output = args.input
    examples = []
    with open(input, 'r', encoding='utf-8') as fin:
        for line in tqdm(fin):
            examples.append(json.loads(line.strip('\n')))
    preds = []
    labels = []
    logits = []
    eval_result = {}
    with open(args.eval_result, 'r', encoding='utf-8') as fin:
        eval_result = json.load(fin)
    preds = eval_result['preds']
    logits = eval_result['logits']
    labels = eval_result['labels']
    print('preds: ', len(preds), 'examples:', len(examples))
    assert len(preds) == len(examples)
    assert len(labels) == len(examples)
    for i, example in tqdm(enumerate(examples)):
        assert example['label'] == str(labels[i])
        examples[i]['cls_score'] = logits[i]

    with open(input, 'w', encoding='utf-8') as fout:
        for line in examples:
            fout.write(json.dumps(line))
            fout.write('\n')





else:
    raise ValueError('wrong convert options')

