import argparse
import json
from tqdm import tqdm
import os

def read_json(input_file):
    data = []
    with open(input_file, 'r') as f:
        data = json.load(f)
    return data

def read_line_json(input_file):
    data = []
    with open(input_file, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip('\n')))
    return data

def combine_examples_results(examples, results):
    preds = results['preds']
    labels = results['labels']
    logits = results['logits']
    print('len preds: {}, len labels: {}'.format(len(preds), len(labels)))
    for example, pred, label, logit in zip(examples, preds, labels, logits):
        label = '1' if label == 1 else '0'
        assert label == example['label']
        example['pred'] = pred
        example['logit'] = logit
    return examples

def examples2dict(para_examples):
    examples_dict = {}
    for example in para_examples:
        _id = example['squad_id']
        title = example['id']
        label = example['label']
        if _id not in examples_dict:
            examples_dict[_id] = [example]
        else:
            examples_dict[_id].append(example)
    return examples_dict

def examples2hotpot(examples_dict):
    hotpot_example_list = []
    for _id, examples in examples_dict.items():
        scored_retrieved = []
        question = None
        for example in examples:
            assert _id == example['squad_id']
            if question:
                assert question == example['question']
            question = example['question']
            title = example['id']
            document = example['document']
            logit = example['logit']
            scored_retrieved.append([title, document, logit[1]])
        scored_retrieved = sorted(scored_retrieved, key=lambda x:x[2], reverse=True)
        hotpot_example_list.append({
            '_id': _id,
            'answer': '',
            'question': question,
            'scored_retrieved': scored_retrieved
        })
    return hotpot_example_list


def update_document_recall(prediction, gold):
    cur_sp_pred = set(prediction)

    gold_sp_pred = set(gold)

    tp= 0
    for e in cur_sp_pred:
        if e in gold_sp_pred:
            tp += 1

    if tp > 0:
        return 1
    else:
        return 0


def update_document(metrics, prediction, gold):
    cur_sp_pred = set(prediction)

    gold_sp_pred = set(gold)

    tp, fp, fn = 0, 0, 0
    for e in cur_sp_pred:
        if e in gold_sp_pred:
            tp += 1
        else:
            fp += 1
    for e in gold_sp_pred:
        if e not in cur_sp_pred:
            fn += 1
    prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    em = 1.0 if fp + fn == 0 else 0.0
    metrics['doc_em'] += em
    metrics['doc_f1'] += f1
    metrics['doc_prec'] += prec
    metrics['doc_recall'] += recall
    return em, prec, recall, f1

def eval_p_level(pred_dict, label_dict, top_n=2):
    print('start to evaluate paragrah performance')
    # assert len(pred_dict.keys()) == len(label_dict.keys())
    # pred_dict_set = set(pred_dict.keys())
    # label_dict_set = set(label_dict.keys())
    # assert pred_dict_set == label_dict_set
    metrics = {'doc_em': 0., 'doc_f1': 0., 'doc_prec': 0., 'doc_recall': 0.}
    pred_num = 0
    pr = 0
    for label_id, example in label_dict.items():
        if label_id not in pred_dict:
            print('missing {}'.format(label_id))
            continue
        pred_num += 1
        sp = example['supporting_facts']
        sp_docs = [x[0] for x in sp]
        pred_docs = pred_dict[label_id]['scored_retrieved'][:top_n]
        pred_docs = [x[0] for x in pred_docs]
        em, prec, recall, f1 = update_document(metrics, pred_docs, sp_docs)
        pr += update_document_recall(pred_docs, sp_docs)

    for key, value in metrics.items():
        metrics[key] /= pred_num
    print('recall num: {} PR is {}'.format(pr, pr * 1.0 / len(label_dict)))
    print('pred num is {}, lables num is {}'.format(pred_num, len(label_dict)))
    print('results is {}'.format(metrics))


def write_json(data, file, data_format='squad'):
    print('write {} format data to {}'.format(data_format, file))
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(data, f)

def convert_topn2squad(scored_retrieval, top_n=5):

    print('start to convert top {} of retrieved paras to squad')
    yes_nums = 0
    no_nums = 0
    str_nums = 0
    squad_data = {'data': [], 'version': 'convert from hotpot qa'}
    for _id, example in tqdm(scored_retrieval.items(), total=len(scored_retrieval), desc='convert retrieved hotpot qa to squad format'):
        _id = example['_id']
        answer = example['answer']
        if answer == 'yes':
            yes_nums += 1
        elif answer == 'no':
            no_nums += 1
        else:
            str_nums += 1
        question = example['question']

        sf = example.get('supporting_facts', None)
        answers = []
        if example.get('context', None):
            for index in range(len(example['context'])):
                example['context'][index][1][0] = ' ' + example['context'][index][1][0]
            context = example['context']
            context_dict = {x[0]: x[1] for x in context}
            type = example['type']
            level = example['level']
            is_impossible = True


            sentences = []
            sentence_labels = []
            for one_sf in sf:
                title = one_sf[0]
                sentence_num = one_sf[1]
                para_list = context_dict[title]
                for sentence_index in range(len(para_list)):
                    if sentence_index == sentence_num:
                        sentence_labels.append("1")
                    else:
                        sentence_labels.append("0")
                    sentences.append(para_list[sentence_index])
            context = ''
            sentence_starts = []
            sentence_start = 0
            for sentence in sentences:
                sentence_starts.append(sentence_start)
                context += sentence
                sentence_start += len(sentence)
            sentence_ends = sentence_starts[1:] + [sentence_start]
            sentence_indices = [[x, y] for x,y in zip(sentence_starts, sentence_ends)]
            for sentence_index in range(len(sentences)):
                sent_start, sent_end = sentence_indices[sentence_index]
                assert context[sent_start:sent_end] == sentences[sentence_index]




            if answer in "yes no":
                answer_text = ''
                answer_start = -1
            else:
                answer_text = answer
                answer_start = -1
                is_impossible = False

        scored_retrieved_topn = example['scored_retrieved'][:top_n]

        retrieved_context = [x[1] for x in scored_retrieved_topn]
        retrieved_context = ' '.join(retrieved_context)
        retrieved_context = ' '.join(retrieved_context.split())
        answers.append({'text': answer_text, 'answer_start': answer_start,
                            'yes_no': True if answer in "yes no" else False, 'type': type, 'level': level,
                            'sp': sf,
                            'sentences': sentences, 'sentence_indices': sentence_indices, "sentence_labels": sentence_labels})

                    # print(short_answers)
        qas = [{'question': example['question'], 'id': example['_id'], "answers": answers,
                'is_impossible': is_impossible}]
        paragraph = {'context': retrieved_context, 'right_context': context,
                     'qas': qas,
                     }
        paragraphs = [paragraph]
        squad_data['data'].append({'title': 'title', 'paragraphs': paragraphs})
    print('yes nums is {}, no nums is {}, text answer nums is {}'.format(yes_nums, no_nums, str_nums))
    return squad_data

def combine_examples_with_scores(hotpot_examples, examples_dict):
    hotpot_examples_combined = []
    module = 'ranked'
    for example in hotpot_examples:
        _id = example['_id']
        paras_ranked = examples_dict.get(_id, [])
        example[module] = []
        if not paras_ranked:
            print('id {} misses paras!')
            continue
        for para in paras_ranked:
            context_title = para['context_title']
            title = para['context_title'][0]
            context = para['context_title'][1]
            score = para['logit'][1]
            example[module].append([module, score, title, context])
        hotpot_examples_combined.append(example)
    return hotpot_examples_combined


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_file', default='', help='evaluated file')
    parser.add_argument('--eval_results', default='', help='evaluated results')
    parser.add_argument('--hotpot_file', default='', help='hotpot file before converted to paras')
    parser.add_argument('--top_n', default=2, type=int, help='default top n')
    args = parser.parse_args()
    eval_examples = read_line_json(args.eval_file)
    eval_results = read_json(args.eval_results)
    hotpot_examples = read_json(args.hotpot_file)

    examples = combine_examples_results(eval_examples, eval_results)
    examples_dict = examples2dict(examples)
    hotpot_examples_with_scores = combine_examples_with_scores(hotpot_examples, examples_dict)
    output_file = args.eval_file + ".ranked"
    write_json(hotpot_examples_with_scores, output_file,data_format='hotpot ranked')
    hotpot_dict = {example['_id']: example for example in hotpot_examples}
    retrieved_hotpot = examples2hotpot(examples_dict)
    retrieved_hotpot_dict = {example['_id']: example for example in retrieved_hotpot}
    results = eval_p_level(retrieved_hotpot_dict, hotpot_dict, args.top_n)
    for _id, example in hotpot_dict.items():
        retrieved = retrieved_hotpot_dict.get(_id, None)
        assert retrieved is not None
        hotpot_dict[_id]['scored_retrieved'] = retrieved['scored_retrieved']

    squad_data = convert_topn2squad(hotpot_dict, top_n=args.top_n)
    # squad_file = args.eval_file + ".squad.top_{}".format(args.top_n)
    # write_json(squad_data, squad_file)







