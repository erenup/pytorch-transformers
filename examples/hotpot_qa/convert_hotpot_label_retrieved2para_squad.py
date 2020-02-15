import argparse
import json
from tqdm import tqdm
import os
import random
import unicodedata

def read_json(input_file):
    data = []
    with open(input_file, 'r') as f:
        data = json.load(f)
    return data

def normalize_text(text):
    return unicodedata.normalize("NFKD", text)

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

def convert2para(data_list, max_para_num=15, mode='train', converted_keys='context', sub_key='', distant=False, para_score_th=0.5, rank2=False):
    print('max para num is: {} convert keys is: {}, convert sub key is :{}, distant {}, mode is {}'.format(max_para_num,
                                                    converted_keys, sub_key, distant, mode))
    para_examples = []
    negtive = 0
    positive = 0
    converted_keys = converted_keys.split('#')
    metrics = {'doc_em': 0., 'doc_f1': 0., 'doc_prec': 0., 'doc_recall': 0.}
    pred_num = 0
    distant_num = 0
    for example in tqdm(data_list):
        pred_num += 1
        para_id = 0
        _id = example['_id']
        question = example['question']
        if mode != 'test':
            answer = example['answer']
            sf = example['supporting_facts']
            sf_dict = {x[0]:x[1] for x in sf}
        else:
            answer = ""
            sf = [["", 0]]
            sf_dict = {}

        if mode == 'train':
            assert 'context' in converted_keys
        context =[]
        if 'context' in converted_keys:
            context = example['context']

        answers = [{'answer_text': answer, 'answer_start': -1}]
        para_num = 0
        context_merged = [x for x in context if x[0] in sf_dict]
        #only choose positive from context

        retrievals = []
        for converted_key in converted_keys:
            if converted_key == 'context':
                continue
            retrievals.extend(example[converted_key])
        if sub_key and sub_key != "random":
            retrievals = [x for x in retrievals if x[0] == sub_key]
        if sub_key == 'random':
            assert mode == 'train'
            retrievals = random.sample(retrievals, len(retrievals))

        retrievals_dict = {x[2]:x for x in retrievals}
        retrievals = [x for _, x in retrievals_dict.items()]
        retrievals = sorted(retrievals, key=lambda x:x[1], reverse=True)

        # confidence_num = 0
        # context_merged_confidence = []
        # for ret in retrievals:
        #     if ret[1] > para_score_th:
        #         confidence_num += 1
        #         context_merged_confidence.append([ret[2], ret[3]])

        # confidence_num2 = 0
        # context_merged_confidence2 = []
        # if rank2 and example.get('ranked2nd', []):
        #     rank2nd = example['ranked2nd']
        #     for ret in rank2nd:
        #         if ret[1] > 0.0:
        #             confidence_num2 += 1
        #             context_merged_confidence2.append([ret[2], ret[3]])


        context_dict = {x[0]: x for x in context}
        context_merged.extend([[x[2], x[3]] for x in retrievals if x[2] not in context_dict])
        context_merged_bk = context_merged[:]
        if mode == 'train':
            if max_para_num > 4:
                context_merged = context_merged[:max_para_num-1]
                if context_merged_bk[max_para_num-1:]:
                    context_merged.append(random.choice(context_merged_bk[max_para_num-1:]))
            else:
                context_merged = context_merged[:max_para_num]
        else:
            # if "context" not in converted_keys:
            #     if confidence_num >= 2:
            #         context_merged = context_merged_confidence[:]
            #     else:
            #         context_merged = context_merged[:max_para_num]
            #     # context_merged.extend(context_merged_confidence2[:2])
            # else:
            context_merged = context_merged[:max_para_num]

        if distant:
            for title_para in context_merged_bk:
                title = title_para[0]
                para_text = "".join(title_para[1])
                if answer not in "yes no" and normalize_text(answer) in normalize_text(para_text) and title not in sf_dict:
                    context_merged.append(title_para)
        example['selected'] = context_merged
        preds = [x[0] for x in context_merged]
        golds = [x[0] for x in sf]
        em, prec, recall, f1 = update_document(metrics, preds, golds)
        for para in context_merged:
            title = para[0]
            sentences = para[1]
            sentence_starts = []
            sentence_start = 0
            para_text = ''
            for sentence in sentences:
                sentence_starts.append(sentence_start)
                para_text += sentence
                sentence_start += len(sentence)
            sentence_ends = sentence_starts[1:] + [sentence_start]
            sentence_indices = [[x, y] for x, y in zip(sentence_starts, sentence_ends)]
            for sentence_index in range(len(sentences)):
                sent_start, sent_end = sentence_indices[sentence_index]
                assert para_text[sent_start:sent_end] == sentences[sentence_index]

            label = "0"
            if title in sf_dict:
                label = "1"
            if distant and answer not in "yes no" and normalize_text(answer) in normalize_text(para_text):
                distant_num += 1
                label = "1"
            para_id += 1
            para_example = {'id': title, 'squad_id': _id, 'question': question, 'document': para_text,
                            'squad_answers': answers, 'label': label, 'sentence_indices': sentence_indices,
                            'context_title': para[:]}
            if label == '1':
                positive += 1
            else:
                negtive += 1
            para_num += 1
            para_examples.append(para_example)
    if distant:
        print('distant num is {}'.format(distant_num))
    print('negtive num is {}, positive num is {}'.format(negtive, positive))
    for key, value in metrics.items():
        metrics[key] /= pred_num
    print('pred num is {}, lables num is {}'.format(pred_num, len(data_list)))
    print('results is {}'.format(metrics))
    return para_examples

def write_line_json(examples, output_file):
    print('write {} examples to {}'.format(len(examples), output_file))
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example))
            f.write('\n')



def write_json(data, file):
    print('write squad format data to {}'.format(file))
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(data, f)

def constructed_sentences(context, sf_dict, answer_text, squad_para_num=4):
    # print('max {} paras converted to squad'.format(squad_para_num))
    context = context[:squad_para_num+1]
    total_context = []
    gold_num = 1
    random_num = 0
    drop_ans_num = 0
    distant_num = 0
    context_random = random.sample(context, len(context))
    drop_one_sf_context = []
    drop_num = 0
    for title_para in context_random:
        title = title_para[0]
        if drop_num == 0 and title in sf_dict:
            drop_num += 1
            continue
        drop_one_sf_context.append(title_para)
    drop_ans_num += 1
    total_context.append(drop_one_sf_context)
    total_context.append(context[:squad_para_num])
    total_context.append(context_random[:squad_para_num])
    random_num += 1
    # deleted_answer_context = []
    # for title_para in context_random:
    #     title = title_para[0]
    #     para_text = "".join(title_para[1])
    #     if drop_num == 0 and answer_text in para_text:
    #         drop_num += 1
    #         continue
    #     deleted_answer_context.append(title_para)
    # total_context.append(deleted_answer_context)

    if answer_text not in "yes no":
        distant_context = []
        distant = False
        for title_para in context_random:
            title = title_para[0]
            para_text = "".join(title_para[1])
            if answer_text in para_text and title not in sf_dict:
                distant = True
                distant_context.append(title_para)
            if answer_text not in para_text and title in sf_dict:
                distant_context.append(title_para)
            if title not in sf_dict:
                distant_context.append(title_para)
        if distant:
            distant_num += 1
            total_context.append(distant_context[:])
    examples_sentences = []
    for context_one in total_context:
        context_dict = {x[0]: x[1] for x in context_one}
        sentence_labels = []
        sentences = []
        sentence_titles = []
        too_long = False
        for title, para_list in context_dict.items():
            sentence_num = sf_dict.get(title, -1)

            for sentence_index in range(len(para_list)):
                if len(para_list[sentence_index].split()) > 300:
                    print('sentence in title {} is too long, length is {}, tokens {}'.format(title,
                                                                                             len(para_list[
                                                                                                     sentence_index]),
                                                                                             len(para_list[
                                                                                                     sentence_index].split())))
                    para_list[sentence_index] = " ".join(para_list[sentence_index].split()[:100])

                if sentence_index == sentence_num:
                    sentence_labels.append("1")
                else:
                    sentence_labels.append("0")
                sentences.append(para_list[sentence_index])
                sentence_titles.append(title)
        if not too_long:
            examples_sentences.append([sentence_titles, sentences, sentence_labels])
    return {'examples_sentences': examples_sentences, 'drop_ans_num': 1, 'distant_num': distant_num,
            'drop_ans_num': drop_ans_num, 'gold_num': gold_num, 'rand_num': random_num}

def constructed_squad(examples_sentences, squad_data, example, mode='train'):
    if mode != 'test':
        answer = example["answer"]
        is_impossible = True
        type = example['type']
        level = example['level']
        sf = example["supporting_facts"]
    else:
        answer = ''
        is_impossible = False,
        type = ''
        level = ''
        sf = [["", 0]]
    cnt_not = 0
    for x in examples_sentences:
        answers = []
        sentence_starts = []
        sentence_start = 0
        context = ''
        sentence_labels = x[2]
        sentence_titles = x[0]
        sentences = x[1]
        for sentence in sentences:
            sentence_starts.append(sentence_start)
            context += sentence
            sentence_start += len(sentence)
        sentence_ends = sentence_starts[1:] + [sentence_start]
        sentence_indices = [[x, y] for x, y in zip(sentence_starts, sentence_ends)]
        for sentence_index in range(len(sentences)):
            sent_start, sent_end = sentence_indices[sentence_index]
            assert context[sent_start:sent_end] == sentences[sentence_index]
        context = normalize_text(context)
        answer = normalize_text(answer)
        if answer in "yes no":
            answer_text = ''
            answer_start = -1
            yes_no_answer = answer
        else:
            answer_text = normalize_text(answer)
            yes_no_answer = 'text'
            answer_start = normalize_text(context).find(normalize_text(answer))
            if mode == 'train':
                if answer_start != -1:
                    is_impossible = False
                else:
                    print('example cannot find answer! in training data')
                    cnt_not += 1
                    answer_text = ''
            if mode == 'dev':
                is_impossible = False
                if answer_start != -1:
                    is_impossible = False
                else:
                    cnt_not += 1
                    # print('example cannot find answer! in dev data')

        answers.append({'text': answer_text, 'answer_start': answer_start,
                        'yes_no_answer': yes_no_answer.lower(), 'type': type, 'level': level,
                        'sp': sf,
                        'sentences': sentences, 'sentence_indices': sentence_indices,
                        "sentence_labels": sentence_labels, 'sentence_titles': sentence_titles})

        # print(short_answers)
        if mode == 'test':
            qas = [{'question': example['question'], 'id': example['_id'], "answers": answers}]
        else:
            qas = [{'question': example['question'], 'id': example['_id'], "answers": answers,
                'is_impossible': is_impossible}]
        paragraph = {'context': context,
                     'qas': qas,
                     'sentences': sentences,
                     'sentence_indices': sentence_indices,
                     'sentence_titles': sentence_titles,
                     }

        paragraphs = [paragraph]
        squad_data['data'].append({'title': 'title', 'paragraphs': paragraphs})
    return cnt_not
def convert2squad(data, option='gold', mode='train', max_para_num=4):
    yes_nums = 0
    no_nums = 0
    str_nums = 0
    squad_data = {'data': [], 'version': 'convert from hotpot qa'}
    gold_num = 0
    distant_num = 0
    rand_num = 0
    drop_ans_num = 0
    cnt_not = 0
    for example in tqdm(data, total=len(data), desc='convert hotpot qa to squad format'):

        _id = example['_id']
        if mode != 'test':
            answer = example['answer']
            if answer == 'yes':
                yes_nums += 1
            elif answer == 'no':
                no_nums += 1
            else:
                str_nums += 1
            sf = example['supporting_facts']
            sf_dict = {x[0]: x[1] for x in sf}
        else:
            sf = [["", 0]]
            sf_dict = {}
            answer = ""
        sentence_labels = []
        sentence_titles = []
        examples_sentences = []
        context = ''
        sentences = []
        if option == 'selected':
            for index in range(len(example['selected'])):
                example['selected'][index][-1][-1] = example['selected'][index][-1][-1] + ' '
            context = example['selected'][:max_para_num]
            context_dict = {x[0]: x[1] for x in context}
            for title, para_list in context_dict.items():
                sentence_num = sf_dict.get(title, -1)
                for sentence_index in range(len(para_list)):
                    if len(para_list[sentence_index].split()) > 300:
                        print('sentence in title {} is too long, length is {}, tokens {}'.format(title,
                                                                                                 len(para_list[sentence_index]),
                                                                                                 len(para_list[sentence_index].split())))
                        para_list[sentence_index] = " ".join(para_list[sentence_index].split()[:100])

                    if sentence_index == sentence_num:
                        sentence_labels.append("1")
                    else:
                        sentence_labels.append("0")
                    sentences.append(para_list[sentence_index])
                    sentence_titles.append(title)

            examples_sentences = [[sentence_titles, sentences, sentence_labels]]

        elif option == 'distant':
            for index in range(len(example['selected'])):
                example['selected'][index][-1][-1] = example['selected'][index][-1][-1] + ' '
            context = example['selected']
            returned_dct = constructed_sentences(context, sf_dict, answer)
            examples_sentences = returned_dct['examples_sentences']

            distant_num += returned_dct['distant_num']
            rand_num += returned_dct['rand_num']
            drop_ans_num += returned_dct['drop_ans_num']
            gold_num += returned_dct['gold_num']

        else:
            raise ValueError('wrong option of converting squad!')

        cnt_not += constructed_squad(examples_sentences, squad_data, example, mode=mode)
    print('{} cannot find answers'.format(cnt_not))
    print('yes nums is {}, no nums is {}, text answer nums is {}'.format(yes_nums, no_nums, str_nums))
    print('distant_num {}, rand_num {}, drop_ans_num {} gold num {}'.format(distant_num, rand_num, drop_ans_num, gold_num))
    return squad_data


def convert2tri(data, max_negtive_num, squad_key='gold', is_test=False):
    tri_examples = []
    for example in tqdm(data, total=len(data), desc='convert hotpot qa to tri'):
        if not is_test:
            sf_dict = {x[0]:x[1] for x in example['supporting_facts']}
        else:
            sf_dict = {}
        if squad_key == 'gold':
            for index in range(len(example['context'])):
                example['context'][index][-1][-1] = example['context'][index][-1][-1] + ' '
            context = example['context']
            retrievals = example['retrieval']
            see_para = {}
            context_merged = [x for x in context]
            for para in retrievals:
                context_merged.append([para[2], para[3]])
        elif squad_key == 'selected':
            context_merged = example['selected']
        else:
            raise ValueError('squad key error!')
        labels = []
        paras  = []
        negtive_num = 0
        positive_num = 0
        context_merged_dict = {x[0]:x for x in context_merged}
        context_merged = [x for _, x in context_merged_dict.items()]
        for para in context_merged:
            if negtive_num >= max_negtive_num:
                break
            title = para[0]
            para_text = "".join(para[1])
            label = "0"
            if title in sf_dict:
                label = "1"
                positive_num += 1
            if label == '0':
                negtive_num += 1
            labels.append(label)
            paras.append(para_text)
        if is_training:
            assert positive_num == 2
        while negtive_num < max_negtive_num:
            print('negtive num {} is smaller than max negtive num {}'.format(negtive_num, max_negtive_num))
            paras.append("negtive")
            labels.append("0")
            negtive_num += 1
        tri_examples.append({'question': example['question'], 'paras': paras,
                             '_id': example['_id'], 'labels': labels, 'title_para': para})
    return tri_examples


def convert_sf2cls(data_list, is_training=False, is_test=False):
    para_examples = []
    positive = 0
    negative = 0
    para_num = 0
    for example in data_list:
        _id = example['_id']
        answer = example.get('answer', "")
        question = example['question']
        sf = example.get("supporting_facts", [["", 0]])
        sf_dict_title_index = {x[0] + "_" + str(x[1]): x[0] for x in sf}
        context_merged = example['selected']

        for para in context_merged:
            title = para[0]
            sentences = para[1]
            for sentence_index, sentence in enumerate(sentences):
                title_index = title + "_" + str(sentence_index)
                label = "0"
                if title_index in sf_dict_title_index:
                    label = "1"
                para_example = {'id': title, 'squad_id': _id, 'question': answer + "</s>" + question, 'document': sentence,
                                 'label': label, 'sentence_index': sentence_index}
                if label == '1':
                    positive += 1
                else:
                    negative += 1
                para_num += 1
                para_examples.append(para_example)

    print('sf para example num is {}, positive {}, negtive {}, rate {}'.format(para_num, positive, negative,
                                                                               negative / positive))
    return para_examples

def convert_yesno(data_list, mode='train'):
    para_examples = []
    positive = 0
    negative = 0
    para_num = 0
    yes_no_num = 0
    for example in data_list:
        _id = example['_id']
        if mode != "test":
            answer = example['answer']
        else:
            answer = ""
        if answer in "yesno":
            yes_no_num += 1
        question = example['question']
        if mode == 'test':
            sf = [["", 0]]
        else:
            sf = example['supporting_facts']
        sf_dict_title_index = {x[0] + "_" + str(x[1]): x[0] for x in sf}
        if mode == 'train':
            context_merged = example['selected']
        else:
            context_merged = [example['selected'][0]]

        for para in context_merged:
            title = para[0]
            para_text = "".join(para[1])
            label = "0"
            if answer == 'yes':
                label = "1"
            elif answer == "no":
                label = "2"

            para_example = {'id': title, 'squad_id': _id, 'question': question, 'document': para_text,
                             'label': label, "title_para": para}
            if label == '1' or label == '2':
                positive += 1
            else:
                negative += 1
            para_num += 1
            para_examples.append(para_example)
    if mode != 'test':
        print('yes no num is {} yes no para example num is {}, positive {}, negtive {}, rate {}'.format(yes_no_num, para_num, positive, negative,
                                                                               negative / positive))
    return para_examples

def convert2second_step(data_list, mode='train'):
    para_examples = []
    positive = 0
    negative = 0
    para_num = 0
    for example in data_list:
        _id = example['_id']
        answer = example['answer']
        question = example['question']
        sf = example['supporting_facts']
        sf_dict = {x[0]:x for x in sf}
        context_merged = example['selected']
        context_sampled = context_merged[:]
        permutatin_indices = [[0, 1], [0, 2],[1, 2], [1, 0], [0, 3]]
        for permutatin_indice in permutatin_indices:
            if permutatin_indice[0] >= len(context_merged) or permutatin_indice[1] >= len(context_merged):
                print('{} para is not enough for second step {}'.format(len(context_merged), mode))
                continue
            context_sampled_new = [context_sampled[x] for x in permutatin_indice]
            para0 = context_sampled_new[0]
            para1 = context_sampled_new[1]
            title0 = para0[0]
            para_text0 = "".join(para0[1])
            title1 = para1[0]
            para_text1 = "".join(para1[1])

            label = "0"
            if title1 in sf_dict:
                label = "1"
            para_example = {'id': title0, 'squad_id': _id, 'question': question + " " + para_text0, 'document': para_text1,
                             'label': label, "title_para": context_sampled_new}
            if label == '1':
                positive += 1
            else:
                negative += 1
            para_num += 1
            para_examples.append(para_example)
    if mode != 'test':
        print('second step para example num is {}, positive {}, negtive {}, rate {}'.format(para_num, positive, negative,
                                                                               negative / positive))
    return para_examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hotpot_file', default='', help='official label and retrieved paras')
    parser.add_argument('--converted_keys', default='retrieval', type=str,help='the key to be converted')
    parser.add_argument('--max_para_num', default=10, type=int)
    parser.add_argument('--output_dir', default='data', type=str)
    parser.add_argument('--split', default=1, type=int)
    parser.add_argument('--sub_key', default='', type=str)
    parser.add_argument('--mode', default='train', type=str)
    parser.add_argument('--para_score_th', default=0.5, type=float)

    parser.add_argument('--squad_key', default='gold', type=str)
    parser.add_argument('--distant', default=0, type=int)
    parser.add_argument('--rank2', default=0, type=int)
    args = parser.parse_args()
    data = read_json(args.hotpot_file)

    dir_name = os.path.join(os.path.dirname(os.path.abspath(args.hotpot_file)), args.output_dir + '_' + str(args.max_para_num))
    if not os.path.exists(dir_name):
        print('dir does not exist! mkdir {}'.format(dir_name))
        os.mkdir(dir_name)
    else:
        print('{} exist, overwrite files!'.format(dir_name))
    file_name = args.hotpot_file.split('/')[-1]
    print('convert mode is {}'.format(args.mode))
    assert args.split > 0
    if args.mode != 'train':
        args.split = 1
    chunk_size = len(data) // args.split

    data_lists = [data[chunk_size * chunk_index: chunk_size * (chunk_index + 1)] for chunk_index in range(args.split)]
    for data_num, data_list in enumerate(data_lists):
        dir_name_sub = dir_name
        if len(data_lists) > 1:
            dir_name_sub = os.path.join(dir_name, 'chunk{}'.format(data_num))
            if not os.path.exists(dir_name_sub):
                os.mkdir(dir_name_sub)
        print('convert chunk {} of {}'.format(data_num + 1, len(data_lists)))
        para_exampels = convert2para(data_list, max_para_num=args.max_para_num, mode=args.mode,
                                     converted_keys=args.converted_keys, sub_key=args.sub_key,
                                     distant=bool(args.distant), para_score_th=args.para_score_th, rank2=bool(args.rank2))
        write_line_json(para_exampels, dir_name_sub + "/{}.para".format(file_name))

        option = args.squad_key
        squad_data = convert2squad(data_list, option=option, mode=args.mode, max_para_num=args.max_para_num)
        squad_file = dir_name_sub + "/{}.squad.{}".format(file_name, args.squad_key)
        write_json(squad_data, squad_file)

        # tri_file = dir_name_sub + "/{}.para.tri".format(file_name)
        # tri_examples = convert2tri(data_list, max_negtive_num=args.max_para_num - 2, squad_key=args.squad_key)
        # write_line_json(tri_examples, tri_file)
        # sf_examples = convert_sf2cls(data_list, is_training=is_training)
        # sf_file = dir_name_sub + "/{}.para.sf".format(file_name)
        # write_line_json(sf_examples, sf_file)
        yes_no_examples = convert_yesno(data_list, mode=args.mode)
        yes_no_file = dir_name_sub + "/{}.para.yes_no".format(file_name)
        write_line_json(yes_no_examples, yes_no_file)
        # second_step_examples = convert2second_step(data_list, mode=args.mode)
        # second_step_file = dir_name_sub + "/{}.para.2nd".format(file_name)
        # write_line_json(second_step_examples, second_step_file)



