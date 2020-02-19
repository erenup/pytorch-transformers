import argparse
import json
def read_json(input_file):
    data = []
    with open(input_file, 'r') as f:
        data = json.load(f)
    return data

def write_line_json(examples, output_file):
    print('write {} examples to {}'.format(len(examples), output_file))
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example))
            f.write('\n')



def write_json(data, file):
    print('write json format data to {}'.format(file))
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(data, f)

def read_line_json(file):
    print('read line json from {}'.format(file))
    data = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:

            data.append(json.loads(line.strip('\n')))
    return  data



def combine_qa_results(datalist, qa_results):
    datalist_new = []
    for example in datalist:
        _id = example['_id']
        if _id not in qa_results:
            print('missing qa prediction for {}'.format(_id))
            continue
        example['pred_answer'] = qa_results[_id]
        datalist_new.append(example)
    return datalist_new

def combine_yes_no(ranked_qa_result, yes_no_input, yes_no_result_logits):
    print('combine yes no results')
    preds = yes_no_result_logits['preds']
    labels = yes_no_result_logits['labels']
    logits = yes_no_result_logits['logits']
    label_dict = {1: '1', 0: '0', 2: '2'}
    yes_no_dict = {'0': "yesno", '1': 'yes', "2": "no"}
    pred_yes_no_dict = {}
    print('len preds: {}, len labels: {}'.format(len(preds), len(labels)))
    for example, pred, label, logit in zip(yes_no_input, preds, labels, logits):
        label = label_dict[label]
        assert label == example['label']
        example['pred'] = pred
        example['logit'] = logit
        _id = example['squad_id']
        pred = label_dict[pred]
        pred_yes_no_dict[_id] = yes_no_dict[pred]
    for _id, qa_result in ranked_qa_result.items():
        if _id not in pred_yes_no_dict:
            print('miss yes no pred for {}'.format(_id))
            continue
        # if pred_yes_no_dict[_id] != "yesno":
        if ranked_qa_result[_id] == "":
            ranked_qa_result[_id] = pred_yes_no_dict[_id]
    return ranked_qa_result

def convert_sf2cls(data_list, sf_max_para=4):
    para_examples = []
    para_num = 0
    for example in data_list:
        _id = example['_id']
        answer = example.get('pred_answer', "")
        question = example['question']
        sf = example.get("supporting_facts", [["", 0]])
        sf_dict_title_index = {x[0] + "_" + str(x[1]): x[0] for x in sf}
        # confidence_num = 0
        # context_merged_confidence = []
        # for rt in example['ranked']:
        #     if rt[1] > 0.5:
        #         context_merged_confidence.append([rt[2], rt[3]])
        #         confidence_num += 1
        # if confidence_num >= 2:
        #     context_merged = context_merged_confidence[:]
        # else:
        context_merged =[[x[2], x[3]] for x in  example['ranked'][:sf_max_para]]
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
                para_num += 1
                para_examples.append(para_example)
    print('sf para example num is {}'.format(para_num))
    return para_examples

def combine_sf(sf_input, sf_result_logits, th=0):
    print('combine sf results')
    if len(sf_input) != len(sf_result_logits['preds']):
        print('sf input {}, sf result {}'.format(len(sf_input), len(sf_result_logits['preds'])))
    assert len(sf_input) == len(sf_result_logits['preds'])
    preds = sf_result_logits['preds']
    labels = sf_result_logits['labels']
    logits = sf_result_logits['logits']
    label_dict = {1: '1', 0: '0'}
    pred_sf_dict = {}
    print('len preds: {}, len labels: {}'.format(len(preds), len(labels)))
    for example, pred, label, logit in zip(sf_input, preds, labels, logits):
        label = label_dict[label]
        assert label == example['label']
        example['pred'] = pred
        example['logit'] = logit
        _id = example['squad_id']
        if not pred_sf_dict.get(_id, []):
            pred_sf_dict[_id] = [[example['id'], example['sentence_index'], logit[1]]]
        else:
            pred_sf_dict[_id].append([example['id'], example['sentence_index'], logit[1]])
    for _id, sf_list in pred_sf_dict.items():
        pred_sf_dict[_id] = sorted(sf_list, key=lambda x: x[2], reverse=True)
        sf_list_new = pred_sf_dict[_id][:2]
        for x in pred_sf_dict[_id][2:]:
            if x[2] > th:
                sf_list_new.append(x)
        sf_list_new = [[x[0], x[1]] for x in sf_list_new]
        pred_sf_dict[_id] = sf_list_new[:]


    return pred_sf_dict


if __name__ == "__main__":
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('--option', default='1', type=str)
    parser.add_argument('--ranked_file', default="", type=str)
    parser.add_argument('--ranked_qa_result', default="", type=str)
    parser.add_argument('--yes_no_cls_input', default='', type=str)
    parser.add_argument('--yes_no_result_file', default="", type=str)
    parser.add_argument('--para_sf_topn', default=2, type=int)

    parser.add_argument('--sf_result_file', default='', type=str)
    parser.add_argument('--sf_input', default='', type=str)
    parser.add_argument('--qa_result', default='', type=str)
    parser.add_argument('--sf_th', default=0, type=float)
    parser.add_argument('--gold_file', default='', type=str)

    args = parser.parse_args()
    if args.option == '1':
        print('1st step!')
        ranked_list = read_json(args.ranked_file)
        ranked_qa_result = read_json(args.ranked_qa_result)

        yes_no_input = read_line_json(args.yes_no_cls_input)
        yes_no_result_logits = read_json(args.yes_no_result_file)
        ranked_qa_result_with_yes = combine_yes_no(ranked_qa_result, yes_no_input, yes_no_result_logits)

        ranked_list_with_qa_result = combine_qa_results(ranked_list, ranked_qa_result_with_yes)

        sf_para_examples = convert_sf2cls(ranked_list_with_qa_result, sf_max_para=args.para_sf_topn)
        yes_no_input_dir = os.path.dirname(os.path.abspath(args.yes_no_cls_input))
        sf_output_dir = os.path.join(yes_no_input_dir, 'pred_sf_para')
        qa_result_file = os.path.join(yes_no_input_dir, '.qa.result')
        write_json(ranked_qa_result_with_yes, qa_result_file)
        if not os.path.exists(sf_output_dir):
            os.mkdir(sf_output_dir)
        sf_output_file = os.path.join(sf_output_dir, 'dev.json.para.sf')
        if os.path.exists(sf_output_file):
            print('overwrite {}'.format(sf_output_file))
        write_line_json(sf_para_examples, sf_output_file)


    elif args.option == '2':
        print('second step!')
        qa_result = read_json(args.qa_result)
        sf_input = read_line_json(args.sf_input)
        sf_input_dir = os.path.dirname(os.path.abspath(args.sf_input))
        sf_pred_dict_file = os.path.join(sf_input_dir, 'sf.result')
        sf_result_logits = read_json(args.sf_result_file)
        pred_sf_dict = combine_sf(sf_input, sf_result_logits, th=args.sf_th)
        result = {'answer': qa_result, 'sp': pred_sf_dict}
        result_output_file = os.path.join(sf_input_dir, 'final.result')
        write_json(result, result_output_file)
        from evaluation_script import eval
        eval(result_output_file, args.gold_file)









