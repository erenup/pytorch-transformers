from utils_file import read_line_json, write_json, write_line_json
import argparse
import os
import re
sent_regex = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s')
import glob

def convert_hardem2para(hard_em_list):
    para_examples = []
    negative = 0
    positive = 0
    for example in hard_em_list:
        answers = example['answers']
        contexts = example['context']
        _id = example['id']
        final_answers = example['final_answers']
        question = example['question']
        final_answers_squad = [{"text": answer, "answer_start": -1} for answer in final_answers]
        total = len(contexts)
        for index, ac in enumerate(zip(answers, contexts)):
            answer, context = ac
            label = '0'
            if answer:
                label = '1'
            context = " ".join(context)
            para_examples.append({'id': str(_id) + "_" + str(index), 'squad_id': _id, 'question': question, 'document': context,
                                'squad_answers': final_answers_squad, 'label': label, 'squad_context': '', 'tfidf_score': total - index})
            if label == '1':
                positive += 1
            else:
                negative += 1
    print('negative {}, positive {}'.format(negative, positive))
    return para_examples



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hardem_dir', default='', type=str)
    parser.add_argument('--hardem_task', default='', type=str)
    args = parser.parse_args()
    dev_file = os.path.join(args.hardem_dir, args.hardem_task + '-dev.json')
    dev_data = read_line_json(dev_file)
    output_dev_file = os.path.join(args.hardem_dir, 'dev.json.para')
    dev_para_examples = convert_hardem2para(dev_data)
    write_line_json(dev_para_examples, output_dev_file)

    train_files = glob.glob('{}/{}-train*.json'.format(args.hardem_dir, args.hardem_task))
    train_data = []
    for train_file in train_files:
        train_data.extend(read_line_json(train_file))
    output_train_file = os.path.join(args.hardem_dir, 'train.json.para')
    train_para_example = convert_hardem2para(train_data)
    write_line_json(train_para_example, output_train_file)

    test_file = os.path.join(args.hardem_dir, args.hardem_task + '-test.json')
    test_data = read_line_json(test_file)
    output_test_file = os.path.join(args.hardem_dir, 'test.json.para')
    test_para_examples = convert_hardem2para(test_data)
    write_line_json(test_para_examples, output_test_file)



