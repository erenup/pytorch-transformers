from utils_file import read_json, write_json
import argparse
import re
sent_regex = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s')

def convert2hotpot_format(squad_data):
    print('convert squad format to hotpot format')
    hotpot_format = []
    for title_para in squad_data:
        title = title_para['title']
        for para in title_para['paragraphs']:
            context = para['context']
            context_sents = sent_regex.split(context)
            for qa in para['qas']:
                answers = qa['answers']
                question = qa['question']
                _id = qa['id']
                answer_text = answers[0]['text']
                if '. ' in answer_text:
                    context_sents = [context]
                sp_index = -1
                for sent_index, sent in enumerate(context_sents):
                    if answer_text.lower() in sent.lower():
                        sp_index = sent_index
                if sp_index == -1:
                    print('answer_text can not find in sentences of squad')
                hotpot_example = {'_id':_id, 'answer':answer_text, 'question':question,
                                  'context': [[title, context_sents]], 'supporting_facts': [title, sp_index],
                                  'type': 'squad', 'level': 'easy', 'squad_answers': answers}
                hotpot_format.append(hotpot_example)
    return hotpot_format



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--squad_file', default='', type=str)
    args = parser.parse_args()
    squad_data = read_json(args.squad_file)['data']
    hotpot_format = convert2hotpot_format(squad_data)
    hotpot_file = args.squad_file + '.hotpot'
    write_json(hotpot_format, hotpot_file)
