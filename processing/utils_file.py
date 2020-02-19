import json
import unicodedata
import os
from tqdm import tqdm
def read_json(file_name):
    print('read json from {}'.format(file_name))
    with open(file_name, 'r') as f:
        data = json.load(f)
        return data


def read_line_json(file_name):
    print('read line json from {}'.format(file_name))
    data = []
    with open(file_name, 'r') as f:
        for line in tqdm(f):
            data.append(json.loads(line.strip('\n')))
    return data


def write_json(data, file_name):
    print('write json to {}'.format(file_name))
    with open(file_name, 'w') as f:
        json.dump(data, f)


def write_line_json(data, file_name):
    print('write line json to {}'.format(file_name))
    with open(file_name, 'w') as f:
        for line in tqdm(data, total=len(data)):
            f.write(json.dumps(line))
            f.write('\n')

def normalize_text(text):
    return unicodedata.normalize("NFKD", text)

def check_make_dir(dir_name):
    if not os.path.exists(dir_name):
        print('dir does not exist! mkdir {}'.format(dir_name))
        os.mkdir(dir_name)
    else:
        print('{} exist, overwrite files!'.format(dir_name))