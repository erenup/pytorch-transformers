# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" GLUE processors and helpers """

import logging
import os
import json
from tqdm import tqdm
from transformers.data.processors.utils import DataProcessor, InputExample
from functools import partial
from multiprocessing import cpu_count
from multiprocessing import Pool

logger = logging.getLogger(__name__)
from sklearn.metrics import precision_recall_fscore_support


def simple_accuracy(preds, labels):
    return (preds == labels).mean()

class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

class HotpotProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""
    def _read_json_line(self, file_path):
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                data.append(json.loads(line.strip('\n')))
        return data

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json_line(os.path.join(data_dir, "train.json.para")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json_line(os.path.join(data_dir, "dev.json.para")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        from tqdm import tqdm
        nums = 0
        for (i, line) in tqdm(enumerate(lines), total=len(lines), desc='read para examples'):
            guid = "%s-%s" % (set_type, str(line["squad_id"]) + str(line['id']))
            text_a = line['question']
            text_b = line['document']
            label = line['label']
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
            nums += 1
            # if nums > 2000:
            #     break
        return examples

class HotpotSFProcessor(HotpotProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json_line(os.path.join(data_dir, "train.json.para.sf")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json_line(os.path.join(data_dir, "dev.json.para.sf")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

class HotpotYesNoProcessor(HotpotProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json_line(os.path.join(data_dir, "train.json.para.yes_no")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json_line(os.path.join(data_dir, "dev.json.para.yes_no")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2"]
class Hotpot2ndNoProcessor(HotpotProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json_line(os.path.join(data_dir, "train.json.para.2nd")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json_line(os.path.join(data_dir, "dev.json.para.2nd")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]


hotpot_tasks_num_labels = {
    "hotpot": 2,
    "hotpot_sf": 2,
    "hotpot_yes_no": 3,
    "hotpot_2nd": 2,
    "sst-2": 2,
}

hotpot_processors = {
    "hotpot": HotpotProcessor,
    "hotpot_sf": HotpotSFProcessor,
    "hotpot_yes_no": HotpotYesNoProcessor,
    "hotpot_2nd": Hotpot2ndNoProcessor,
    "sst-2": Sst2Processor,
}

hotpot_output_modes = {
    "hotpot": "classification",
    "hotpot_sf": "classification",
    "hotpot_yes_no": "classification",
    "hotpot_2nd": "classification",
    "sst-2": "classification",
}


def hotpot_compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "hotpot":
        return {"result": precision_recall_fscore_support(y_pred=preds, y_true=labels)}
    elif task_name == "hotpot_sf":
        return {"result": precision_recall_fscore_support(y_pred=preds, y_true=labels)}
    elif task_name == "hotpot_yes_no":
        return {"result": precision_recall_fscore_support(y_pred=preds, y_true=labels)}
    elif task_name == "hotpot_2nd":
        return {"result": precision_recall_fscore_support(y_pred=preds, y_true=labels)}
    elif task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)