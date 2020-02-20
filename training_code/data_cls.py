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
import random
import copy
from tqdm import tqdm
from transformers.data.processors.utils import DataProcessor, InputExample
from functools import partial
from multiprocessing import cpu_count
from multiprocessing import Pool

logger = logging.getLogger(__name__)
from sklearn.metrics import precision_recall_fscore_support

class DCInputFeatures(object):
    """
    A single set of features of data.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """

    def __init__(self, input_ids, attention_mask=None, token_type_ids=None,
                 input_ids_a=None, attention_mask_a=None, token_type_ids_a=None,
                 input_ids_b=None, attention_mask_b=None, token_type_ids_b=None, label=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids

        self.input_ids_a = input_ids_a
        self.attention_mask_a = attention_mask_a
        self.token_type_ids_a = token_type_ids_a

        self.input_ids_b = input_ids_b
        self.attention_mask_b = attention_mask_b
        self.token_type_ids_b = token_type_ids_b
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

def dcbert_convert_examples_to_feature(example,
    tokenizer=None,
    max_length=512,
    task=None,
    label_list=None,
    output_mode=None,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    mask_padding_with_zero=True,
    max_length_a=64,
    max_length_b=200,
):
    label_map = {label: i for i, label in enumerate(label_list)}
    inputs = tokenizer.encode_plus(example.text_a, example.text_b, add_special_tokens=True, max_length=max_length, )
    input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

    inputs_a = tokenizer.encode_plus(example.text_a, max_length=max_length_a, )
    input_ids_a, token_type_ids_a = inputs_a["input_ids"], inputs_a["token_type_ids"]

    inputs_b = tokenizer.encode_plus(example.text_b, max_length=max_length_b, )
    input_ids_b, token_type_ids_b = inputs_b["input_ids"], inputs_b["token_type_ids"]

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
    attention_mask_a = [1 if mask_padding_with_zero else 0] * len(input_ids_a)
    attention_mask_b = [1 if mask_padding_with_zero else 0] * len(input_ids_b)

    # Zero-pad up to the sequence length.
    padding_length = max_length - len(input_ids)
    padding_length_a = max_length_a - len(input_ids_a)
    padding_length_b = max_length_b - len(input_ids_b)
    if pad_on_left:
        input_ids = ([pad_token] * padding_length) + input_ids
        attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
        token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids

        input_ids_a = ([pad_token] * padding_length_a) + input_ids_a
        attention_mask_a = ([0 if mask_padding_with_zero else 1] * padding_length_a) + attention_mask_a
        token_type_ids_a = ([pad_token_segment_id] * padding_length_a) + token_type_ids_a

        input_ids_b = ([pad_token] * padding_length_b) + input_ids_b
        attention_mask_b = ([0 if mask_padding_with_zero else 1] * padding_length_b) + attention_mask_b
        token_type_ids_b = ([pad_token_segment_id] * padding_length_b) + token_type_ids_b
    else:
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        input_ids_a = input_ids_a + ([pad_token] * padding_length_a)
        attention_mask_a = attention_mask_a + ([0 if mask_padding_with_zero else 1] * padding_length_a)
        token_type_ids_a = token_type_ids_a + ([pad_token_segment_id] * padding_length_a)

        input_ids_b = input_ids_b + ([pad_token] * padding_length_b)
        attention_mask_b = attention_mask_b + ([0 if mask_padding_with_zero else 1] * padding_length_b)
        token_type_ids_b = token_type_ids_b + ([pad_token_segment_id] * padding_length_b)

    assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
    assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
        len(attention_mask), max_length
    )
    assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(
        len(token_type_ids), max_length
    )

    assert len(input_ids_a) == max_length_a, "Error with input length {} vs {}".format(len(input_ids_a), max_length_a)
    assert len(attention_mask_a) == max_length_a, "Error with input length {} vs {}".format(
        len(attention_mask_a), max_length_a
    )
    assert len(token_type_ids_a) == max_length_a, "Error with input length {} vs {}".format(
        len(token_type_ids_a), max_length_a
    )

    assert len(input_ids_b) == max_length_b, "Error with input length {} vs {}".format(len(input_ids_b), max_length_b)
    assert len(attention_mask_b) == max_length_b, "Error with input length {} vs {}".format(
        len(attention_mask_b), max_length_b
    )
    assert len(token_type_ids_b) == max_length_b, "Error with input length {} vs {}".format(
        len(token_type_ids_b), max_length_b
    )

    if output_mode == "classification":
        label = label_map[example.label]
    elif output_mode == "regression":
        label = float(example.label)
    else:
        raise KeyError(output_mode)

    return DCInputFeatures(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
            input_ids_a=input_ids_a, attention_mask_a=attention_mask_a, token_type_ids_a=token_type_ids_a,
            input_ids_b=input_ids_b, attention_mask_b=attention_mask_b, token_type_ids_b=token_type_ids_b, label=label
        )


def dcbert_convert_examples_to_features(
    examples,
    tokenizer,
    max_length=512,
    task=None,
    label_list=None,
    output_mode=None,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    mask_padding_with_zero=True,
    max_length_a=64,
    max_length_b=200,
):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    logging.info('converting features with dcbert')
    features = []
    with Pool(cpu_count()) as p:
        annotate_ = partial(
            dcbert_convert_examples_to_feature,
            tokenizer=tokenizer,
            max_length=max_length,
            task=task,
            label_list=label_list,
            output_mode=output_mode,
            pad_on_left=pad_on_left,
            pad_token=pad_token,
            pad_token_segment_id=pad_token_segment_id,
            mask_padding_with_zero=mask_padding_with_zero,
            max_length_a=max_length_a,
            max_length_b=max_length_b,
        )
        features = list(
            tqdm(
                p.imap(annotate_, examples, chunksize=32),
                total=len(examples),
                desc="convert {} examples".format(task),
            )
        )
    return features


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

class RankerProcessor(DataProcessor):
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

    def get_train_examples(self, data_dir, nsp=1.0):
        """See base class."""
        return self._create_examples(self._read_json_line(os.path.join(data_dir, "train.json.para")), "train", nsp=nsp)

    def get_dev_examples(self, data_dir, nsp=1.0):
        """See base class."""
        return self._create_examples(self._read_json_line(os.path.join(data_dir, "dev.json.para")), "dev", nsp=nsp)

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type, nsp=1.0):
        """Creates examples for the training and dev sets."""
        examples = []
        from tqdm import tqdm
        nums = 0
        for (i, line) in tqdm(enumerate(lines), total=len(lines), desc='read para examples'):
            guid = "%s-%s" % (set_type, str(line["squad_id"]) + str(line['id']))
            text_a = line['question']
            text_b = line['document']
            label = line['label']

            if line['label'] == '0' and random.random() > nsp:
                continue
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
            nums += 1
            # if nums > 2000:
            #     break
        return examples

class HotpotSFProcessor(RankerProcessor):
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

class HotpotYesNoProcessor(RankerProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir, nsp=1.0):
        """See base class."""
        return self._create_examples(self._read_json_line(os.path.join(data_dir, "train.json.para.yes_no")), "train")

    def get_dev_examples(self, data_dir, nsp=1.0):
        """See base class."""
        return self._create_examples(self._read_json_line(os.path.join(data_dir, "dev.json.para.yes_no")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2"]
class Hotpot2ndNoProcessor(RankerProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir, nsp=1.0):
        """See base class."""
        return self._create_examples(self._read_json_line(os.path.join(data_dir, "train.json.para.2nd")), "train")

    def get_dev_examples(self, data_dir, nsp=1.0):
        """See base class."""
        return self._create_examples(self._read_json_line(os.path.join(data_dir, "dev.json.para.2nd")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]


hotpot_tasks_num_labels = {
    "ranker": 2,
    "hotpot_sf": 2,
    "hotpot_yes_no": 3,
    "hotpot_2nd": 2,
    "sst-2": 2,
}

hotpot_processors = {
    "ranker": RankerProcessor,
    "hotpot_sf": HotpotSFProcessor,
    "hotpot_yes_no": HotpotYesNoProcessor,
    "hotpot_2nd": Hotpot2ndNoProcessor,
    "sst-2": Sst2Processor,
}

hotpot_output_modes = {
    "ranker": "classification",
    "hotpot_sf": "classification",
    "hotpot_yes_no": "classification",
    "hotpot_2nd": "classification",
    "sst-2": "classification",
}


def hotpot_compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "ranker":
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