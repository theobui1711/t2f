# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import os
import logging
import random
from typing import Dict, Generator, Tuple, List
from abc import ABC, abstractmethod
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer, torch_distributed_zero_first
from transformers import default_data_collator

from arguments import DataTrainingArguments
from input_example import InputFeatures, InputExample
from input_formats import INPUT_FORMATS
from output_formats import OUTPUT_FORMATS, key2ind, ind2key


class BaseDatasetO(Dataset, ABC):
    """
    Base class for all datasets.
    """
    name = None  # name of the dataset
    data_name = None  # name of the directory, if different from the name of the dataset
    task_descriptor = None  # string to prepend to every input sentence if multitask=True (default is self.name)

    default_input_format = 'plain'
    default_output_format = None
    default_data_dir = 'data'

    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            max_input_length: int,
            max_output_length: int,
            overwrite_cache: bool = False,
            mode: str = 'train',
            local_rank: int = -1,
            train_subset: float = 1,  # a number < 1 is to use only a subset of training data (random)
            seed: int = None,
            shuffle: bool = True,
            data_args: DataTrainingArguments = None,
            is_eval: bool = False,
    ):
        if seed is not None:
            # set random seed for repeatability
            random.seed(seed)

        self.data_args = data_args
        self.tokenizer = tokenizer

        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

        self.input_format = INPUT_FORMATS[
            data_args.input_format if data_args.input_format is not None else self.default_input_format
        ]()
        self.output_format = OUTPUT_FORMATS[
            data_args.output_format if data_args.output_format is not None else self.default_output_format
        ]()

        self.data_path = data_args.data_dir if data_args.data_dir is not None else self.default_data_dir

        self.is_eval = is_eval
        self.eval_nll = data_args.eval_nll

        cached_data_file = os.path.join(
            self.data_dir(),
            f"cached_{self.name}_{mode}_{tokenizer.__class__.__name__}_{max_input_length}_{max_output_length}_{self.data_args.exp}"
            f"{'_multitask' if data_args.multitask else ''}.pth"
        )

        with torch_distributed_zero_first(local_rank):
            # make sure only the first process in distributed training processes the dataset,
            # and the others can use the cached version

            if os.path.exists(cached_data_file) and not overwrite_cache:
                self.load_cached_data(cached_data_file)

            else:
                self.load_schema()  # here the dataset can load information such as entity/relation types
                self.examples = self.load_data(mode=mode, seed=seed)

                # assign examples to this dataset
                for example in self.examples:
                    example.dataset = self

                self.features = self.compute_features(
                    max_input_length=max_input_length,
                    max_output_length=max_output_length,
                    multitask=data_args.multitask,
                )

                if local_rank in [-1, 0]:
                    # save data
                    self.save_data(cached_data_file)

            # shuffle indices
            self.indices = list(range(len(self.examples)))
            if seed is not None and shuffle:
                random.shuffle(self.indices)

            # compute effective size of the dataset
            self.effective_size = round(train_subset * len(self.examples))
            if train_subset != 1:
                logging.info(f"Effective dataset size reduced to {self.effective_size} ({train_subset * 100:.0f}%)")

    def __repr__(self):
        return f'Dataset {self.name}'

    def __len__(self):
        return self.effective_size

    def __getitem__(self, i: int) -> InputFeatures:
        return self.features[self.indices[i]]

    def get_example(self, i: int) -> InputExample:
        return self.examples[self.indices[i]]

    def data_dir(self):
        if self.data_name is not None:
            return os.path.join(self.data_path, self.data_name)
        else:
            return os.path.join(self.data_path, self.name)

    def load_cached_data(self, cached_data_file: str):
        d = torch.load(cached_data_file)
        self.examples, self.features = d['examples'], d['features']

    def save_data(self, cached_data_file: str):
        torch.save({
            'examples': self.examples,
            'features': self.features,
        }, cached_data_file)

    def load_schema(self):
        """
        Load extra dataset information, such as entity/relation types.
        """
        pass

    @abstractmethod
    def load_data_single_split(self, split: str, seed: int = None) -> List[InputExample]:
        """
        Load data for a single split (train, dev, or test).
        """
        pass

    def load_data(self, mode: str, seed: int = None) -> List[InputExample]:
        """
        Load all data, where 'mode' is a list of comma-separated splits to use.
        """
        examples = []

        if isinstance(mode, str):
            splits = mode.split(',')
        else:
            assert isinstance(mode, (list, tuple))
            splits = mode

        for split in splits:
            examples += self.load_data_single_split(split, seed=seed)

        return examples

    def _warn_max_sequence_length(self, max_sequence_length: int, sentences: List[str], name: str):
        max_length_needed = max(len(self.tokenizer.tokenize(x)) for x in sentences)
        if max_length_needed > max_sequence_length:
            logging.warning(
                f'Max sequence length is {max_sequence_length} but the longest {name} sequence is '
                f'{max_length_needed} long'
            )

    @staticmethod
    def batch_encode_output_(output_index, max_output_length):
        input_ids = []
        attention_mask = []
        for ins in output_index:
            input_ids.append(ins + (max_output_length - len(ins)) * [0])
            attention_mask.append(len(ins) * [1] + (max_output_length - len(ins)) * [0])
        input_ids = torch.tensor(input_ids, dtype=torch.int64)
        attention_mask = torch.tensor(attention_mask, dtype=torch.int64)
        output_tok = {'input_ids': input_ids, 'attention_mask': attention_mask}
        return output_tok

    def compute_features(self, max_input_length: int, max_output_length: int, multitask: bool = False):
        input_sentences = [self.input_format.format_input(example, multitask=multitask) for example in self.examples]
        if self.data_args.output_format_type == 'short-relation':
            output_sentences = [self.output_format.format_short_output_with_relation(example) for example in
                                self.examples]
        elif self.data_args.output_format_type == 'short':
            output_sentences = [self.output_format.format_short_output(example) for example in self.examples]
        elif self.data_args.output_format_type == 'long':
            output_sentences = [self.output_format.format_long_output(example) for example in self.examples]
        elif self.data_args.output_format_type == 'original':
            output_sentences = [self.output_format.format_short_output_(example) for example in self.examples]
        boundary_sentences = [' '.join(example.boundary_tokens) for example in self.examples]

        if self.data_args.boundary_in_where == 'Encoder':
            if self.data_args.exp.startswith('no_boundary'):
                pass
            else:
                # print("Boundary information is added to the end of the input sequence and used in Encoder!")
                logging.info("Boundary information is added to the end of the input sequence and used in Encoder.")
                input_sentences = [
                    ((self.input_format.format_input(example, multitask=multitask)) + ' '.join(example.boundary_tokens))
                    for example in self.examples]
                # input_sentences = [( ' '.join(example.boundary_tokens) ) + (self.input_format.format_input(example, multitask=multitask)) for example in self.examples] # reverse description and boundary token orders

        print(f'Example input sententece: {input_sentences[0]}')
        print(f'Example output sententece: {output_sentences[0]}')

        # logging.info(f'Example input sententece: {input_sentences[0]}')
        # logging.info(f'Example output sententece: {output_sentences[0]}')

        num_rooms = [len(example.rooms) for example in self.examples]
        regr_labels = []
        for example in self.examples:
            regr_label = []
            for room in example.rooms:
                regr_label.extend([room.x, room.y, room.h, room.w])
            regr_labels.append(regr_label)
        # check sanity of regression labels
        for i in range(len(num_rooms)):
            assert num_rooms[i] == len(regr_labels[i]) / 4

        print('**********')
        print(regr_labels[0])
        print('**********')
        print(num_rooms[0])

        input_tok = self.tokenizer.batch_encode_plus(
            input_sentences,
            max_length=max_input_length,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
        )
        self._warn_max_sequence_length(max_input_length, input_sentences, "input")

        # output_index = [self.output_format.format_output_index(example) for example in self.examples]
        # output_tok = self.batch_encode_output_(output_index, max_output_length)

        output_tok = self.tokenizer.batch_encode_plus(
            output_sentences,
            max_length=max_output_length,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
        )
        self._warn_max_sequence_length(max_output_length, output_sentences, "output")

        boundary_tok = self.tokenizer.batch_encode_plus(
            boundary_sentences,
            max_length=50,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
        )

        # assert len(input_tok.input_ids) == len(output_tok['input_ids'])
        # assert len(input_tok.input_ids) == len(boundary_tok.input_ids)
        assert input_tok.input_ids.size(0) == output_tok['input_ids'].size(0)
        assert input_tok.input_ids.size(0) == boundary_tok.input_ids.size(0)

        features = []
        if self.data_args.boundary_in_where == 'Encoder':
            for sentence_input_ids, att_mask, label_input_ids, num_room, regr_l in zip(input_tok.input_ids,
                                                                                       input_tok.attention_mask,
                                                                                       output_tok['input_ids'],
                                                                                       num_rooms, regr_labels):
                features.append(InputFeatures(
                    input_ids=sentence_input_ids.tolist(),
                    attention_mask=att_mask.tolist(),
                    labels=label_input_ids.tolist(),
                    num_rooms=num_room,
                    regr_labels=regr_l
                ))
            # for sentence_input_ids, att_mask, label_input_ids, num_room, regr_l, decoder_attention_mask in zip(input_tok.input_ids, input_tok.attention_mask,
            #                                                         output_tok['input_ids'], num_rooms, regr_labels, output_tok['attention_mask']):
            #     features.append(InputFeatures(
            #         input_ids=sentence_input_ids.tolist(),
            #         attention_mask=att_mask.tolist(),
            #         label_ids=label_input_ids.tolist(),
            #         num_rooms=num_room,
            #         regr_labels=regr_l,
            #         decoder_attention_mask = decoder_attention_mask.tolist()
            #     ))

        else:  # boundary_in_where == 'Decoder"
            for sentence_input_ids, att_mask, label_input_ids, boundary_input_ids, boundary_tok_mask in zip(
                    input_tok.input_ids, input_tok.attention_mask,
                    output_tok.input_ids, boundary_tok.input_ids, boundary_tok.attention_mask):
                features.append(InputFeatures(
                    input_ids=sentence_input_ids.tolist(),
                    attention_mask=att_mask.tolist(),
                    boundary_ids=boundary_input_ids.tolist(),
                    boundary_mask=boundary_tok_mask.tolist(),
                    label_ids=label_input_ids.tolist()
                ))

        return features

    @staticmethod
    def decode_new(prediction):
        prediction = prediction.tolist()
        string = ""
        for pre in prediction:
            if pre == 0:
                pass
            else:
                string += f'{ind2key[pre]} '
        return string

    def generate_output_sentences(self, data_args: DataTrainingArguments, model, device, batch_size: int, features) \
            -> Generator[Tuple[InputExample, str], None, None]:
        """
        Generate pairs (example, output_sentence) for evaluation.
        """
        test_data_loader = DataLoader(
            self,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=default_data_collator,
        )
        # total = len(test_data_loader)
        for i, inputs in tqdm(enumerate(test_data_loader), total=len(test_data_loader)):
            if data_args.boundary_in_where == 'Encoder':
                predictions = model.generate(
                    inputs['input_ids'].to(device),
                    max_length=data_args.max_output_seq_length_eval,
                    num_beams=data_args.num_beams
                )
            elif data_args.boundary_in_where == 'Decoder':
                predictions = model.generate(
                    inputs['input_ids'].to(device),
                    max_length=data_args.max_output_seq_length_eval,
                    num_beams=data_args.num_beams,
                    features=inputs['boundary_ids']
                )

            for j, (input_ids, label_ids, prediction) in enumerate(
                    zip(inputs['input_ids'], inputs['labels'], predictions)):
                if data_args.boundary_in_where == 'Encoder':
                    current_id = i * batch_size + j
                    example = self.get_example(current_id)
                    # output_sentence = self.decode_new(prediction)
                    # pre_list = prediction.tolist()
                    output_sentence = self.tokenizer.decode(prediction, skip_special_tokens=True,
                                                            clean_up_tokenization_spaces=False)
                elif data_args.boundary_in_where == 'Decoder':
                    current_id = i * batch_size + j
                    example = self.get_example(current_id)
                    output_sentence = self.tokenizer.decode(prediction[51:], skip_special_tokens=True,
                                                            clean_up_tokenization_spaces=False)

                yield example, output_sentence, None

    @abstractmethod
    def evaluate_dataset(self, data_args: DataTrainingArguments, model, device, batch_size: int, macro: bool = False) \
            -> Dict[str, float]:
        """
        Evaluate model on this dataset, returning the task-relevant metrics.
        """
        pass


class BaseDataset(Dataset, ABC):
    """
    Base class for all datasets.
    """
    name = None  # name of the dataset
    data_name = None  # name of the directory, if different from the name of the dataset
    task_descriptor = None  # string to prepend to every input sentence if multitask=True (default is self.name)

    default_input_format = 'plain'
    default_output_format = None
    default_data_dir = 'data'

    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            max_input_length: int,
            max_output_length: int,
            overwrite_cache: bool = False,
            mode: str = 'train',
            local_rank: int = -1,
            train_subset: float = 1,  # a number < 1 is to use only a subset of training data (random)
            seed: int = None,
            shuffle: bool = True,
            data_args: DataTrainingArguments = None,
            is_eval: bool = False,
    ):
        if seed is not None:
            # set random seed for repeatability
            random.seed(seed)

        self.data_args = data_args
        self.tokenizer = tokenizer

        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

        self.input_format = INPUT_FORMATS[
            data_args.input_format if data_args.input_format is not None else self.default_input_format
        ]()
        self.output_format = OUTPUT_FORMATS[
            data_args.output_format if data_args.output_format is not None else self.default_output_format
        ]()

        self.data_path = data_args.data_dir if data_args.data_dir is not None else self.default_data_dir

        self.is_eval = is_eval
        self.eval_nll = data_args.eval_nll

        cached_data_file = os.path.join(
            self.data_dir(),
            f"cached_{self.name}_{mode}_{tokenizer.__class__.__name__}_{max_input_length}_{max_output_length}_{self.data_args.exp}"
            f"{'_multitask' if data_args.multitask else ''}.pth"
        )

        with torch_distributed_zero_first(local_rank):
            # make sure only the first process in distributed training processes the dataset,
            # and the others can use the cached version

            if os.path.exists(cached_data_file) and not overwrite_cache:
                self.load_cached_data(cached_data_file)

            else:
                self.load_schema()  # here the dataset can load information such as entity/relation types
                self.examples = self.load_data(mode=mode, seed=seed)

                # assign examples to this dataset
                for example in self.examples:
                    example.dataset = self

                self.features = self.compute_features(
                    max_input_length=max_input_length,
                    max_output_length=max_output_length,
                    multitask=data_args.multitask,
                )

                if local_rank in [-1, 0]:
                    # save data
                    self.save_data(cached_data_file)

            # shuffle indices
            self.indices = list(range(len(self.examples)))
            if seed is not None and shuffle:
                random.shuffle(self.indices)

            # compute effective size of the dataset
            self.effective_size = round(train_subset * len(self.examples))
            if train_subset != 1:
                logging.info(f"Effective dataset size reduced to {self.effective_size} ({train_subset * 100:.0f}%)")

    def __repr__(self):
        return f'Dataset {self.name}'

    def __len__(self):
        return self.effective_size

    def __getitem__(self, i: int) -> dict:
        return self.features[self.indices[i]]

    def get_example(self, i: int) -> InputExample:
        return self.examples[self.indices[i]]

    def data_dir(self):
        if self.data_name is not None:
            return os.path.join(self.data_path, self.data_name)
        else:
            return os.path.join(self.data_path, self.name)

    def load_cached_data(self, cached_data_file: str):
        d = torch.load(cached_data_file)
        self.examples, self.features = d['examples'], d['features']

    def save_data(self, cached_data_file: str):
        torch.save({
            'examples': self.examples,
            'features': self.features,
        }, cached_data_file)

    def load_schema(self):
        """
        Load extra dataset information, such as entity/relation types.
        """
        pass

    @abstractmethod
    def load_data_single_split(self, split: str, seed: int = None) -> List[InputExample]:
        """
        Load data for a single split (train, dev, or test).
        """
        pass

    def load_data(self, mode: str, seed: int = None) -> List[InputExample]:
        """
        Load all data, where 'mode' is a list of comma-separated splits to use.
        """
        examples = []

        if isinstance(mode, str):
            splits = mode.split(',')
        else:
            assert isinstance(mode, (list, tuple))
            splits = mode

        for split in splits:
            examples += self.load_data_single_split(split, seed=seed)

        return examples

    def _warn_max_sequence_length(self, max_sequence_length: int, sentences: List[str], name: str):
        max_length_needed = max(len(self.tokenizer.tokenize(x)) for x in sentences)
        if max_length_needed > max_sequence_length:
            logging.warning(
                f'Max sequence length is {max_sequence_length} but the longest {name} sequence is '
                f'{max_length_needed} long'
            )

    @staticmethod
    def batch_encode_output_(output_index, max_output_length):
        input_ids = []
        attention_mask = []
        for ins in output_index:
            input_ids.append(ins + (max_output_length - len(ins)) * [0])
            attention_mask.append(len(ins) * [1] + (max_output_length - len(ins)) * [0])
        input_ids = torch.tensor(input_ids, dtype=torch.int64)
        attention_mask = torch.tensor(attention_mask, dtype=torch.int64)
        output_tok = {'input_ids': input_ids, 'attention_mask': attention_mask}
        return output_tok

    def tokenize_function(self, examples, max_length: int):
        return self.tokenizer.batch_encode_plus(
            examples,
            max_length=max_length,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )

    def compute_features(self, max_input_length: int, max_output_length: int, multitask: bool = False):
        input_sentences = [self.input_format.format_input(example, multitask=multitask) for example in self.examples]
        if self.data_args.output_format_type == 'short-relation':
            output_sentences = [self.output_format.format_short_output_with_relation(example) for example in
                                self.examples]
        elif self.data_args.output_format_type == 'short':
            output_sentences = [self.output_format.format_short_output(example) for example in self.examples]
        elif self.data_args.output_format_type == 'long':
            output_sentences = [self.output_format.format_long_output(example) for example in self.examples]
        elif self.data_args.output_format_type == 'original':
            output_sentences = [self.output_format.format_short_output_(example) for example in self.examples]
        boundary_sentences = [' '.join(example.boundary_tokens) for example in self.examples]

        # logging.info("Boundary information is added to the end of the input sequence and used in Encoder.")
        # input_sentences = [
        #     ((self.input_format.format_input(example, multitask=multitask)) + ' '.join(example.boundary_tokens))
        #     for example in self.examples]

        logging.info(f'Example input sentence: {input_sentences[0]}')
        logging.info(f'Example output sentence: {output_sentences[0]}')

        num_rooms = [len(example.rooms) for example in self.examples]
        regr_labels = []
        for example in self.examples:
            regr_label = []
            for room in example.rooms:
                regr_label.extend([room.x, room.y, room.h, room.w])
            regr_labels.append(regr_label)
        # check sanity of regression labels
        for i in range(len(num_rooms)):
            assert num_rooms[i] == len(regr_labels[i]) / 4

        input_tok = self.tokenize_function(input_sentences, max_input_length)
        self._warn_max_sequence_length(max_input_length, input_sentences, "input")

        # output_index = [self.output_format.format_output_index(example) for example in self.examples]
        # output_tok = self.batch_encode_output_(output_index, max_output_length)

        output_tok = self.tokenize_function(output_sentences, max_output_length)
        self._warn_max_sequence_length(max_output_length, output_sentences, "output")

        boundary_tok = self.tokenize_function(boundary_sentences, 50)

        assert input_tok.input_ids.size(0) == output_tok['input_ids'].size(0)
        assert input_tok.input_ids.size(0) == boundary_tok.input_ids.size(0)

        cuda_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        input_features_format = {
            'input_ids': None,
            'attention_mask': None,
            'num_rooms': None,
            'reg_labels': None,
            'decoder_boundary_ids': None,
            'decoder_boundary_mask': None,
            'label_ids': None
        }

        features = []

        for sentence_input_ids, att_mask, num_room, reg_l, \
                boundary_input_ids, boundary_tok_mask, label_input_ids \
                in zip(input_tok.input_ids, input_tok.attention_mask, num_rooms, regr_labels,
                       boundary_tok.input_ids, boundary_tok.attention_mask, output_tok.input_ids):
            input_features_format['input_ids'] = sentence_input_ids.tolist()
            input_features_format['attention_mask'] = att_mask.tolist()
            input_features_format['num_rooms'] = torch.tensor(num_room, dtype=torch.int64).to(cuda_device)
            input_features_format['reg_labels'] = torch.tensor(reg_l, dtype=torch.int64).to(cuda_device)
            input_features_format['decoder_boundary_ids'] = boundary_input_ids.tolist()
            input_features_format['decoder_boundary_mask'] = boundary_tok_mask.tolist()
            input_features_format['label_ids'] = label_input_ids.tolist()
            features.append(input_features_format)

        return features

    @staticmethod
    def decode_new(prediction):
        prediction = prediction.tolist()
        string = ""
        for pre in prediction:
            if pre == 0:
                pass
            else:
                string += f'{ind2key[pre]} '
        return string

    def generate_output_sentences(self, data_args: DataTrainingArguments, model, device, batch_size: int, features) \
            -> Generator[Tuple[InputExample, str], None, None]:
        """
        Generate pairs (example, output_sentence) for evaluation.
        """
        test_data_loader = DataLoader(
            self,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=default_data_collator,
        )
        # total = len(test_data_loader)
        for i, inputs in tqdm(enumerate(test_data_loader), total=len(test_data_loader)):
            if data_args.boundary_in_where == 'Encoder':
                predictions = model.generate(
                    inputs['input_ids'].to(device),
                    max_length=data_args.max_output_seq_length_eval,
                    num_beams=data_args.num_beams
                )
            elif data_args.boundary_in_where == 'Decoder':
                predictions = model.generate(
                    inputs['input_ids'].to(device),
                    max_length=data_args.max_output_seq_length_eval,
                    num_beams=data_args.num_beams,
                    features=inputs['boundary_ids']
                )

            for j, (input_ids, label_ids, prediction) in enumerate(
                    zip(inputs['input_ids'], inputs['labels'], predictions)):
                if data_args.boundary_in_where == 'Encoder':
                    current_id = i * batch_size + j
                    example = self.get_example(current_id)
                    # output_sentence = self.decode_new(prediction)
                    # pre_list = prediction.tolist()
                    output_sentence = self.tokenizer.decode(prediction, skip_special_tokens=True,
                                                            clean_up_tokenization_spaces=False)
                elif data_args.boundary_in_where == 'Decoder':
                    current_id = i * batch_size + j
                    example = self.get_example(current_id)
                    output_sentence = self.tokenizer.decode(prediction[51:], skip_special_tokens=True,
                                                            clean_up_tokenization_spaces=False)

                yield example, output_sentence, None

    @abstractmethod
    def evaluate_dataset(self, data_args: DataTrainingArguments, model, device, batch_size: int, macro: bool = False) \
            -> Dict[str, float]:
        """
        Evaluate model on this dataset, returning the task-relevant metrics.
        """
        pass
