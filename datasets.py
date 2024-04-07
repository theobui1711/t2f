import bisect
import copy
import os
import logging
import json
from itertools import islice
from collections import Counter, defaultdict
# from cmd2 import categorize
# from more_itertools import ichunked
import numpy as np
import random
import networkx as nx
# from typing import DefaultDict, Dict, List, Tuple, Set
from typing import List, Dict
# from sympy import EX
import torch
from transformers import PreTrainedTokenizer
import string
import re
from shapely.geometry import box
from shapely.ops import unary_union

from arguments import DataTrainingArguments
from input_example import InputFeatures, EntityType, RelationType, Entity, Relation, Intent, InputExample, \
    CorefDocument, Room
from base_dataset import BaseDataset
from utils import get_precision_recall_f1, calculate_intersection, calculate_union, render_image, \
    process_boundary_info, calculate_iou, render_image_

from coreference_metrics import CorefAllMetrics
from input_formats import INPUT_FORMATS
from output_formats import OUTPUT_FORMATS

DATASETS = {}


def register_dataset(dataset_class):
    DATASETS[dataset_class.name] = dataset_class
    return dataset_class


def load_dataset(
        dataset_name: str,
        data_args: DataTrainingArguments,
        tokenizer: PreTrainedTokenizer,
        split: str,
        max_input_length: int,
        max_output_length: int,
        train_subset: float = 1,
        seed: int = None,
        shuffle: bool = True,
        is_eval: bool = False
):
    """
    Load a registered dataset.
    """
    return DATASETS[dataset_name](
        tokenizer=tokenizer,
        max_input_length=max_input_length,
        max_output_length=max_output_length,
        mode=split,
        overwrite_cache=data_args.overwrite_cache,
        train_subset=train_subset,
        seed=seed,
        shuffle=shuffle,
        data_args=data_args,
        is_eval=is_eval,
    )


def generate_all(rm):
    request_template = [
        f"Make {rm.type} ",
        f"The {rm.type} should be ",
        "It would be " + random.choice(["great", "good"]) + f" to have {rm.type} ",
        f"I would like to have {rm.type} ",
        f"Can you make {rm.type} ",
        f"Can we have {rm.type} to be "
    ]
    request_idx = random.choice(range(len(request_template)))
    approx_phrase = ["about", "around", "approx"]
    approx_phrase = random.choice(approx_phrase)
    round_room_size = str(rm.size)
    round_room_ratio = str(rm.aspect_ratio)
    eos = "?" if request_idx > 3 else "."

    if rm.location is not None:
        desc = request_template[request_idx] + \
               "at {}".format(rm.location) + \
               f" with {approx_phrase} {round_room_size} sqft and the aspect ratio of {round_room_ratio}" + eos
    else:
        desc = request_template[request_idx] + \
               f" with {approx_phrase} {round_room_size} sqft and the aspect ratio of {round_room_ratio}" + eos
    return desc


def generate_size_ar(rm):
    request_template = [
        f"Make {rm.type} ",
        f"The {rm.type} should be ",
        "It would be " + random.choice(["great", "good"]) + f" to have {rm.type} ",
        f"I would like to have {rm.type} ",
        f"Can you make {rm.type} ",
        f"Can we have {rm.type} to be "
    ]
    request_idx = random.choice(range(len(request_template)))
    approx_phrase = ["about", "around", "approx"]
    round_room_size = str(rm.size)
    round_room_ratio = str(rm.aspect_ratio)
    eos = "?" if request_idx > 3 else "."

    desc = request_template[request_idx] + \
           random.choice(approx_phrase) + \
           f" {round_room_size} sqft with the aspect ratio of {round_room_ratio}" + eos
    return desc


def generate_ar(rm):
    request_template = [
        f"Make the aspect ratio of {rm.type} ",
        f"The aspect ratio of {rm.type} should be ",
        "It would be " + random.choice(["great", "good"]) + f" to have the aspect ratio of {rm.type} ",
        f"I would like to have the aspect ratio of {rm.type} ",
        f"Can you make the aspect ratio of {rm.type} ",
        f"Can we have the aspect ratio of {rm.type} to be "
    ]
    request_idx = random.choice(range(len(request_template)))
    approx_phrase = ["about", "around", "approx"]
    round_room_ratio = str(rm.aspect_ratio)
    eos = "?" if request_idx > 3 else "."

    desc = request_template[request_idx] + \
           random.choice(approx_phrase) + \
           f" {round_room_ratio}" + eos
    return desc


def generate_size(rm):
    request_template = [
        f"Make {rm.type} ",
        f"The {rm.type} should be ",
        "It would be " + random.choice(["great", "good"]) + f" to have {rm.type} ",
        f"I would like to have {rm.type} ",
        f"Can you make {rm.type} ",
        f"Can we have {rm.type} to be "
    ]
    request_idx = random.choice(range(len(request_template)))
    approx_phrase = ["about", "around", "approx"]
    round_room_size = str(rm.size)
    eos = "?" if request_idx > 3 else "."

    desc = request_template[request_idx] + \
           random.choice(approx_phrase) + \
           f" {round_room_size} sqft" + eos
    return desc


def generate_location(rm):
    location_template = [
        "Place {} at the {} of the apartment.",
        "The {} should be at the {} of the apartment.",
        "I would like to place {} at the {} of the apartment.",
        "It would be " + random.choice(["great", "good"]) + " to place {} at the {} of the apartment.",
        "Can we have {} to be at {}. "
    ]
    if rm.type.startswith('living'):
        return None
    else:
        loc_des = random.choice(location_template).format(rm.type, rm.location)
        return loc_des


# TODO: Only consider near or next to
def generate_relation(rm):
    relation_template = [
        "The {} should be next to the {}.",
        "The {} connects to the {}.",
        "The {} attaches to the {}.",
    ]
    if rm.type.startswith('living'):
        return None
    else:
        if rm.relation:
            # # Select only one nearby room
            # loc_des = random.choice(relation_template).format(rm.type, random.choice(rm.relation))
            # Select all nearby rooms
            loc_des = random.choice(relation_template).format(rm.type, ", ".join(rm.relation))
        else:
            loc_des = ""
        return loc_des


def generate_private(rm):
    if rm.private == 'None':
        return None
    else:
        if rm.type.startswith('master') or rm.type.startswith('common'):
            return (f'The {rm.type} should have an en-suite bathroom.')
        elif rm.type.startswith('bathroom'):
            if rm.private == 'True':
                strings = [f'The {rm.type} is private.',
                           f'The {rm.type} is in an en-suite bathroom.']
                return (random.choice(strings))
            elif rm.private == 'False':
                strings = [
                    # f'The {room_type} is shared.',
                    f'The {rm.type} can be used by guest.',
                    # f'The {rm.type} can be directly accessed from living room.'
                ]
                return (random.choice(strings))
            else:
                raise Exception('Error1')
        elif rm.type.startswith('balcony'):
            if rm.private == 'True':
                return (f'The {rm.type} is private.')
            elif rm.private == 'False':
                # return (f'The {rm.type} can be directly accessed from living room.')
                return ""
            else:
                raise Exception('Error2')
        else:
            raise Exception('Error3')


def generate_rt(rm):
    request_template = [
        f"Make a {rm.type} ",
        f"The {rm.type} should be considered",
        "It would be " + random.choice(["great", "good"]) + f" to have a {rm.type} ",
        f"I would like to have a {rm.type} ",
        f"Can you make a {rm.type} ",
        f"Can we have a {rm.type} "
    ]
    request_idx = random.choice(range(len(request_template)))
    eos = "?" if request_idx > 3 else "."

    desc = request_template[request_idx] + eos
    return desc


def miss_attributes(self, rooms_attributes):
    if self.data_args.exp == 'miss1attri_0':  # miss aspect ratio
        strings = ''
        editing_rooms = []
        for room in rooms_attributes:
            editing_room = defaultdict()
            editing_room['room_type'] = room
            editing_parts = ['aspect_ratio']
            editing_room['editing_parts'] = editing_parts
            editing_rooms.append(editing_room)

            strings += (rooms_attributes[room]['size'] + ' ')
            if rooms_attributes[room]['location']:
                strings += (rooms_attributes[room]['location'] + ' ')
            if rooms_attributes[room]['private']:
                strings += (rooms_attributes[room]['private'])
        tokens = re.findall(r"[\w']+|[.,!?;]", strings)
    elif self.data_args.exp == 'miss1attri_1':  # miss size
        strings = ''
        editing_rooms = []
        for room in rooms_attributes:
            editing_room = defaultdict()
            editing_room['room_type'] = room
            editing_parts = ['size']
            editing_room['editing_parts'] = editing_parts
            editing_rooms.append(editing_room)

            strings += (rooms_attributes[room]['aspect ratio'] + ' ')
            if rooms_attributes[room]['location']:
                strings += (rooms_attributes[room]['location'] + ' ')
            if rooms_attributes[room]['private']:
                strings += (rooms_attributes[room]['private'])
        tokens = re.findall(r"[\w']+|[.,!?;]", strings)
    elif self.data_args.exp == 'miss1attri_2':  # miss location
        strings = ''
        editing_rooms = []
        for room in rooms_attributes:
            editing_room = defaultdict()
            editing_room['room_type'] = room
            editing_parts = ['location']
            editing_room['editing_parts'] = editing_parts
            editing_rooms.append(editing_room)

            strings += (rooms_attributes[room]['size+aspect ratio'] + ' ')
            if rooms_attributes[room]['private']:
                strings += (rooms_attributes[room]['private'])
        tokens = re.findall(r"[\w']+|[.,!?;]", strings)
    elif self.data_args.exp == 'miss1room':  # miss one room's all attributes
        strings = ''
        drop_idx = random.choice(range(len(rooms_attributes)))
        editing_rooms = []
        for idx, room in enumerate(rooms_attributes):
            editing_room = defaultdict()
            if int(idx) == int(drop_idx):
                strings += (rooms_attributes[room]['room_type'] + ' ')
                editing_room['room_type'] = room
                editing_parts = ['location', 'size', 'aspect_ratio']
                editing_room['editing_parts'] = editing_parts
                editing_rooms.append(editing_room)
            else:
                strings += (rooms_attributes[room]['size+aspect ratio'] + ' ')
                if rooms_attributes[room]['location']:
                    strings += (rooms_attributes[room]['location'] + ' ')
                if rooms_attributes[room]['private']:
                    strings += (rooms_attributes[room]['private'])
        tokens = re.findall(r"[\w']+|[.,!?;]", strings)
    elif self.data_args.exp == 'missrandom':  # random miss one attributes for each room
        strings = ''
        editing_rooms = []
        dropping_attri = ['size', 'aspect_ratio', 'location']
        for room in rooms_attributes:
            # 0:size 1:aspect ratio 2:location
            if room.startswith('living'):  # living room does not have location attribute
                drop_idx = random.choice(range(2))
                editing_room = defaultdict()
                editing_room['room_type'] = room
                editing_parts = [dropping_attri[drop_idx]]
                editing_room['editing_parts'] = editing_parts
                editing_rooms.append(editing_room)

                if drop_idx == 0:  # drop size
                    strings += (rooms_attributes[room]['aspect ratio'] + ' ')
                elif drop_idx == 1:  # drop aspect
                    strings += (rooms_attributes[room]['size'] + ' ')
                else:
                    raise Exception('randommiss error 1')
            else:
                drop_idx = random.choice(range(3))
                editing_room = defaultdict()
                editing_room['room_type'] = room
                editing_parts = [dropping_attri[drop_idx]]
                editing_room['editing_parts'] = editing_parts
                editing_rooms.append(editing_room)

                if drop_idx == 0:  # drop size
                    strings += (rooms_attributes[room]['aspect ratio'] + ' ')
                    if rooms_attributes[room]['location']:
                        strings += (rooms_attributes[room]['location'] + ' ')
                    if rooms_attributes[room]['private']:
                        strings += (rooms_attributes[room]['private'])
                elif drop_idx == 1:  # drop aspect ratio
                    strings += (rooms_attributes[room]['size'] + ' ')
                    if rooms_attributes[room]['location']:
                        strings += (rooms_attributes[room]['location'] + ' ')
                    if rooms_attributes[room]['private']:
                        strings += (rooms_attributes[room]['private'])
                elif drop_idx == 2:  # drop location
                    strings += (rooms_attributes[room]['size+aspect ratio'] + ' ')
                    if rooms_attributes[room]['private']:
                        strings += (rooms_attributes[room]['private'])
                else:
                    raise Exception('randommiss error 2')
        tokens = re.findall(r"[\w']+|[.,!?;]", strings)
    else:
        raise Exception(
            'wrong exp name about missing attributes experiments! miss1attri_0/miss1attri_1/miss1attri_2/miss1room/missrandom')
    return tokens, editing_rooms


def generate_editing_data(self, example, output_sentence):
    gt_rooms = []
    for room in example.rooms:
        r = defaultdict()
        r['room_type'] = room.type
        r['x'] = room.x
        r['y'] = room.y
        r['h'] = room.h
        r['w'] = room.w
        r['location'] = room.location
        r['size'] = room.size
        r['aspect_ratio'] = room.aspect_ratio
        r['private'] = room.private
        gt_rooms.append(r)

    boundary = example.boundary

    boundary_boxes = []
    num_boxes = len(example.boundary_tokens) / 5
    assert (num_boxes % 1) == 0
    for i in range(int(num_boxes)):
        start_index = 0 + i * 5
        single_box_token = example.boundary_tokens[start_index: start_index + 5]
        single_box = defaultdict()
        single_box['room_type'] = single_box_token[0]
        single_box['x'] = single_box_token[1]
        single_box['y'] = single_box_token[2]
        single_box['h'] = single_box_token[3]
        single_box['w'] = single_box_token[4]
        boundary_boxes.append(single_box)

    # generate predicted rooms
    res = self.output_format.run_inference(
        example,
        output_sentence
    )
    predicted_rooms_by_name, predicted_rooms, raw_predicted_relations, wrong_reconstruction, format_error, label_error = res

    predicted_attri = defaultdict()
    for attribute_tuple in raw_predicted_relations:
        attribute_type, value, room_tuple, room_type = attribute_tuple
        if room_type not in predicted_attri:
            predicted_attri[room_type] = defaultdict()
            # Set invalid value to 128
        try:
            value = int(value)
        except:
            value = 128
        predicted_attri[room_type][attribute_type] = value

    correct_attributes = ['x coordinate', 'y coordinate', 'height', 'width']
    wrong_room = []
    for room_type in predicted_attri:
        if set(list(predicted_attri[room_type].keys())) != set(correct_attributes):
            print('wrong output format:')
            print(predicted_attri[room_type])
            wrong_room.append(room_type)
    for wrong_r in wrong_room:
        predicted_attri.pop(wrong_r)

    generated_rooms = []
    for room_type in predicted_attri:
        generated_room = defaultdict()
        generated_room['room_type'] = room_type
        generated_room['x'] = predicted_attri[room_type]['x coordinate']
        generated_room['y'] = predicted_attri[room_type]['y coordinate']
        generated_room['height'] = predicted_attri[room_type]['height']
        generated_room['width'] = predicted_attri[room_type]['width']
        generated_rooms.append(generated_room)

    # editing rooms' information
    editing_rooms = example.editing_rooms

    editing_instance = defaultdict()
    editing_instance['generated_rooms'] = generated_rooms
    editing_instance['gt_rooms'] = gt_rooms
    editing_instance['boundary'] = boundary
    editing_instance['boundary_boxes'] = boundary_boxes
    editing_instance['editing_rooms'] = editing_rooms

    return editing_instance


class JointERDataset(BaseDataset):
    """
    Base class for datasets of joint entity and relation extraction.
    """
    entity_types = None
    relation_types = None

    natural_entity_types = None  # dictionary from entity types given in the dataset to the natural strings to use
    natural_relation_types = None  # dictionary from relation types given in the dataset to the natural strings to use

    default_output_format = 'joint_er'

    def load_cached_data(self, cached_features_file):
        d = torch.load(cached_features_file)
        self.entity_types, self.relation_types, self.examples, self.features = \
            d['entity_types'], d['relation_types'], d['examples'], d['features']

    def save_data(self, cached_features_file):
        torch.save({
            'entity_types': self.entity_types,
            'relation_types': self.relation_types,
            'examples': self.examples,
            'features': self.features,
        }, cached_features_file)

    def load_schema(self):
        """
        Load entity and relation types.

        This is the default implementation which uses the dictionaries natural_entity_types and natural_relation_types.
        """
        if self.natural_entity_types is not None:
            self.entity_types = {short: EntityType(
                short=short,
                natural=natural,
            ) for short, natural in self.natural_entity_types.items()}

        if self.natural_relation_types is not None:
            self.relation_types = {short: RelationType(
                short=short,
                natural=natural,
            ) for short, natural in self.natural_relation_types.items()}

    def load_data_single_split(self, split: str, seed: int = None) -> List[InputExample]:
        """
        Load data for a single split (train, dev, or test).

        This is the default implementation for datasets in the SpERT format
        (see https://github.com/markus-eberts/spert).
        """
        examples = []
        name = self.name if self.data_name is None else self.data_name
        file_path = os.path.join(self.data_dir(), f'{name}_{split}.json')

        with open(file_path, 'r') as f:
            data = json.load(f)
            logging.info(f"Loaded {len(data)} sentences for split {split} of {self.name}")

            for i, x in enumerate(data):
                entities = [
                    Entity(id=j, type=self.entity_types[y['type']], start=y['start'], end=y['end'])
                    for j, y in enumerate(x['entities'])
                ]

                relations = [
                    Relation(
                        type=self.relation_types[y['type']], head=entities[y['head']], tail=entities[y['tail']]
                    )
                    for y in x['relations']
                ]

                tokens = x['tokens']

                example = InputExample(
                    id=f'{split}-{i}',
                    tokens=tokens,
                    entities=entities,
                    relations=relations,
                )

                examples.append(example)

        return examples

    def evaluate_example(self, example: InputExample, output_sentence: str, model=None, tokenizer=None) -> Counter:
        """
        Evaluate an output sentence on a single example of this dataset.
        """
        # extract entities and relations from output sentence
        res = self.output_format.run_inference(
            example,
            output_sentence,
            entity_types=self.entity_types,
            relation_types=self.relation_types,
        )
        predicted_entities, predicted_relations = res[:2]
        if len(res) == 6:
            # the output format provides information about errors
            wrong_reconstruction, label_error, entity_error, format_error = res[2:]
        else:
            # in case the output format does not provide information about errors
            wrong_reconstruction = label_error = entity_error = format_error = False

        predicted_entities_no_type = set([entity[1:] for entity in predicted_entities])

        # load ground truth entities
        gt_entities = set(entity.to_tuple() for entity in example.entities)
        gt_entities_no_type = set([entity[1:] for entity in gt_entities])

        # compute correct entities
        correct_entities = predicted_entities & gt_entities
        correct_entities_no_type = gt_entities_no_type & predicted_entities_no_type

        # load ground truth relations
        gt_relations = set(relation.to_tuple() for relation in example.relations)

        # compute correct relations
        correct_relations = predicted_relations & gt_relations

        assert len(correct_entities) <= len(predicted_entities)
        assert len(correct_entities) <= len(gt_entities)
        assert len(correct_entities_no_type) <= len(predicted_entities_no_type)
        assert len(correct_entities_no_type) <= len(gt_entities_no_type)

        assert len(correct_relations) <= len(predicted_relations)
        assert len(correct_relations) <= len(gt_relations)

        res = Counter({
            'num_sentences': 1,
            'wrong_reconstructions': 1 if wrong_reconstruction else 0,
            'label_error': 1 if label_error else 0,
            'entity_error': 1 if entity_error else 0,
            'format_error': 1 if format_error else 0,
            'gt_entities': len(gt_entities),
            'predicted_entities': len(predicted_entities),
            'correct_entities': len(correct_entities),
            'gt_entities_no_type': len(gt_entities_no_type),
            'predicted_entities_no_type': len(predicted_entities_no_type),
            'correct_entities_no_type': len(correct_entities_no_type),
            'gt_relations': len(gt_relations),
            'predicted_relations': len(predicted_relations),
            'correct_relations': len(correct_relations),
        })

        # add information about each entity/relation type so that we can compute the macro-F1 scores
        if self.entity_types is not None:
            for entity_type in self.entity_types.values():
                predicted = set(entity for entity in predicted_entities if entity[0] == entity_type.natural)
                gt = set(entity for entity in gt_entities if entity[0] == entity_type.natural)
                correct = predicted & gt
                res['predicted_entities', entity_type.natural] = len(predicted)
                res['gt_entities', entity_type.natural] = len(gt)
                res['correct_entities', entity_type.natural] = len(correct)

        if self.relation_types is not None:
            for relation_type in self.relation_types.values():
                predicted = set(relation for relation in predicted_relations if relation[0] == relation_type.natural)
                gt = set(relation for relation in gt_relations if relation[0] == relation_type.natural)
                correct = predicted & gt
                res['predicted_relations', relation_type.natural] = len(predicted)
                res['gt_relations', relation_type.natural] = len(gt)
                res['correct_relations', relation_type.natural] = len(correct)

        return res

    def evaluate_dataset(self, data_args: DataTrainingArguments, model, device, batch_size: int, macro: bool = False) \
            -> Dict[str, float]:
        """
        Evaluate model on this dataset.
        """
        results = Counter()

        for example, output_sentence in self.generate_output_sentences(data_args, model, device, batch_size):
            new_result = self.evaluate_example(
                example=example,
                output_sentence=output_sentence,
                model=model,
                tokenizer=self.tokenizer,
            )
            results += new_result

        entity_precision, entity_recall, entity_f1 = get_precision_recall_f1(
            num_correct=results['correct_entities'],
            num_predicted=results['predicted_entities'],
            num_gt=results['gt_entities'],
        )

        entity_precision_no_type, entity_recall_no_type, entity_f1_no_type = get_precision_recall_f1(
            num_correct=results['correct_entities_no_type'],
            num_predicted=results['predicted_entities_no_type'],
            num_gt=results['gt_entities_no_type'],
        )

        entity_precision_by_type = []
        entity_recall_by_type = []
        entity_f1_by_type = []

        if macro:
            # compute also entity macro scores
            for entity_type in self.entity_types.values():
                precision, recall, f1 = get_precision_recall_f1(
                    num_correct=results['correct_entities', entity_type.natural],
                    num_predicted=results['predicted_entities', entity_type.natural],
                    num_gt=results['gt_entities', entity_type.natural],
                )
                entity_precision_by_type.append(precision)
                entity_recall_by_type.append(recall)
                entity_f1_by_type.append(f1)

        relation_precision, relation_recall, relation_f1 = get_precision_recall_f1(
            num_correct=results['correct_relations'],
            num_predicted=results['predicted_relations'],
            num_gt=results['gt_relations'],
        )

        res = {
            'wrong_reconstruction': results['wrong_reconstructions'] / results['num_sentences'],
            'label_error': results['label_error'] / results['num_sentences'],
            'entity_error': results['entity_error'] / results['num_sentences'],
            'format_error': results['format_error'] / results['num_sentences'],
            'entity_precision': entity_precision,
            'entity_recall': entity_recall,
            'entity_f1': entity_f1,
            'relation_precision': relation_precision,
            'relation_recall': relation_recall,
            'relation_f1': relation_f1,
            'entity_precision_no_type': entity_precision_no_type,
            'entity_recall_no_type': entity_recall_no_type,
            'entity_f1_no_type': entity_f1_no_type,
        }

        if macro:
            res.update({
                'entity_macro_precision': np.mean(np.array(entity_precision_by_type)),
                'entity_macro_recall': np.mean(np.array(entity_recall_by_type)),
                'entity_macro_f1': np.mean(np.array(entity_f1_by_type)),
            })

        return res


@register_dataset
class FloorplanDataset(JointERDataset):
    name = 'floorplan'
    default_output_format = 'floorplan'

    def load_data_single_split(self, split: str, seed: int = None) -> List[InputExample]:
        """
        Load data for a single split (train, dev, or test).

        """
        examples = []
        name = self.name if self.data_name is None else self.data_name
        file_path = os.path.join(self.data_dir(), f'{name}_{split}.json')

        with open(file_path, 'r') as f:
            data = json.load(f)
            logging.info(f"Loaded {len(data)} sentences for split {split} of {self.name}")

            apartment_text = defaultdict()
            for i, description in enumerate(data):

                # TODO: read current data scheme
                rooms = [Room(type=room['room_type'], x=room['x'], y=room['y'], h=room['h'], w=room['w'],
                              x_min=room['x_min'], y_min=room['y_min'], x_max=room['x_max'], y_max=room['y_max'],
                              near_x_min=room['near_x_min'], near_y_min=room['near_y_min'],
                              near_x_max=room['near_x_max'], near_y_max=room['near_y_max'],
                              relation=room['relation'], location=room['location'], size=str(room['size']),
                              aspect_ratio=room['aspect ratio'], private=str(room['private'])) for j, room in
                         enumerate(description['rooms'])]

                # generate text
                # living room should only have size and aspect ratio description
                # balcony, bathroom, and bedroom has "private" value True/False. Others None.
                rooms_attributes = defaultdict()
                for rm in rooms:
                    rooms_attributes[rm.type] = {'location': None, 'size': None, 'aspect ratio': None,
                                                 'size+aspect ratio': None, 'private': None, 'room_type': None}
                    # generate location, size, aspect ratio
                    all_descp = generate_all(rm)
                    rooms_attributes[rm.type]['all'] = all_descp
                    # generate size+aspect ratio text
                    size_as_descp = generate_size_ar(rm)
                    rooms_attributes[rm.type]['size+aspect ratio'] = size_as_descp
                    # generate aspect ratio text
                    as_descp = generate_ar(rm)
                    rooms_attributes[rm.type]['aspect ratio'] = as_descp
                    # generate size text
                    size_descp = generate_size(rm)
                    rooms_attributes[rm.type]['size'] = size_descp
                    # generate location text
                    location_descp = generate_location(rm)
                    rooms_attributes[rm.type]['location'] = location_descp
                    # generate relation text
                    relation_descp = generate_relation(rm)
                    rooms_attributes[rm.type]['relation'] = relation_descp
                    # generate whether private text
                    private_descp = generate_private(rm)
                    rooms_attributes[rm.type]['private'] = private_descp
                    # generate room type text
                    rt_descp = generate_rt(rm)
                    rooms_attributes[rm.type]['room_type'] = rt_descp

                # mising attribute study
                editing_rooms = []
                if self.data_args.exp.startswith('miss'):
                    tokens, editing_rooms = miss_attributes(self, rooms_attributes)
                if self.data_args.exp.endswith('finetune'):
                    strings = description['annotated_strings']
                    tokens = re.findall(r"[\w']+|[.,!?;]", strings)
                else:  # full attributes
                    # for evaluate 1st stage model on human data
                    if 'annotated_strings' in description:
                        strings = description['annotated_strings']
                        tokens = re.findall(r"[\w']+|[.,!?;]", strings)
                    else:
                        strings = ''
                        for room in rooms_attributes:
                            # # ZY version
                            # strings += (rooms_attributes[room]['all']+' ')
                            # # with relation
                            # if rooms_attributes[room]['relation']:
                            #     strings += (rooms_attributes[room]['relation']+' ')
                            # baseline version
                            if rooms_attributes[room]['location']:
                                strings += (rooms_attributes[room]['location'] + ' ')
                            if rooms_attributes[room]['size+aspect ratio']:
                                strings += (rooms_attributes[room]['size+aspect ratio'] + ' ')
                            # TODO: Private still necessary?
                            if rooms_attributes[room]['private']:
                                strings += (rooms_attributes[room]['private'] + ' ')
                        tokens = re.findall(r"[\w']+|[.,!?;]", strings)
                        apartment_text[f'{split}-{i}'] = strings

                # with no text
                if self.data_args.exp == 'no_text':
                    tokens = []

                boundary = description['boundary']

                boundary_tokens = description['boundary_boxs']
                boundary_tokens = process_boundary_info(boundary_tokens)

                # random shuffle the room order in target sequence
                random.shuffle(rooms)

                if 'img_id' in description:
                    example = InputExample(
                        id=f'{split}-{i}',
                        tokens=tokens,
                        rooms=rooms,
                        boundary=boundary,
                        boundary_tokens=boundary_tokens,
                        editing_rooms=editing_rooms,
                        image_id=description['img_id']
                    )
                else:
                    example = InputExample(
                        id=f'{split}-{i}',
                        tokens=tokens,
                        rooms=rooms,
                        boundary=boundary,
                        boundary_tokens=boundary_tokens,
                        editing_rooms=editing_rooms
                    )

                examples.append(example)

        try:
            os.mkdir(f'experiments/{self.data_args.exp}/')
        except FileExistsError:
            pass
        with open(f'experiments/{self.data_args.exp}/{split}_description.json', 'w', encoding='utf-8') as f:
            json.dump(apartment_text, f)

        return examples

    def evaluate_example(self, example: InputExample, output_sentence: str, model=None, tokenizer=None, output_dir=None,
                         prediction=None) -> Counter:
        """
        Evaluate an output sentence on a single example of this dataset.
        """
        # extract entities and relations from output sentence
        res = self.output_format.run_inference(
            example,
            output_sentence,
            prediction
        )
        predicted_rooms_by_name, predicted_rooms, raw_predicted_relations, wrong_reconstruction, format_error, label_error = res

        # calculate IoU
        gt_boxes = defaultdict()
        for room in example.rooms:
            gt_x, gt_y, gt_h, gt_w = room.x, room.y, room.h, room.w
            gt_box = [
                [int(gt_x - gt_h / 2), int(gt_y - gt_w / 2)], [int(gt_x + gt_h / 2), int(gt_y - gt_w / 2)],
                [int(gt_x - gt_h / 2), int(gt_y + gt_w / 2)], [int(gt_x + gt_h / 2), int(gt_y + gt_w / 2)]]
            gt_boxes[room.type] = gt_box

        predicted_attri = defaultdict()
        for attribute_tuple in raw_predicted_relations:
            attribute_type, value, room_tuple, room_type = attribute_tuple
            if room_type not in predicted_attri:
                predicted_attri[room_type] = defaultdict()
                # Set invalid value to 128
            try:
                value = int(value)
            except:
                value = 128
            predicted_attri[room_type][attribute_type] = value

        # TODO: examien the predicted_attri patterns
        correct_attributes = ['x coordinate', 'y coordinate', 'height', 'width']
        wrong_room = []
        for room_type in predicted_attri:
            if set(list(predicted_attri[room_type].keys())) != set(correct_attributes):
                print('wrong output format:')
                print(predicted_attri[room_type])
                wrong_room.append(room_type)
        for wrong_r in wrong_room:
            predicted_attri.pop(wrong_r)

        all_gt_rooms = [room.type for room in example.rooms]
        predicted_boxes = defaultdict()
        for room in predicted_attri:
            predicted_boxes[room] = [
                [int(float(predicted_attri[room]['x coordinate']) - float(predicted_attri[room]['height']) / 2),
                 int(float(predicted_attri[room]['y coordinate']) - float(predicted_attri[room]['width']) / 2)],
                [int(float(predicted_attri[room]['x coordinate']) + float(predicted_attri[room]['height']) / 2),
                 int(float(predicted_attri[room]['y coordinate']) - float(predicted_attri[room]['width']) / 2)],
                [int(float(predicted_attri[room]['x coordinate']) - float(predicted_attri[room]['height']) / 2),
                 int(float(predicted_attri[room]['y coordinate']) + float(predicted_attri[room]['width']) / 2)],
                [int(float(predicted_attri[room]['x coordinate']) + float(predicted_attri[room]['height']) / 2),
                 int(float(predicted_attri[room]['y coordinate']) + float(predicted_attri[room]['width']) / 2)]
            ]
        for room in gt_boxes:
            ymin = gt_boxes[room][0][1]
            xmin = gt_boxes[room][0][0]
            ymax = gt_boxes[room][3][1]
            xmax = gt_boxes[room][3][0]
            gt_boxes[room] = (ymin, xmin, ymax, xmax)

        for room in predicted_boxes:
            ymin = predicted_boxes[room][0][1]
            xmin = predicted_boxes[room][0][0]
            ymax = predicted_boxes[room][3][1]
            xmax = predicted_boxes[room][3][0]
            predicted_boxes[room] = (ymin, xmin, ymax, xmax)

        # render images for gt and predicted boxes for preview
        render_image_(example, predicted_boxes, all_gt_rooms, gt_boxes, output_dir)

        # macro_average_iou, micro_average_iou = calculate_iou(gt_boxes, predicted_boxes)

        # average_iou = (macro_average_iou+micro_average_iou)/2

        # res = Counter({
        #     'num_sentences': 1,
        #     'wrong_reconstructions': 1 if wrong_reconstruction else 0,
        #     'label_error': 1 if label_error else 0,
        #     'format_error': 1 if format_error else 0,
        #     'gt_rooms': len(example.rooms),
        #     'predicted_rooms': len(predicted_rooms),
        #     'macro_average_iou': macro_average_iou,
        #     'micro_average_iou' : micro_average_iou
        # })

        # return res, average_iou
        return

    def evaluate_dataset(self, data_args: DataTrainingArguments, model, device, batch_size: int, output_dir: str,
                         macro: bool = False) -> Dict[str, float]:
        """
        Evaluate model on this dataset.
        """
        results = Counter()
        try:
            os.mkdir(f'{output_dir}output_images/')
        except FileExistsError:
            pass

        editing_instances = []
        iou = defaultdict()
        for example, output_sentence, predicted_index in self.generate_output_sentences(data_args, model, device,
                                                                                        batch_size, self.features):
            self.evaluate_example(
                example=example,
                output_sentence=output_sentence,
                model=model,
                tokenizer=self.tokenizer,
                output_dir=output_dir,
                prediction=predicted_index
            )
        return "Rendering Done!"
        #     new_result, average_iou = self.evaluate_example(
        #             example=example,
        #             output_sentence=output_sentence,
        #             model=model,
        #             tokenizer=self.tokenizer,
        #             output_dir=output_dir,
        #             prediction = predicted_index
        #         )
        #     results+=new_result
        #     iou[example.id] = average_iou

        #     #TODO: store data for Editing model
        #     if data_args.editing_data == True:
        #         editing_instance = generate_editing_data(self, example, output_sentence)
        #         editing_instances.append(editing_instance)

        # sorted_iou = {k: v for k, v in sorted(iou.items(), key=lambda item: item[1])}
        # with open(f'{output_dir}sorted_iou.json', 'w', encoding='utf-8') as f:
        #         json.dump(sorted_iou, f)

        # if data_args.editing_data == True:
        #     #TODO: save editing data
        #     with open(f'./data/floorplan/{data_args.exp}_editing.json', 'w', encoding='utf-8') as f:
        #         json.dump(editing_instances, f)

        # results['macro_average_iou'] = results['macro_average_iou']/results['num_sentences']
        # results['micro_average_iou'] = results['micro_average_iou']/results['num_sentences']

        # return results
