# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
from typing import Tuple, List, Dict
from shapely.geometry import Polygon
import cv2
import numpy as np
from collections import Counter, defaultdict
from shapely.geometry import box
from shapely.ops import unary_union
import itertools
import cv2
from collections import defaultdict
from input_example import InputExample, Room
from typing import Tuple

room_idx = {'living room': 0, 'living room 1': 0, 'living room 2': 0, 'master room': 1, 'kitchen': 2, 'bathroom 1': 3,
            'bathroom 2': 3, 'bathroom 3': 3, 'dining room': 4, 'common room 2': 5, 'common room 3': 5,
            'common room 1': 5, 'common room 4': 5, 'balcony 1': 9, 'balcony 2': 9, 'balcony 3': 9, 'entrance': 10,
            'storage': 11, 'bathroom': 3, 'balcony': 9, 'common room': 5,
            'master room 2': 1, 'master room 1': 1, 'master room 4': 1, 'kitchen 1': 2, 'kitchen 2': 2,
            'master room 3': 1, 'storage 1': 11, 'storage 2': 11, 'storage 3': 11}

color_idx = {15: (255, 165, 0), 16: (255, 165, 0), 17: (240, 128, 128), 18: (240, 128, 128), 0: (170, 232, 238),
             1: (0, 165, 255), 5: (0, 215, 255), 6: (255, 215, 0), 7: (255, 215, 0), 8: (255, 215, 0),
             2: (128, 128, 240), 3: (230, 216, 173), 4: (218, 112, 214), 9: (35, 142, 107), 10: (255, 255, 0),
             11: (221, 160, 221), 12: (173, 216, 230), 13: (107, 142, 35), 14: (255, 215, 0), 19: (255, 165, 0)}


def render_image_(example, predicted_boxes, all_gt_rooms, gt_boxes, output_dir):
    image_height = 256
    image_width = 256
    number_of_color_channels = 3
    background_color = (255, 255, 255)
    gt_image = np.full((image_height, image_width, number_of_color_channels), background_color, dtype=np.uint8)
    predicted_image = np.full((image_height, image_width, number_of_color_channels), background_color, dtype=np.uint8)
    boundary_color = [0, 0, 0]

    left_boundary = int(example.boundary_tokens[1])

    living = defaultdict()
    common = defaultdict()
    master = defaultdict()
    balcony = defaultdict()
    bathroom = defaultdict()
    kitchen = defaultdict()
    storage = defaultdict()
    dining = defaultdict()
    for room in predicted_boxes:
        if room.startswith('living'):
            living[room] = predicted_boxes[room]
        elif room.startswith('common'):
            common[room] = predicted_boxes[room]
        elif room.startswith('master'):
            master[room] = predicted_boxes[room]
        elif room.startswith('balcony'):
            balcony[room] = predicted_boxes[room]
        elif room.startswith('bathroom'):
            bathroom[room] = predicted_boxes[room]
        elif room.startswith('kitchen'):
            kitchen[room] = predicted_boxes[room]
        elif room.startswith('storage'):
            storage[room] = predicted_boxes[room]
        elif room.startswith('dining'):
            dining[room] = predicted_boxes[room]
    room_type_list = [living, common, master, balcony, bathroom, kitchen, storage, dining]
    for room_type in room_type_list:
        for room in room_type:
            left_top_pr = (room_type[room][0], room_type[room][1])
            right_bt_pr = (room_type[room][2], room_type[room][3])

            color = color_idx[room_idx[room]]
            # draw room on predicted image
            cv2.rectangle(predicted_image, left_top_pr, right_bt_pr, color, -1)

    for boundary_pixel in example.boundary:
        predicted_image[boundary_pixel[0], boundary_pixel[1]] = boundary_color

    # cv2.putText(predicted_image,"Seq2Seq",(10,10),0,0.3,boundary_color)
    # cv2.imwrite(f'./image_{example.image_id}.png', predicted_image)
    cv2.imwrite(f'{output_dir}draw/{example.image_id}.png', predicted_image)


def process_boundary_info(boundary_boxes):
    boundary_tokens = []
    for box in boundary_boxes:
        # boundary_t = ['boundary','box','central','point','coordinate','=','height','=','width','=']
        boundary_t = []
        box_type = box["room_type"]
        box_x = str(box['x'])
        box_y = str(box['y'])
        box_h = str(box['h'])
        box_w = str(box['w'])
        box_x_min = str(box['x_min'])
        box_y_min = str(box['y_min'])
        box_x_max = str(box['x_max'])
        box_y_max = str(box['y_max'])
        if box_type == 'positive':
            boundary_t.insert(0, '+')
        elif box_type == 'negative':
            boundary_t.insert(0, '-')
        else:
            raise Exception("the field room_type in boundary_boxs should be positive or negative!")
        # # using x y h w to represent boundary boxes
        # boundary_t.append(box_x)
        # boundary_t.append(box_y)
        # boundary_t.append(box_h)
        # boundary_t.append(box_w)

        # use x_min y_min x_max y_max to represent boundary boxes
        boundary_t.append(box_x_min)
        boundary_t.append(box_y_min)
        boundary_t.append(box_x_max)
        boundary_t.append(box_y_max)
        boundary_tokens += boundary_t
    return boundary_tokens


def render_image(example, predicted_boxes, all_gt_rooms, gt_boxes, output_dir):
    image_height = 256
    image_width = 256
    number_of_color_channels = 3
    background_color = (255, 255, 255)
    gt_image = np.full((image_height, image_width, number_of_color_channels), background_color, dtype=np.uint8)
    predicted_image = np.full((image_height, image_width, number_of_color_channels), background_color, dtype=np.uint8)
    boundary_color = [0, 0, 0]

    left_boundary = int(example.boundary_tokens[1])
    # # draw compass
    # start_point = [240, 40]
    # end_point = [240, 20]
    # thickness = 2
    # cv2.arrowedLine(predicted_image, start_point, end_point,
    #                             boundary_color, thickness,tipLength=0.35)
    # cv2.putText(predicted_image,"N",(230,35),0,0.3,(0,0,0))

    # # draw measure
    # start_point = [left_boundary, 245]
    # end_point = [left_boundary+40, 245]
    # cv2.line(predicted_image, start_point, end_point, boundary_color, thickness)

    # start_point = [left_boundary, 245]
    # end_point = [left_boundary, 240]
    # cv2.line(predicted_image, start_point, end_point, boundary_color, thickness)

    # start_point = [left_boundary+40, 245]
    # end_point = [left_boundary+40, 240]
    # cv2.line(predicted_image, start_point, end_point, boundary_color, thickness)
    # cv2.putText(predicted_image,"10feet",(left_boundary,235),0,0.3,(0,0,0))

    # draw boundary
    for boundary_pixel in example.boundary:
        gt_image[boundary_pixel[0], boundary_pixel[1]] = boundary_color
        predicted_image[boundary_pixel[0], boundary_pixel[1]] = boundary_color

    for room in predicted_boxes:
        if room in all_gt_rooms:
            left_top_pr = (predicted_boxes[room][0], predicted_boxes[room][1])
            right_bt_pr = (predicted_boxes[room][2], predicted_boxes[room][3])
            left_top_gt = (gt_boxes[room][0], gt_boxes[room][1])
            right_bt_gt = (gt_boxes[room][2], gt_boxes[room][3])

            color = color_idx[room_idx[room]]
            # draw room on ground truth image
            cv2.rectangle(gt_image, left_top_gt, right_bt_gt, color, 2)
            # cv2.putText(gt_image,str(room),(left_top_gt[0]+5,left_top_gt[1]+10),0,0.3,color)
            # draw room on predicted image
            cv2.rectangle(predicted_image, left_top_pr, right_bt_pr, color, 2)
            # cv2.putText(predicted_image,str(room),(left_top_pr[0]+5,left_top_pr[1]+10),0,0.3,color)

        else:
            left_top_pr = (predicted_boxes[room][0], predicted_boxes[room][1])
            right_bt_pr = (predicted_boxes[room][2], predicted_boxes[room][3])

            color = color = color_idx[room_idx[room]]
            # draw room on predicted image
            cv2.rectangle(predicted_image, left_top_pr, right_bt_pr, color, 2)
            # cv2.putText(predicted_image,str(room),(left_top_pr[0]+5,left_top_pr[1]+10),0,0.3,color)

    for room in all_gt_rooms:
        if room not in predicted_boxes:
            left_top_gt = (gt_boxes[room][0], gt_boxes[room][1])
            right_bt_gt = (gt_boxes[room][2], gt_boxes[room][3])
            color = color = color = color_idx[room_idx[room]]  # randomize a color for one specific room
            # draw room on ground truth image
            cv2.rectangle(gt_image, left_top_gt, right_bt_gt, color, 2)
            # cv2.putText(gt_image,str(room),(left_top_gt[0]+5,left_top_gt[1]+10),0,0.3,color)

    # cv2.putText(predicted_image,"predicted",(10,10),0,0.3,boundary_color)
    cv2.putText(gt_image, "ground truth", (10, 10), 0, 0.3, boundary_color)

    im_h = cv2.hconcat([gt_image, predicted_image])
    # cv2.imshow('image',im_h)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    # cv2.imwrite(f'{output_dir}output_images/{example.image_id}.png', im_h)
    # cv2.imwrite(f'{output_dir}predicted/{example.image_id}.png', predicted_image)
    cv2.imwrite(f'{output_dir}no_label/{example.image_id}.png', predicted_image)


def calculate_intersection(box1, box2):
    # ymin, xmin, ymax, xmax = box
    y11, x11, y21, x21 = box1
    y12, x12, y22, x22 = box2

    yi1 = max(y11, y12)
    xi1 = max(x11, x12)
    yi2 = min(y21, y22)
    xi2 = min(x21, x22)
    inter_area = max(((xi2 - xi1) * (yi2 - yi1)), 0)
    return inter_area


def calculate_union(box1, box2):
    # ymin, xmin, ymax, xmax = box
    y11, x11, y21, x21 = box1
    y12, x12, y22, x22 = box2

    yi1 = max(y11, y12)
    xi1 = max(x11, x12)
    yi2 = min(y21, y22)
    xi2 = min(x21, x22)
    inter_area = max(((xi2 - xi1) * (yi2 - yi1)), 0)
    box1_area = (x21 - x11) * (y21 - y11)
    box2_area = (x22 - x12) * (y22 - y12)
    union_area = box1_area + box2_area - inter_area
    return union_area


def calculate_iou(gt_boxes, predicted_boxes):
    # group by same room type
    gt_group = defaultdict()
    for gt_b in gt_boxes.keys():
        if gt_b.split()[0] not in gt_group:
            gt_group[gt_b.split()[0]] = [gt_b]
        else:
            gt_group[gt_b.split()[0]].append(gt_b)
    pre_group = defaultdict()
    for pred_b in predicted_boxes.keys():
        if pred_b.split()[0] not in pre_group:
            pre_group[pred_b.split()[0]] = [pred_b]
        else:
            pre_group[pred_b.split()[0]].append(pred_b)
    # ymin, xmin, ymax, xmax = box

    # macro average IoU
    union = 0
    intersection = 0
    gt_group1 = gt_group.copy()
    for group in pre_group.keys():
        if group in gt_group1.keys():

            group1 = pre_group[group]
            g1_boxes = []
            for r1 in group1:
                g1_boxes.append(
                    box(predicted_boxes[r1][0], predicted_boxes[r1][1], predicted_boxes[r1][2], predicted_boxes[r1][3]))

            group2 = gt_group1[group]
            g2_boxes = []
            for r2 in group2:
                g2_boxes.append(box(gt_boxes[r2][0], gt_boxes[r2][1], gt_boxes[r2][2], gt_boxes[r2][3]))

            all_boxes = g1_boxes + g2_boxes
            all_union = unary_union(all_boxes)
            all_union_area = all_union.area
            union += all_union_area

            union1 = unary_union(g1_boxes)
            union2 = unary_union(g2_boxes)
            all_intersection = union1.intersection(union2)
            all_intersection_area = all_intersection.area
            intersection += all_intersection_area
            gt_group1.pop(group)
        else:  # if predicted doesnt exist in ground truth
            group1 = pre_group[group]
            g1_boxes = []
            for r1 in group1:
                g1_boxes.append(
                    box(predicted_boxes[r1][0], predicted_boxes[r1][1], predicted_boxes[r1][2], predicted_boxes[r1][3]))
            all_union = unary_union(g1_boxes)
            all_union_area = all_union.area
            all_intersection_area = 0
            union += all_union_area
    for group in gt_group1.keys():  # if ground truth doesnt exist in predicted
        group2 = gt_group1[group]
        g2_boxes = []
        for r2 in group2:
            g2_boxes.append(box(gt_boxes[r2][0], gt_boxes[r2][1], gt_boxes[r2][2], gt_boxes[r2][3]))
        all_union = unary_union(g2_boxes)
        all_union_area = all_union.area
        all_intersection_area = 0
        union += all_union_area
    if union != 0:
        macro_average_iou = intersection / union
    else:
        macro_average_iou = 0

    # micro average IoU
    num = 0
    total_iou = 0
    gt_group2 = gt_group.copy()
    for group in pre_group.keys():
        if group in gt_group2.keys():

            group1 = pre_group[group]
            g1_boxes = []
            for r1 in group1:
                g1_boxes.append(
                    box(predicted_boxes[r1][0], predicted_boxes[r1][1], predicted_boxes[r1][2], predicted_boxes[r1][3]))

            group2 = gt_group2[group]
            g2_boxes = []
            for r2 in group2:
                g2_boxes.append(box(gt_boxes[r2][0], gt_boxes[r2][1], gt_boxes[r2][2], gt_boxes[r2][3]))

            all_boxes = g1_boxes + g2_boxes
            all_union = unary_union(all_boxes)
            all_union_area = all_union.area
            # union += all_union_area

            union1 = unary_union(g1_boxes)
            union2 = unary_union(g2_boxes)
            all_intersection = union1.intersection(union2)
            all_intersection_area = all_intersection.area
            # intersection += all_intersection_area
            total_iou += all_intersection_area / all_union_area
            num += 1
            gt_group2.pop(group)
        else:  # if predicted doesnt exist in ground truth
            num += 1
    for group in gt_group2.keys():  # if ground truth doesnt exist in predicted
        num += 1
    if num != 0:
        micro_average_iou = total_iou / num
    else:
        micro_average_iou = 0

    return macro_average_iou, micro_average_iou
    # # macro average IoU
    # all_gt_rooms_copy1 = all_gt_rooms.copy()
    # intersection = 0
    # union = 0
    # for room in predicted_boxes:
    #     if room in all_gt_rooms_copy1:
    #         all_gt_rooms_copy1.remove(room)
    #         intersection += calculate_intersection(gt_boxes[room], predicted_boxes[room])
    #         union += calculate_union(gt_boxes[room], predicted_boxes[room])
    #     else:
    #         union += calculate_union((0,0,0,0), predicted_boxes[room])
    # for room in all_gt_rooms_copy1: # rooms that does not apear in predicted results
    #     union += calculate_union((0,0,0,0), gt_boxes[room])
    # if union != 0:
    #     macro_average_iou = intersection/union
    # else:
    #     macro_average_iou = 0

    # # micro average IoU
    # all_gt_rooms_copy2 = all_gt_rooms.copy()
    # total_iou = 0
    # num = 0
    # for room in predicted_boxes:
    #     if room in all_gt_rooms_copy2:
    #         all_gt_rooms_copy2.remove(room)
    #         intersection += calculate_intersection(gt_boxes[room], predicted_boxes[room])
    #         union += calculate_union(gt_boxes[room], predicted_boxes[room])
    #         total_iou += intersection/union
    #         num += 1
    #     else:
    #         num += 1
    # for room in all_gt_rooms_copy2: # rooms that does not apear in predicted results
    #     num += 1
    # if num != 0:
    #     micro_average_iou = total_iou/num
    # else:
    #     micro_average_iou = 0


def get_episode_indices(episodes_string: str) -> List[int]:
    """
    Parse a string such as '2' or '1-5' into a list of integers such as [2] or [1, 2, 3, 4, 5].
    """
    episode_indices = []

    if episodes_string and episodes_string != '':
        ll = [int(item) for item in episodes_string.split('-')]

        if len(ll) == 1:
            episode_indices = ll

        else:
            _start, _end = ll
            episode_indices = list(range(_start, _end + 1))

    return episode_indices


def expand_tokens(tokens: List[str], augmentations: List[Tuple[List[tuple], int, int]],
                  entity_tree: Dict[int, List[int]], root: int,
                  begin_entity_token: str, sep_token: str, relation_sep_token: str, end_entity_token: str) \
        -> List[str]:
    """
    Recursively expand the tokens to obtain a sentence in augmented natural language.

    Used in the augment_sentence function below (see the documentation there).
    """
    new_tokens = []
    root_start, root_end = augmentations[root][1:] if root >= 0 else (0, len(tokens))
    i = root_start  # current index

    for entity_index in entity_tree[root]:
        tags, start, end = augmentations[entity_index]

        # add tokens before this entity
        new_tokens += tokens[i:start]

        # expand this entity
        new_tokens.append(begin_entity_token)
        new_tokens += expand_tokens(tokens, augmentations, entity_tree, entity_index,
                                    begin_entity_token, sep_token, relation_sep_token, end_entity_token)

        for tag in tags:
            if tag[0]:
                # only append tag[0] if it is a type, otherwise skip the type
                new_tokens.append(sep_token)
                new_tokens.append(tag[0])

            for x in tag[1:]:
                new_tokens.append(relation_sep_token)
                new_tokens.append(x)

        new_tokens.append(end_entity_token)
        i = end

    # add tokens after all entities
    new_tokens += tokens[i:root_end]

    return new_tokens


def augment_sentence(tokens: List[str], augmentations: List[Tuple[List[tuple], int, int]], begin_entity_token: str,
                     sep_token: str, relation_sep_token: str, end_entity_token: str) -> str:
    """
    Augment a sentence by adding tags in the specified positions.

    Args:
        tokens: Tokens of the sentence to augment.
        augmentations: List of tuples (tags, start, end).
        begin_entity_token: Beginning token for an entity, e.g. '['
        sep_token: Separator token, e.g. '|'
        relation_sep_token: Separator token for relations, e.g. '='
        end_entity_token: End token for an entity e.g. ']'

    An example follows.

    tokens:
    ['Tolkien', 'was', 'born', 'here']

    augmentations:
    [
        ([('person',), ('born in', 'here')], 0, 1),
        ([('location',)], 3, 4),
    ]

    output augmented sentence:
    [ Tolkien | person | born in = here ] was born [ here | location ]
    """
    # sort entities by start position, longer entities first
    augmentations = list(sorted(augmentations, key=lambda z: (z[1], -z[2])))

    # check that the entities have a tree structure (if two entities overlap, then one is contained in
    # the other), and build the entity tree
    root = -1  # each node is represented by its position in the list of augmentations, except that the root is -1
    entity_tree = {root: []}  # list of children of each node
    current_stack = [root]  # where we are in the tree

    for j, x in enumerate(augmentations):
        tags, start, end = x
        if any(augmentations[k][1] < start < augmentations[k][2] < end for k in current_stack):
            # tree structure is not satisfied!
            logging.warning(f'Tree structure is not satisfied! Dropping annotation {x}')
            continue

        while current_stack[-1] >= 0 and \
                not (augmentations[current_stack[-1]][1] <= start <= end <= augmentations[current_stack[-1]][2]):
            current_stack.pop()

        # add as a child of its father
        entity_tree[current_stack[-1]].append(j)

        # update stack
        current_stack.append(j)

        # create empty list of children for this new node
        entity_tree[j] = []

    return ' '.join(expand_tokens(
        tokens, augmentations, entity_tree, root, begin_entity_token, sep_token, relation_sep_token, end_entity_token
    ))


def get_span(l: List[str], span: List[int]):
    assert len(span) == 2
    return " ".join([l[i] for i in range(span[0], span[1]) if i < len(l)])


def get_precision_recall_f1(num_correct, num_predicted, num_gt):
    assert 0 <= num_correct <= num_predicted
    assert 0 <= num_correct <= num_gt

    precision = num_correct / num_predicted if num_predicted > 0 else 0.
    recall = num_correct / num_gt if num_gt > 0 else 0.
    f1 = 2. / (1. / precision + 1. / recall) if num_correct > 0 else 0.

    return precision, recall, f1


def format_short_output_(example: InputExample) -> str:
    string = ''
    start_token = '['
    end_token = ']'
    sep = '|'
    for room in example.rooms:
        string += f'{start_token} {room.type} {sep} x coordinate = {str(room.x)} {sep} y coordinate = {str(room.y)} {sep} height = {str(room.h)} {sep} width = {str(room.w)} {end_token} '
    return string


def zy_parse_output_sentence(output_sentence):
    output_tokens = []
    unmatched_predicted_entities = []

    BEGIN_ENTITY_TOKEN = '['
    END_ENTITY_TOKEN = ']'
    SEPARATOR_TOKEN = '|'
    RELATION_SEPARATOR_TOKEN = '='

    # add spaces around special tokens, so that they are alone when we split
    padded_output_sentence = output_sentence
    for special_token in [
        BEGIN_ENTITY_TOKEN, END_ENTITY_TOKEN,
        SEPARATOR_TOKEN, RELATION_SEPARATOR_TOKEN,
    ]:
        padded_output_sentence = padded_output_sentence.replace(special_token, ' ' + special_token + ' ')

    entity_stack = []  # stack of the entities we are extracting from the output sentence
    # this is a list of lists [start, state, entity_name_tokens, entity_other_tokens]
    # where state is "name" (before the first | separator) or "other" (after the first | separator)

    for token in padded_output_sentence.split():
        if len(token) == 0:
            continue

        elif token == BEGIN_ENTITY_TOKEN:
            # begin entity
            start = len(output_tokens)
            entity_stack.append([start, "other", [], []])

        elif token == END_ENTITY_TOKEN and len(entity_stack) > 0:
            # end entity
            start, state, entity_name_tokens, entity_other_tokens = entity_stack.pop()

            entity_name = ' '.join(entity_name_tokens).strip()
            end = len(output_tokens)

            tags = []

            # split entity_other_tokens by |
            splits = [
                list(y) for x, y in itertools.groupby(entity_other_tokens, lambda z: z == SEPARATOR_TOKEN)
                if not x
            ]

            if state == "other" and len(splits) > 0:
                for x in splits:
                    tags.append(tuple(' '.join(x).split(' ' + RELATION_SEPARATOR_TOKEN + ' ')))

            unmatched_predicted_entities.append((entity_name, tags, start, end))

        else:
            # a normal token
            if len(entity_stack) > 0:
                # inside some entities
                if token == SEPARATOR_TOKEN:
                    x = entity_stack[-1]

                    if x[1] == "name":
                        # this token marks the end of name tokens for the current entity
                        x[1] = "other"
                    else:
                        # simply add this token to entity_other_tokens
                        x[3].append(token)

                else:
                    is_name_token = True

                    for x in reversed(entity_stack):
                        # check state
                        if x[1] == "name":
                            # add this token to entity_name_tokens
                            x[2].append(token)

                        else:
                            # add this token to entity_other tokens and then stop going up in the tree
                            x[3].append(token)
                            is_name_token = False
                            break

                    if is_name_token:
                        output_tokens.append(token)

            else:
                # outside
                output_tokens.append(token)

    # update predicted entities with the positions in the original sentence
    predicted_entities = []

    for entity_name, entity_tags, start, end in unmatched_predicted_entities:
        new_start = None  # start in the original sequence
        new_end = None  # end in the original sequence

        entity_tuple = (entity_name, entity_tags, new_start, new_end)
        predicted_entities.append(entity_tuple)

    return predicted_entities


def run_inference(output_sentence: str):
    new_rooms = []
    output_tokens = output_sentence.split()
    if '[' in output_tokens and '|' in output_tokens and 'x_max' in output_tokens:  # long output format
        output_tokens = output_sentence.split('[')
        output_tokens.pop(0)
        tokens = "".join(output_tokens).split(']')
        tokens.pop(-1)
        for token in tokens:
            tok = token.split('|')
            room_type = tok[0].strip()
            for t in tok[1:]:
                if 'x min' in t:
                    x_min = t.split('=')[1].strip()
                elif 'y min' in t:
                    y_min = t.split('=')[1].strip()
                elif 'x_max' in t:
                    x_max = t.split('=')[1].strip()
                elif 'y_max' in t:
                    y_max = t.split('=')[1].strip()
            x = int((int(x_min) + int(x_max)) / 2)
            y = int((int(y_min) + int(y_max)) / 2)
            w = int(y_max) - int(y_min)
            h = int(x_max) - int(x_min)
            room = Room(type=room_type, x=x, y=y, h=h, w=w)
            new_rooms.append(room)
        predicted_examples = InputExample(rooms=new_rooms)
        output_sentence_ = format_short_output_(predicted_examples)
        pass
    elif '[' in output_tokens and '|' in output_tokens:  # original output format
        output_sentence_ = output_sentence
        pass
    elif '-1' in output_tokens:  # short-relation output format
        rooms = []
        room_attributes = []
        index = []
        flag = False
        for i in range(len(output_tokens)):
            if flag == False:
                try:
                    token = int(output_tokens[i])
                    if token > 4:
                        flag = True
                        index.append(i)
                except:
                    pass
            else:
                try:
                    token = int(output_tokens[i])
                except:
                    flag = False
        prev_idx = 0
        for idx in index:
            room_attributes.append(output_tokens[idx:idx + 4])
            rooms.append(" ".join(output_tokens[prev_idx:idx]))
            prev_idx = idx + 8
        for i in range(len(rooms)):
            # xmin ymin xmax ymax
            x = int((int(room_attributes[i][0]) + int(room_attributes[i][2])) / 2)
            y = int((int(room_attributes[i][1]) + int(room_attributes[i][3])) / 2)
            w = int(room_attributes[i][3]) - int(room_attributes[i][1])
            h = int(room_attributes[i][2]) - int(room_attributes[i][0])
            new_rooms.append(Room(type=rooms[i], x=x, y=y, h=h, w=w))
        predicted_examples = InputExample(rooms=new_rooms)
        output_sentence_ = format_short_output_(predicted_examples)
        pass
    else:  # short output format
        rooms = []
        room_attributes = []
        index = []
        flag = False
        for i in range(len(output_tokens)):
            if flag == False:
                try:
                    token = int(output_tokens[i])
                    if token > 4:
                        flag = True
                        index.append(i)
                except:
                    pass
            else:
                try:
                    token = int(output_tokens[i])
                except:
                    flag = False
        prev_idx = 0
        for idx in index:
            room_attributes.append(output_tokens[idx:idx + 4])
            rooms.append(" ".join(output_tokens[prev_idx:idx]))
            prev_idx = idx + 4
        for i in range(len(rooms)):
            # xmin ymin xmax ymax
            x = int((int(room_attributes[i][0]) + int(room_attributes[i][2])) / 2)
            y = int((int(room_attributes[i][1]) + int(room_attributes[i][3])) / 2)
            w = int(room_attributes[i][3]) - int(room_attributes[i][1])
            h = int(room_attributes[i][2]) - int(room_attributes[i][0])
            new_rooms.append(Room(type=rooms[i], x=x, y=y, h=h, w=w))
        predicted_examples = InputExample(rooms=new_rooms)
        output_sentence_ = format_short_output_(predicted_examples)

    room_types = ['living room', 'master room', 'kitchen', 'bathroom', 'dining room', 'common room 2',
                  'common room 3', 'common room 1', 'common room 4', 'balcony'
        , 'entrance', 'storage', 'common room']
    attribute_types = ['x coordinate', 'y coordinate', 'height', 'width']
    format_error = False  # whether the augmented language format is invalid
    label_error = False

    if output_sentence_.count('[') != output_sentence_.count(']'):
        # the parentheses do not match
        format_error = True

    raw_predictions, wrong_reconstruction = zy_parse_output_sentence(output_sentence_), None

    # update predicted entities with the positions in the original sentence
    predicted_rooms_by_name = defaultdict(list)
    predicted_rooms = set()
    raw_predicted_relations = []

    # process and filter entities
    for entity_name, tags, start, end in raw_predictions:
        if len(tags) == 0 or len(tags[0]) > 1:
            # we do not have a tag for the room type
            format_error = True
            continue

        room_type = tags[0][0]

        if room_type in room_types or room_type[:-2] in room_types:
            room_tuple = (room_type, start, end)
            predicted_rooms.add(room_tuple)
            predicted_rooms_by_name[room_type].append(room_tuple)

            # process tags to get relations
            for tag in tags[1:]:
                if len(tag) == 2:
                    attribute_type, value = tag
                    if attribute_type in attribute_types:
                        raw_predicted_relations.append((attribute_type, value, room_tuple, room_type))
                    else:
                        label_error = True

                else:
                    # the relation tag has the wrong length
                    format_error = True

        else:
            # the predicted entity type does not exist
            label_error = True

        # error = format_error or label_error or wrong_reconstruction  # whether there is syntax error

    return predicted_rooms_by_name, predicted_rooms, raw_predicted_relations, \
        wrong_reconstruction, format_error, label_error


def render_floor_plan_by_output_sentence(output_sentence):
    res = run_inference(output_sentence)
    predicted_rooms_by_name, predicted_rooms, raw_predicted_relations, wrong_reconstruction, format_error, label_error = res

    predicted_attribute = defaultdict()
    for attribute_tuple in raw_predicted_relations:
        attribute_type, value, room_tuple, room_type = attribute_tuple
        if room_type not in predicted_attribute:
            predicted_attribute[room_type] = defaultdict()
        try:
            value = int(value)
        except:
            value = 128
        predicted_attribute[room_type][attribute_type] = value

    # TODO: examine the predicted_attribute patterns
    correct_attributes = ['x coordinate', 'y coordinate', 'height', 'width']
    wrong_room = []
    for room_type in predicted_attribute:
        if set(list(predicted_attribute[room_type].keys())) != set(correct_attributes):
            print('wrong output format:')
            print(predicted_attribute[room_type])
            wrong_room.append(room_type)
    for wrong_r in wrong_room:
        predicted_attribute.pop(wrong_r)

    predicted_boxes = defaultdict()
    for room in predicted_attribute:
        predicted_boxes[room] = [
            [int(predicted_attribute[room]['x coordinate'] - predicted_attribute[room]['height'] / 2),
             int(predicted_attribute[room]['y coordinate'] - predicted_attribute[room]['width'] / 2)],
            [int(predicted_attribute[room]['x coordinate'] + predicted_attribute[room]['height'] / 2),
             int(predicted_attribute[room]['y coordinate'] - predicted_attribute[room]['width'] / 2)],
            [int(predicted_attribute[room]['x coordinate'] - predicted_attribute[room]['height'] / 2),
             int(predicted_attribute[room]['y coordinate'] + predicted_attribute[room]['width'] / 2)],
            [int(predicted_attribute[room]['x coordinate'] + predicted_attribute[room]['height'] / 2),
             int(predicted_attribute[room]['y coordinate'] + predicted_attribute[room]['width'] / 2)]
        ]

    for room in predicted_boxes:
        y_min = predicted_boxes[room][0][1]
        x_min = predicted_boxes[room][0][0]
        y_max = predicted_boxes[room][3][1]
        x_max = predicted_boxes[room][3][0]
        predicted_boxes[room] = (y_min, x_min, y_max, x_max)

    # render_image_
    image_height = 256
    image_width = 256
    number_of_color_channels = 3
    background_color = (255, 255, 255)
    predicted_image = np.full((image_height, image_width, number_of_color_channels), background_color, dtype=np.uint8)
    boundary_color = [0, 0, 0]

    living = defaultdict()
    common = defaultdict()
    master = defaultdict()
    balcony = defaultdict()
    bathroom = defaultdict()
    kitchen = defaultdict()
    storage = defaultdict()
    dining = defaultdict()
    for room in predicted_boxes:
        if room.startswith('living'):
            living[room] = predicted_boxes[room]
        elif room.startswith('common'):
            common[room] = predicted_boxes[room]
        elif room.startswith('master'):
            master[room] = predicted_boxes[room]
        elif room.startswith('balcony'):
            balcony[room] = predicted_boxes[room]
        elif room.startswith('bathroom'):
            bathroom[room] = predicted_boxes[room]
        elif room.startswith('kitchen'):
            kitchen[room] = predicted_boxes[room]
        elif room.startswith('storage'):
            storage[room] = predicted_boxes[room]
        elif room.startswith('dining'):
            dining[room] = predicted_boxes[room]

    room_type_list = [living, common, master, balcony, bathroom, kitchen, storage, dining]
    image_room_list = []
    for room_type in room_type_list:
        for room in room_type:
            image_room_list.append(room)
            left_top_pr = (room_type[room][0], room_type[room][1])
            right_bt_pr = (room_type[room][2], room_type[room][3])

            color = color_idx[room_idx[room]]
            # draw room on predicted image
            cv2.rectangle(predicted_image, left_top_pr, right_bt_pr, color, -1)

    return predicted_image, image_room_list


def render_floor_plan(example, output_sentence, dataset, with_boundary=True):
    predicted_index = None
    image_id = vars(example)['image_id']

    gt_boxes = defaultdict(list)
    for room in example.rooms:
        gt_x, gt_y, gt_h, gt_w = room.x, room.y, room.h, room.w
        gt_box = [
            [int(gt_x - gt_h / 2), int(gt_y - gt_w / 2)], [int(gt_x + gt_h / 2), int(gt_y - gt_w / 2)],
            [int(gt_x - gt_h / 2), int(gt_y + gt_w / 2)], [int(gt_x + gt_h / 2), int(gt_y + gt_w / 2)]
        ]
        gt_boxes[room.type] = gt_box
    res = dataset.output_format.run_inference(example, output_sentence, predicted_index)
    predicted_rooms_by_name, predicted_rooms, raw_predicted_relations, wrong_reconstruction, format_error, label_error = res

    predicted_attribute = defaultdict()
    for attribute_tuple in raw_predicted_relations:
        attribute_type, value, room_tuple, room_type = attribute_tuple
        if room_type not in predicted_attribute:
            predicted_attribute[room_type] = defaultdict()
        try:
            value = int(value)
        except:
            value = 128
        predicted_attribute[room_type][attribute_type] = value

    # TODO: examine the predicted_attribute patterns
    correct_attributes = ['x coordinate', 'y coordinate', 'height', 'width']
    wrong_room = []
    for room_type in predicted_attribute:
        if set(list(predicted_attribute[room_type].keys())) != set(correct_attributes):
            print('wrong output format:')
            print(predicted_attribute[room_type])
            wrong_room.append(room_type)
    for wrong_r in wrong_room:
        predicted_attribute.pop(wrong_r)

    predicted_boxes = defaultdict()
    for room in predicted_attribute:
        predicted_boxes[room] = [
            [int(predicted_attribute[room]['x coordinate'] - predicted_attribute[room]['height'] / 2),
             int(predicted_attribute[room]['y coordinate'] - predicted_attribute[room]['width'] / 2)],
            [int(predicted_attribute[room]['x coordinate'] + predicted_attribute[room]['height'] / 2),
             int(predicted_attribute[room]['y coordinate'] - predicted_attribute[room]['width'] / 2)],
            [int(predicted_attribute[room]['x coordinate'] - predicted_attribute[room]['height'] / 2),
             int(predicted_attribute[room]['y coordinate'] + predicted_attribute[room]['width'] / 2)],
            [int(predicted_attribute[room]['x coordinate'] + predicted_attribute[room]['height'] / 2),
             int(predicted_attribute[room]['y coordinate'] + predicted_attribute[room]['width'] / 2)]
        ]

    for room in predicted_boxes:
        y_min = predicted_boxes[room][0][1]
        x_min = predicted_boxes[room][0][0]
        y_max = predicted_boxes[room][3][1]
        x_max = predicted_boxes[room][3][0]
        predicted_boxes[room] = (y_min, x_min, y_max, x_max)

    # render_image_
    image_height = 256
    image_width = 256
    number_of_color_channels = 3
    background_color = (255, 255, 255)
    predicted_image = np.full((image_height, image_width, number_of_color_channels), background_color, dtype=np.uint8)
    boundary_color = [0, 0, 0]

    living = defaultdict()
    common = defaultdict()
    master = defaultdict()
    balcony = defaultdict()
    bathroom = defaultdict()
    kitchen = defaultdict()
    storage = defaultdict()
    dining = defaultdict()
    for room in predicted_boxes:
        if room.startswith('living'):
            living[room] = predicted_boxes[room]
        elif room.startswith('common'):
            common[room] = predicted_boxes[room]
        elif room.startswith('master'):
            master[room] = predicted_boxes[room]
        elif room.startswith('balcony'):
            balcony[room] = predicted_boxes[room]
        elif room.startswith('bathroom'):
            bathroom[room] = predicted_boxes[room]
        elif room.startswith('kitchen'):
            kitchen[room] = predicted_boxes[room]
        elif room.startswith('storage'):
            storage[room] = predicted_boxes[room]
        elif room.startswith('dining'):
            dining[room] = predicted_boxes[room]

    room_type_list = [living, common, master, balcony, bathroom, kitchen, storage, dining]
    image_room_list = []
    for room_type in room_type_list:
        for room in room_type:
            image_room_list.append(room)
            left_top_pr = (room_type[room][0], room_type[room][1])
            right_bt_pr = (room_type[room][2], room_type[room][3])

            color = color_idx[room_idx[room]]
            # draw room on predicted image
            cv2.rectangle(predicted_image, left_top_pr, right_bt_pr, color, -1)

    if with_boundary:
        for boundary_pixel in example.boundary:
            predicted_image[boundary_pixel[0], boundary_pixel[1]] = boundary_color

    return predicted_image, image_room_list
