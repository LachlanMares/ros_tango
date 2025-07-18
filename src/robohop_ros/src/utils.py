#!/usr/bin/env python3
"""
Author:
    Lachlan Mares, lachlan.mares@adelaide.edu.au

License:
    GPL-3.0

Description:

"""
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np


class ModeFilterClass:
    def __init__(self, filter_length: int):
        self.filter_length = filter_length
        self.buffer = np.arange(filter_length)

    def update(self, new_value: int) -> int:
        self.buffer = np.roll(a=self.buffer, shift=1)
        self.buffer[0] = new_value
        uniques, counts = np.unique(ar=self.buffer, return_counts=True)
        return uniques[np.argmax(counts)]


def rle_to_mask(rle) -> np.ndarray:
    """Compute a binary mask from an uncompressed RLE."""
    h, w = rle["size"]
    mask = np.empty(h * w, dtype=bool)
    idx = 0
    parity = False
    for count in rle["counts"]:
        mask[idx: idx + count] = parity
        idx += count
        parity ^= True
    mask = mask.reshape(w, h)
    return mask.transpose()  # Put in C order


def nodes2key(node_indices, key, graph=None):
    """ """
    _key = key
    if key == 'coords':
        _key = 'segmentation'

    if isinstance(node_indices[0], dict):
        if _key == 'segmentation' and type(node_indices[0][_key]) == dict:
            values = np.array([rle_to_mask(n[_key]) for n in node_indices])
        else:
            values = np.array([n[_key] for n in node_indices])

    else:
        assert graph is not None, "nodes can either be dict or indices of nx.Graph"
        if _key == 'segmentation' and type(graph.nodes[node_indices[0]][_key]) == dict:
            values = np.array([rle_to_mask(graph.nodes[n][_key]) for n in node_indices])
        else:
            values = np.array([graph.nodes[n][_key] for n in node_indices])

    if key == 'coords':
        values = np.array([np.array(np.nonzero(v)).mean(1)[::-1].astype(int) for v in values])

    return values