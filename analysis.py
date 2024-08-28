import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

sys.path.append("./GSM_rsc_VGG_unity_project/src/")

import toolbox as tb


def _get_psame_t(coord1, coord2, pmap):

    n_iter = pmap.shape[0]

    pmap_a = pmap[:, :, coord1[0], coord1[1]]

    seg_a = pmap_a.argmax(1)

    pmap_b = pmap[:, :, coord2[0], coord2[1]]

    seg_b = pmap_b.argmax(1)

    psame_t = np.asarray([np.dot(pmap_a[i], pmap_b[i]) for i in range(n_iter)])
    seg_flag_t = seg_a == seg_b

    assert len(psame_t) == len(seg_flag_t)

    return psame_t, seg_flag_t


def get_logits(pair, pmap):
    p, _ = _get_psame_t(pair[0], pair[1], pmap)
    logit = np.log((p) / (1 - p))

    return logit


def get_final_segmentation_assignment(pair, pmap):
    _, sf = _get_psame_t(pair[0], pair[1], pmap)

    return sf[-1]


def get_model_reaction_time(pair, pmap, decision_bounds):
    logits = get_logits(pair, pmap)
    seg_flag = get_final_segmentation_assignment(pair, pmap)

    if seg_flag:
        try:
            rt = np.where(logits > decision_bounds[1])[0][0]
        except:
            rt = np.nan
    else:
        try:
            rt = np.where(logits < decision_bounds[0])[0][0]
        except:
            rt = np.nan

    return rt


def get_bin(_bin, pairs, reaction_times=None, responses=None):
    distances = np.asarray(
        [
            tb.euclidean_distance(pairs[pair][0], pairs[pair][1])
            for pair in range(len(pairs))
        ]
    )
    edges = np.histogram(distances, bins=10)[1]
    edges = np.asarray(edges)

    if type(pairs) != np.ndarray:
        pairs = np.asarray(pairs)

    if reaction_times is not None:
        assert len(reaction_times) == len(pairs)
    if responses is not None:
        assert len(responses) == len(pairs)

    assert _bin >= 0, "Bin ID must be positive int"

    if _bin == 0:
        cond = (distances > 0) & (distances < edges[0])
    else:
        cond = (distances > edges[_bin - 1]) & (distances < edges[_bin])

    keys = ["pairs", "distances", "reaction_times", "responses"]
    values = [pairs, distances, reaction_times, responses]

    d = {k: v[cond] for k, v in zip(keys, values) if v is not None}

    return d


def _get_bin_df(
    _bin, pairs, pmap, reaction_times=None, responses=None, decision_bounds=None
):

    d = get_bin(_bin, pairs, reaction_times, responses)

    pairs_binned = d["pairs"]
    distances = d["distances"]
    if reaction_times is not None:
        reaction_times = d["reaction_times"]
        reaction_times = [time for time in reaction_times]
    if responses is not None:
        responses = d["responses"]
        responses = [response for response in responses]

    pairs_binned = [pair for pair in pairs_binned]
    distances = [dist for dist in distances]
    bin_num = [_bin] * len(pairs_binned)
    coord1 = [pair[0] for pair in pairs_binned]
    coord2 = [pair[1] for pair in pairs_binned]
    if decision_bounds is not None:
        decision_bounds = decision_bounds
    else:
        decision_bounds = (-5, 5)
    rt = [get_model_reaction_time(pair, pmap, decision_bounds) for pair in pairs_binned]
    seg_flag = [get_final_segmentation_assignment(pair, pmap) for pair in pairs_binned]

    d = {
        "pair_coords": pairs_binned,
        "distance": distances,
        "bin_num": bin_num,
        "coord1": coord1,
        "coord2": coord2,
        "model_rt": rt,
        "rt": reaction_times,
        "seg_flag": seg_flag,
        "responses": responses,
    }

    df = pd.DataFrame.from_dict(d)

    return df


def get_df(
    pairs,
    pmap,
    reaction_times=None,
    responses=None,
    bins=range(1, 10),
    decision_bounds=None,
    condition=None,
):
    if decision_bounds is not None:
        decision_bounds = decision_bounds
    else:
        decision_bounds = (-5, 5)
    df = pd.concat(
        [
            _get_bin_df(
                _bin,
                pairs,
                pmap,
                reaction_times=reaction_times,
                responses=responses,
                decision_bounds=decision_bounds,
            )
            for _bin in bins
        ],
        axis=0,
        ignore_index=True,
    )
    if condition is not None:
        df["condition"] = condition

    return df


# def bin_df(
# _bin,
# pmap,
# ddm_bounds=(-5, 5),
# edges=edges,
# pairs=all_coord_pairs,
# distances=distances,
# ):
# if _bin >= 0:
# to_compute, distances = get_bin(_bin, pairs=pairs)
# else:
# to_compute = pairs

# bin_labels = [_bin] * len(to_compute)
# pairs = list(to_compute)
# distances = list(distances)

# rts = [get_RT(pair[0], pair[1], pmap, ddm_bounds=ddm_bounds)[0] for pair in pairs]

# seg_flags = [
# get_RT(pair[0], pair[1], pmap, ddm_bounds=ddm_bounds)[1] for pair in pairs
# ]

# d = {
# "bin_num": bin_labels,
# "pair": pairs,
# "distances": distances,
# "RT": rts,
# "seg_flag": seg_flags,
# }

# df = pd.DataFrame.from_dict(d)

# return df
