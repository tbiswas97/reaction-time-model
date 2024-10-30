import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import entropy
import sys

sys.path.append("./GSM_rsc_VGG_unity_project/src/")

import toolbox as tb


def _get_kld_ij(coord1, coord2, pmap):
    n_iter = pmap.shape[0]

    pmap_a = pmap[:, :, coord1[0], coord1[1]]

    pmap_b = pmap[:, :, coord2[0], coord2[1]]

    return entropy(pmap_a, pmap_b, axis=1)


def _get_psame_t(coord1, coord2, pmap):
    """
    Given two coordinates and a EM segmentation probability map:

    Parameters:
    ------------
    coord1 : int coord2 : int
    pmap : ndarray of shape (n_iter,n_components,ny,nx)
        n_iter : number of EM iterations
        n_components : number of components in mixture
        ny : height of image
        nx : width of image

    Returns:
    ------------
    psame_t : ndarray of float length n_iter
    seg_flag : ndarray of bool of length n_iter
    """

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
    """
    Given an EM segmentation probability map, returns the log odds of the
    probability of the pair being in the same segment (p_same)

    log odds = log((p_same)/(1-p_same))

    Parameters:
    -----------
    pair : tup or tup-like
    pmap : ndarray of shape (n_iter,n_components,ny,nx)
        n_iter : number of EM iterations
        n_components : number of components in mixture
        ny : height of image
        nx : width of image

    Returns:
    --------
    logit : ndarray of float of length n_iter
    """
    # uses only the first output of analysis._get_psame_t
    p, _ = _get_psame_t(pair[0], pair[1], pmap)
    logit = np.log((p) / (1 - p))

    return logit, p[-1]


def get_final_segmentation_assignment(pair, pmap):
    """
    Given an EM segmentation probability map, returns whether a pair of pixels n
    and m are given then same segmentation assignment.

    segmentation assignment = (argmax(p(n=k)) == argmax(p(m==k))

    Parameters:
    -----------
    pair : tup or tup-like pmap : ndarray of shape (n_iter,n_components,ny,nx)
        n_iter : number of EM iterations n_components : number of components in
        mixture ny : height of image nx : width of image

    pmap : ndarray of shape (n_iter,n_components,ny,nx)
        n_iter : number of EM iterations n_components : number of components in
        mixture ny : height of image nx : width of image

    Returns:
    --------
    sf : ndarray of bool of length n_iter
    """
    _, sf = _get_psame_t(pair[0], pair[1], pmap)

    return sf[-1]


def get_model_reaction_time(pair, pmap, decision_bounds, evidence="first"):
    """
    Given an EM segmentation probability map, returns the model's decision
    "reaction time" for how long it takes to accumulate evidence that a pair is
    in the same or different segment.

    if log odds > decision_bounds the decision is made, the EM iteration where
    this first occurs is the "reaction time"

    Parameters:
    -----------
    pair : tup or tup-like pmap : ndarray of shape (n_iter,n_components,ny,nx)
        n_iter : number of EM iterations n_components : number of components in
        mixture ny : height of image nx : width of image
    decision_bounds : tup of float
        if (log odds > decision_bounds[1] or
            log odds < decision_bounds[0]):

            the decision is made and the EM iteration index is returned as rt
    evidence : str
        Defines what kind of evidence will be used in determining the reaction
        time:
            "first" : evidence is the first time the accumulation curve crosses
                the decision boundary
            "area" : evidence is the area underneath the
                accumulation curve
    Returns:
    --------
    rt : int or float

    """
    # returns logit at each EM iteration
    logits = get_logits(pair, pmap)[0]
    if evidence == "first":
        assert len(decision_bounds) == 2
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
    elif evidence == "area":
        rt = -1 * np.log(np.sum((abs(logits))))
        # rt = 1 / (np.sum(abs(logits)))
        # just use sum instead of area?
        # rt = np.trapz(abs_v)
    return rt


def get_bin(_bin, pairs, reaction_times=None, responses=None, p_same=None):
    """
    Given a list of pairs calculates all distances between them and bins these
    distances into n_bins bins

    Parameters:
    -----------
    _bin : int
        the bin at which to gather data, must be positive
    pairs : list of tup or tup-like
        the pairs to consider
    reaction_times : list
        provided as a field in Response class
    responses : list
        provided as a field in Response class

    Returns:
    ---------
    d : dict
        A dictionary with the information from a specific bin as specified by
        _bin parameter
    """

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
    if p_same is not None:
        assert len(p_same) == len(responses)

    assert _bin >= 0, "Bin ID must be positive int"

    if _bin == 0:
        cond = (distances > 0) & (distances < edges[0])
    else:
        cond = (distances > edges[_bin - 1]) & (distances < edges[_bin])

    keys = ["pairs", "distances", "reaction_times", "responses", "p_same"]
    values = [pairs, distances, reaction_times, responses, p_same]

    d = {k: v[cond] for k, v in zip(keys, values) if v is not None}

    return d


def _get_bin_df(
    _bin,
    pairs,
    pmap,
    reaction_times=None,
    responses=None,
    p_same=None,
    decision_bounds=None,
    evidence="first",
):
    """
    Given a list of pairs calculates all distances between them, bins these
    distances into n_bins bins, and outputs a DataFrame which includes Model information as well

    Parameters:
    -----------
    _bin : int
        the bin at which to gather data, must be positive
    pmap : ndarray of shape (n_iter,n_components,ny,nx)
        n_iter : number of EM iterations
        n_components : number of components in mixture
        ny : height of image
        nx : width of image
    pairs : list of tup or tup-like
        the pairs to consider
    reaction_times : list
        provided as a field in Response class
    responses : list
        provided as a field in Response class

    Returns:
    ---------
    df : pd.DataFrame
        A DataFrame with the information from a specific bin as specified by
        _bin parameter
    """
    d = get_bin(_bin, pairs, reaction_times, responses, p_same)

    pairs_binned = d["pairs"]
    distances = d["distances"]
    if reaction_times is not None:
        reaction_times = d["reaction_times"]
        reaction_times = [time for time in reaction_times]
    if responses is not None:
        responses = d["responses"]
        responses = [response for response in responses]
    if p_same is not None:
        p_same = d["p_same"]
        p_same = [p for p in p_same]

    pairs_binned = [pair for pair in pairs_binned]
    distances = [dist for dist in distances]
    bin_num = [_bin] * len(pairs_binned)
    coord1 = [pair[0] for pair in pairs_binned]
    coord2 = [pair[1] for pair in pairs_binned]
    if decision_bounds is not None:
        decision_bounds = decision_bounds
    else:
        decision_bounds = (-5, 5)

    logits = [get_logits(pair, pmap)[0] for pair in pairs_binned]

    p_same_last = [get_logits(pair, pmap)[1] for pair in pairs_binned]

    rt = [
        get_model_reaction_time(pair, pmap, decision_bounds, evidence=evidence)
        for pair in pairs_binned
    ]
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
        "model_p_same": p_same_last,
        "p_same": p_same,
        "responses": responses,
    }

    d_logit = {"logit_{}".format(i): logits[i] for i in range(len(logits))}

    df = pd.DataFrame.from_dict(d)
    df_log = pd.DataFrame.from_dict(d_logit)

    df = pd.concat([df, df_log], axis=1)

    if p_same is not None:
        df["kl_divergence"] = df["model_p_same"] * np.log(df["model_p_same"]) - df[
            "model_p_same"
        ] * np.log(df["p_same"])

    return df


def get_df(
    pairs,
    pmap,
    reaction_times=None,
    responses=None,
    p_same=None,
    bins=range(1, 10),
    decision_bounds=None,
    evidence="first",
    condition=None,
):
    """
    Wrapper around _get_bin_df, which performs that function for every bin
    in bins

    Parameters:
    -----------
    pairs : list of tup or tup-like
        the pairs to consider
    pmap : ndarray of shape (n_iter,n_components,ny,nx)
        n_iter : number of EM iterations n_components : number of components
        in mixture ny : height of image nx : width of image
    reaction_times : list
        provided as a field in Response class
    responses : list
        provided as a field in Response class
    bins : iterable of int
        range(N) specifies the number of bins
    decision_bounds : tup of float
        Passed to above get_model_reaction_time function
        if (log odds >
        decision_bounds[1] or
            log odds < decision_bounds[0]):

            the decision is made and the EM iteration index is returned as
            rt
    condition : str
        Input that notes the particular conditions for that experiment. A
        tag that is associated with each dataframe as an identifier (eg.
        Voronoi prior condition vs. Random prior condition)

    Returns:
    ---------
    df : pd.DataFrame
        A DataFrame with the information from a specific bin as specified by
        _bin parameter
    """
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
                p_same=p_same,
                decision_bounds=decision_bounds,
                evidence=evidence,
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
