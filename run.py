import sys

sys.path.append("./GSM_rsc_VGG_unity_project/src/")
import import_utils
import toolbox as tb
import numpy as np
import pandas as pd
import utils
import matplotlib.pyplot as plt
from glob import glob as glob
from natsort import natsorted as ns
from Response import Response as Res
from SegmentationMap import SegmentationMap as SM
import analysis


def df_from_file(file,decision_bounds=(-0.69,0.69),use_rt=True):
    # import Responses file
    print("Importing file: {}".format(file))
    R = Res(file)
    # create random initial Voronoi/Random priors
    print("\tGenerating voronoi prior...")
    vd = utils.generate_random_voronoi_prior(*R.image.shape[:2], R.kSeg, end_iter=0.05)
    print("\tGenerating random prior...")
    rand = utils.generate_random_prior(*R.image.shape[:2], R.kSeg)
    # segment the image in the Response file
    Voronoi = SM((0, R.image), mode="array")
    Random = SM((1, R.image), mode="array")
    model = "c"
    k = R.kSeg

    print("\tSegmenting Voronoi... ")
    Voronoi.fit_model(
        model=model,
        n_components=np.array([k]),
        layer_stop=1,
        keep=True,
        init=vd,
        init_eps=0.05,
    )
    print("\tSegmenting Random... ")
    Random.fit_model(
        model=model,
        n_components=np.array([k]),
        layer_stop=1,
        keep=True,
        init=rand,
        init_eps=0.1,
    )
    # extract probability maps
    pmaps = {
        "Voronoi": utils.get_SegMap_pmaps(Voronoi, R.kSeg),
        "Random": utils.get_SegMap_pmaps(Random, R.kSeg),
    }
    if use_rt:
        pairs = R.get_tested_pairs(all_pairs=False)
        rts = R.reactionTime
        responses = R.Response
    else:
        pairs = R.get_tested_pairs()
        rts = None
        responses = None

    df = pd.concat(
        [
            analysis.get_df(
                pairs,
                pmaps["Voronoi"],
                decision_bounds=(decision_bounds[0], decision_bounds[1]),
                reaction_times=rts,
                responses=responses,
                condition="voronoi",
            ),
            analysis.get_df(
                pairs,
                pmaps["Random"],
                decision_bounds=(decision_bounds[0], decision_bounds[1]),
                reaction_times=rts,
                responses=responses,
                condition="random",
            ),
        ],
        axis=0,
        ignore_index=True,
    )

    df["k"] = k
    for item in list(R.fileinfo.keys())[1:-1]:
        df[item] = R.fileinfo[item]

    return df


def run_trials(file, n_trials):
    dfs = []
    for i in range(n_trials):
        print("File: {} | Trial {}".format(file, i))
        print("\t")
        df = df_from_file(file)
        df["trial"] = i
        dfs.append(df)
        print("\t")
    out = pd.concat(dfs, axis=0, ignore_index=True)

    return out


def run_all_files(files, n_trials=10):
    out = pd.concat(
        [run_trials(file, n_trials) for file in files], axis=0, ignore_index=True
    )

    return out 

if __name__ == "__main__":
    print("Hello")

    files = ns(glob("data/*"))

    df = run_all_files(files)

    df.to_csv("reaction_time_responses_model_all_files.csv")