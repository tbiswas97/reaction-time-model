import sys
import os

sys.path.append("/gs/gsfs0/users/tbiswas/bayes_seg/src")
import import_utils

pname = sys.argv[0]
homedir = str(sys.argv[1])
response_file = str(sys.argv[2])
smoothing = int(sys.argv[3])
if smoothing == 1:
    smooth_key = "smooth"
else:
    smooth_key = "unsmooth"

random_init = int(sys.argv[4])
if random_init == 1:
    init_key = "random"
else:
    init_key = "hmap"

from Response import Response as Res

"""
a = "data/data_processing/sub_12508_exp1_session1_cat2_img3.mat_Segmentation.pkl"

In [5]: a.split("/")
Out[5]:
['data',
'data_processing',
'sub_12508_exp1_session1_cat2_img3.mat_Segmentation.pkl']

In [6]: a.split("/")[-1]
Out[6]: 'sub_12508_exp1_session1_cat2_img3.mat_Segmentation.pkl'

In [7]: a.split("/")[-1].split(".")
Out[7]: ['sub_12508_exp1_session1_cat2_img3', 'mat_Segmentation', 'pkl']

In [8]: a.split("/")[-1].split(".")[0]
Out[8]: 'sub_12508_exp1_session1_cat2_img3'
"""
filekey = response_file.split("/")[-1].split(".")[0]

R = Res(homedir, response_file)
R.fit()
R.run_dynamics_model(
    n_trials=1,
    layer=1,
    smooth=smoothing,
    random_init=random_init,
    n_pseudocoords=10,
    n_pca=6,
)
import_utils._pickle(R, os.path.join(homedir, "out_hpa_1", f"{filekey}_layer_1_Segmentation.pkl"))

import Model as M

for key in ["ai_both", "ei", "ei_wt_drift", "ei_wt_sp", "ei_wt_both"]:
    cv = M.CrossValidator(R, key=key)

    n_folds = 5
    cv.get_test_train_split(n_folds)
    for i in range(n_folds):
        cv._train(i)
        cv._test(i)
    cv.cross_val_summary()

    cv.data["smooth_key"] = smooth_key
    cv.data["init_key"] = init_key

    cv.lkldf["smooth_key"] = smooth_key
    cv.lkldf["init_key"] = init_key

    output_dir = os.path.join(homedir, "out_hpa_1")

    cv.lkldf.to_csv(
        os.path.join(
            output_dir, f"{filekey}_{key}_{smooth_key}_{init_key}_layer_1_fit_params.csv"
        )
    )
    cv.data.to_csv(
        os.path.join(
            output_dir, f"{filekey}_{key}_{smooth_key}_{init_key}_layer_1_test_data.csv"
        )
    )
