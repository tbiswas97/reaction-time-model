import sys
import os

sys.path.append("/gs/gsfs0/users/tbiswas/bayes_seg/src")
import import_utils

pname = sys.argv[0]
homedir = str(sys.argv[1])
response_file = str(sys.argv[2])
lut_name = str(sys.argv[3])
smoothing = int(sys.argv[4])
if smoothing == 1:
    smooth_key = "smooth"
else:
    smooth_key = "unsmooth"

random_init = int(sys.argv[5])
if random_init == 1:
    init_key = "random"
else:
    init_key = "hmap"

from Response import Response as Res

filekey = response_file.split("/")[-1].split(".")[0]

R = Res(homedir, response_file)
R.fit()
R.run_dynamics_model(
    n_trials=1,
    layer=0,
    smooth=smoothing,
    random_init=random_init,
    n_pseudocoords=10,
    n_pca=6,
    lut_name=lut_name,
)
import_utils._pickle(R, os.path.join(homedir, "out_bl9", f"{filekey}_Segmentation.pkl"))

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

    output_dir = os.path.join(homedir, "out_bl9")

    cv.lkldf.to_csv(
        os.path.join(
            output_dir, f"{filekey}_{key}_{smooth_key}_{init_key}_fit_params.csv"
        )
    )
    cv.data.to_csv(
        os.path.join(
            output_dir, f"{filekey}_{key}_{smooth_key}_{init_key}_test_data.csv"
        )
    )
