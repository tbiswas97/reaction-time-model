import sys
import os

sys.path.append("/gs/gsfs0/users/tbiswas/bayes_seg/src")
import import_utils

pname = sys.argv[0]
homedir = str(sys.argv[1])
response_file = str(sys.argv[2])

print(f"Import {response_file}")
R = import_utils._load(response_file)

filekey = R.filekey

smooth_key = "smooth"

init_key = "hmap"

import Model as M

for key in ["ai_b", "ai_lambda"]:
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

    output_dir = os.path.join(homedir, "out_bl5_hpa_smooth_fits_singlerules")

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
