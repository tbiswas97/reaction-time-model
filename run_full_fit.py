import sys
import os

sys.path.append("/gs/gsfs0/users/tbiswas/bayes_seg/src")
import import_utils

pname = sys.argv[0]
homedir = str(sys.argv[1])
response_file = str(sys.argv[2])

filekey = response_file.split("/")[-1].split(".")[0]

print(f"Import {response_file}")
R = import_utils._load(response_file)

from Model import Model as Mod

for key in ["ai_both", "ei", "ei_wt_drift", "ei_wt_sp", "ei_wt_both"]:
    M = Mod(R,key=key)
    print(f"Fitting model {key}") 
    M.fit()

    df = M.get_df()

    df["model_key"] = key
    df["loss"] = M.opt_error[M.key]
    df["sub"] = M.subject
    df["img"] = M.img
    df["k"] = M.k
    df["layer"] = M.best_layer

    output_dir = os.path.join(homedir,"out")
    df.to_csv(os.path.join(output_dir, f"{filekey}_{key}_1_0_full_fit.csv"))
    
