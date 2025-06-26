import sys
import numpy as np
import seaborn as sns
import pandas as pd

sys.path.append("../src/")
import import_utils
import dynamics
import toolbox as tb
from numpy.lib.stride_tricks import sliding_window_view
from matplotlib import pyplot as plt

from scipy.optimize import basinhopping
from scipy.optimize import dual_annealing
from scipy.stats import iqr

from sklearn.model_selection import KFold

# TODO: deprecate penalize zeros


class Model:
    def __init__(self, ResponseObj_file, key="ai_lambda"):
        """
        Collects only the output of segmentation probabilities per iteration of
        the Model. Image information is not included, but optimization functions
        are the same.

        This class contains 7 types of models:

        - Approx. inference model
            - fit with $b$ "ai_b"
            - fit with $\lambda$ "ai_lambda"
            - >>> fit with both "ai_both"
        - >>> Standard evidence integration model "ei"
        - >>> Weighted evidence integration models:
            - >>> Weighted starting point "ei_wt_drift"
            - >>> Weighted drift "ei_wt_sp"

        Parameters:
        ------------
        ResponseObj : Response
            ResponseObj contains image information while ResultObj contains only
            numerical information.

        Attributes:
        ------------
        self.key : str
            The type of model that is being fit
        """

        if type(ResponseObj_file) == str:
            assert ".pkl" in ResponseObj_file, "Can only handle pickled inputs"
            Response = import_utils._load(ResponseObj_file)
        else:
            Response = ResponseObj_file

        self.key = key
        if "ei" in self.key:
            self.fit_dim = 2
        elif self.key == "ai_both":
            self.fit_dim = 2
        else:
            self.fit_dim = 1

        self.distances = np.asarray([Trial.distances for Trial in Response.Models])
        self.subject = Response.fileinfo["subject"]
        self.img = Response.fileinfo["img"]
        self.k = Response.kSeg
        self.best_layer = Response.best_layer

        self.logits = Response.logits
        self.sfs_t = Response.seg_flags
        if (self.key == "ai_both") or (self.key == "ai_lambda"):
            self.smooth_logits = sliding_window_view(self.logits, 3, axis=-1).mean(-1)
            diff = lambda x: (x[-1] - x[0]) / len(x)
            self.logit_deriv = np.apply_along_axis(
                diff, -1, sliding_window_view(self.logits, 3, axis=-1)
            )
        num_samples = self.logits.shape[-1]
        self.flat_logits = self.logits.reshape((-1, num_samples))
        self.noise_struct = np.random.normal(0, 1, size=self.logits.shape)

        self.human_rt = Response.reactionTime
        self.human_responses = Response.Response.astype("bool")
        self.human_seg_flag = Response.human_seg_flag

        if "ai" in self.key:
            ai_initial_guess = iqr(self.logits.reshape(-1), rng=(1, 2))
            self.hyperparams = {
                "ai_lambda": {
                    "x0": np.array([0.01]),
                    "stepsize": 0.005,
                    "T": 100,
                    "niter_success": 100,
                },
                "ai_b": {
                    "x0": np.array([ai_initial_guess]),
                    "stepsize": 1,
                    "T": 100,
                    "niter_success": 100,
                },
                "ai_both": {
                    "x0": np.array([50, 0.01]),
                    "stepsize": 0.1,
                    "T": 1,
                    "niter_success": 100,
                },
            }
        elif "ei" in self.key:
            self.int_noisy_evidence(1, 5)
            ei_initial_guess = iqr(self.evidence.reshape(-1), rng=(1, 2))
            self.hyperparams = {
                "ei": {
                    "x0": np.array([ei_initial_guess, 1]),
                    "stepsize": 1,
                    "T": 100,
                    "niter_success": 100,
                },
            }
            if "wt" in self.key:
                self.int_noisy_evidence(1, 10)
                ei_initial_guess = iqr(self.evidence.reshape(-1), rng=(1, 2))
                self.hyperparams = {
                    "ei_wt_drift": {
                        "x0": np.array([ei_initial_guess, 1]),
                        "stepsize": 1,
                        "T": 0.001,
                        "niter_success": 100,
                    },
                    "ei_wt_sp": {
                        "x0": np.array([ei_initial_guess, 1]),
                        "stepsize": 1,
                        "T": 0.001,
                        "niter_success": 100,
                    },
                    "ei_wt_both": {
                        "x0": np.array([ei_initial_guess, 1]),
                        "stepsize": 1,
                        "T": 0.001,
                        "niter_success": 100,
                    },
                }

        self.opt_params = {}
        self.opt_error = {}

    # make evidence integration
    def int_noisy_evidence(self, lam, noise):
        """
        Numerically integrates the DDM

        Parameters:
        ------------
        lam : float
            The drift gain
        noise : float
            The noise gain (multiplied to Wiener Process)
        """
        num_samples = self.logits.shape[-1]
        self.noise_arr = self.noise_struct.reshape((-1, num_samples))
        flat_logits = self.logits.reshape((-1, num_samples))
        flat_sfs_t = self.sfs_t.reshape((-1, num_samples))
        canvas = np.zeros(flat_logits.shape)

        # standard ei
        if "ei" in self.key:
            drift_rate_arr = flat_logits[:, -1] / num_samples
            # weighted drift
            if self.key == "ei_wt_drift":
                temp = drift_rate_arr[..., np.newaxis] + noise * self.noise_arr
                # Numerical integration step
                temp = np.cumsum(temp, axis=1)
                temp *= lam
                canvas[:, 1:] = temp[:, :-1]
            # weighted prior and weighted drift
            elif self.key == "ei_wt_both":
                temp = drift_rate_arr[..., np.newaxis] + noise * self.noise_arr
                temp = np.cumsum(temp, axis=1)
                temp *= lam
                canvas[1:, :] = temp[:-1, :]
                canvas = canvas + drift_rate_arr[..., np.newaxis]
            # standard ei
            elif (self.key == "ei") or (self.key == "ei_wt_sp"):
                pos_drift_rate = np.mean(flat_logits[:, -1][flat_sfs_t[:, -1]])
                neg_drift_rate = np.mean(flat_logits[:, -1][~flat_sfs_t[:, -1]])
                self.global_drift_rate = [pos_drift_rate, neg_drift_rate]
                decision_drift = np.zeros(flat_logits[:, -1].shape)
                decision_drift[flat_sfs_t[:, -1]] += pos_drift_rate
                decision_drift[~flat_sfs_t[:, -1]] += neg_drift_rate
                decision_drift = decision_drift[:, np.newaxis] / num_samples
                temp = decision_drift + noise * self.noise_arr
                temp = np.cumsum(temp, axis=1)
                temp = temp * lam
                canvas[:, 1:] = temp[:, :-1]
                # standard ei and weighted prior
                if self.key == "ei_wt_sp":
                    # canvas = canvas + drift_rate_arr
                    canvas += drift_rate_arr[..., np.newaxis]

            evidence = canvas.reshape(self.logits.shape)

        self.evidence = evidence
        # starting point weighted

        # drift weighted
        # both weighted

    def _sweep(
        self,
        values,
        conv_failure="argmax",
        add_one=True,
    ):
        """
        Parameters:
        -----------
        values : list
            Order is [$b$,$\lambda$]
        Returns:
        """
        if "ei" in self.key:
            self.int_noisy_evidence(values[1], 10)
            self.rts_struct = dynamics._get_rt_from_boundary(
                self.evidence,
                values[0],
                return_mean=False,
                output_flat=False,
                mean_axis=(0, -1),
                add_one=add_one,
            )

            rts = self.rts_struct.mean(axis=(0, -1))
        elif self.key == "ai_b":
            self.evidence = self.logits
            self.rts_struct = dynamics._get_rt_from_boundary(
                self.evidence,
                values,
                return_mean=False,
                output_flat=False,
                mean_axis=(0, -1),
                add_one=add_one,
            )

            rts = self.rts_struct.mean(axis=(0, -1))
        elif (self.key == "ai_lambda") or (self.key == "ai_both"):
            if self.key == "ai_lambda":
                self.rts_struct = dynamics._get_rt_from_deriv(
                    self.smooth_logits,
                    self.logit_deriv,
                    values,
                    output_flat=False,
                    return_mean=False,
                    failure_mode=conv_failure,
                    mean_axis=(0, -1),
                    use_boundary=None,
                    add_one=add_one,
                )
                rts = self.rts_struct.mean(axis=(0, -1))
            if self.key == "ai_both":
                self.rts_struct = dynamics._get_rt_from_deriv(
                    self.smooth_logits,
                    self.logit_deriv,
                    values[1],
                    output_flat=False,
                    return_mean=False,
                    failure_mode=conv_failure,
                    mean_axis=(0, -1),
                    use_boundary=values[0],
                    add_one=add_one,
                )
                rts = self.rts_struct.mean(axis=(0, -1))

        return rts

    def loss(self, values, loss_type="mle", penalize_zeros=False):
        if penalize_zeros:
            model_data_full = self._sweep(values, add_one=False)
        else:
            model_data_full = self._sweep(values, add_one=True)

        n_zeros = len(self.human_rt) - np.count_nonzero(model_data_full)
        if loss_type == "mle":
            n_zeros *= 1000

        human_data_full = self.human_rt
        log_hd_full = np.log(human_data_full)

        z_score = lambda x: (x - x.mean()) / x.std()

        z_hd = z_score(log_hd_full)

        self.z_exc_idxs = z_hd < 1.282

        log_hd_trunc = log_hd_full[z_hd < 1.282]
        model_data_trunc = model_data_full[z_hd < 1.282]
        human_responses_trunc = self.human_responses[z_hd < 1.282]
        self.td_hat = model_data_trunc

        if penalize_zeros:
            model_data = model_data_trunc[model_data_trunc != 0]
            h_log_data = log_hd_trunc[model_data_trunc != 0]
            human_responses = human_responses_trunc[model_data_trunc != 0]
        else:
            model_data = model_data_trunc
            h_log_data = log_hd_trunc
            human_responses = human_responses_trunc

        m_log_data = np.log(model_data)

        model_resc = (m_log_data - np.min(m_log_data)) / np.ptp(m_log_data)
        human_resc = (h_log_data - np.min(h_log_data)) / np.ptp(h_log_data)

        self.log_td_hat = model_resc
        self.log_td = human_resc

        model_split = {
            True: model_resc[human_responses],
            False: model_resc[~human_responses],
        }

        human_split = {
            True: human_resc[human_responses],
            False: human_resc[~human_responses],
        }

        if loss_type == "mle":
            lkls_split = {
                True: tb.emp_lkl_mkii(model_split[True], human_split[True]),
                False: tb.emp_lkl_mkii(model_split[False], human_split[False]),
            }

            lkls = np.zeros(model_resc.shape)
            lkls[human_responses] = lkls_split[True]
            lkls[~human_responses] = lkls_split[False]
            self.lkls = lkls

            terms = []
            for i in [True, False]:
                terms.append(lkls_split[i])

        elif loss_type == "mse":
            terms = []
            for i in [True, False]:
                terms.append((np.mean(human_split[i]) - np.mean(model_split[i])) ** 2)

        term1 = sum(terms)

        if penalize_zeros:
            loss = term1 + n_zeros
        else:
            loss = term1

        return loss

    def _fit_param(
        self,
        loss_type="mle",
        annealing_step=True,
        penalize_zeros=False,
        verbose=False,
    ):

        loss_ = lambda x: self.loss(
            x,
            loss_type=loss_type,
            penalize_zeros=penalize_zeros,
        )

        cbf = lambda x, f, accept: True if (f < (1e-3)) and (accept) else False

        opt_res = basinhopping(
            loss_,
            **self.hyperparams[self.key],
            minimizer_kwargs={"bounds": [(1e-4, 20)]},
            disp=verbose,
            callback=cbf,
        )

        if annealing_step:
            if verbose:
                cbf_anneal = print
            else:
                cbf_anneal = None

            if self.key == "ai_b":
                optt_res = dual_annealing(
                    loss_,
                    [
                        (
                            opt_res.x[0] - 0.5 * opt_res.x[0],
                            opt_res.x[0] + 0.5 * opt_res.x[0],
                        )
                    ],
                    x0=opt_res.x,
                    callback=cbf_anneal,
                    maxiter=100,
                )
            elif self.key == "ai_lambda":
                optt_res = dual_annealing(
                    loss_,
                    [
                        (
                            opt_res.x[0] - 0.1 * opt_res.x[0],
                            opt_res.x[0] + 0.1 * opt_res.x[0],
                        )
                    ],
                    x0=opt_res.x,
                    callback=print,
                    maxiter=20,
                )
            res = optt_res
        else:
            res = opt_res

        self.opt_error[self.key] = res.fun
        self.opt_params[self.key] = res.x[0]

    def _fit_param_2d(
        self,
        loss_type="mle",
        annealing_step=True,
        penalize_zeros=False,
        verbose=False,
        n_optimizations=2,
    ):
        self.n_2d_opts = n_optimizations

        loss_ = lambda x: self.loss(
            x,
            loss_type=loss_type,
            penalize_zeros=penalize_zeros,
        )

        cbf = lambda x, f, accept: True if (f < (1e-3)) and (accept) else False

        if self.key == "ai_both":
            bound_dict = {"bounds": [(1e-4, 30), (1e-4, 1)]}
        elif "ei" in self.key:
            bound_dict = {"bounds": [(1e-4, 50), (1e-4, 10)]}

        for i in range(n_optimizations):
            if i > 0:
                self.hyperparams[self.key]["x0"] = res.x
            opt_res = basinhopping(
                loss_,
                **self.hyperparams[self.key],
                minimizer_kwargs=bound_dict,
                disp=verbose,
                callback=cbf,
            )

            if annealing_step:
                optt_res = dual_annealing(
                    loss_,
                    [
                        (
                            opt_res.x[0] - 0.5 * opt_res.x[0],
                            opt_res.x[0] + 0.5 * opt_res.x[0],
                        ),
                        (
                            opt_res.x[1] - 0.5 * opt_res.x[1],
                            opt_res.x[1] + 0.5 * opt_res.x[1],
                        ),
                    ],
                    x0=opt_res.x,
                    callback=print,
                    maxiter=100,
                )

                res = optt_res
            else:
                res = opt_res

            self.opt_params[self.key] = res.x

        self.opt_params[self.key] = res.x
        self.opt_error[self.key] = res.fun

    def fit_ai(self, verbose=True):
        ig_2d = np.array([0.0, 0.0])

        self.key = "ai_b"
        self._fit_param(verbose=verbose)
        ig_2d[0] = self.opt_params[self.key]

        self.key = "ai_lambda"
        self._fit_param(verbose=verbose)
        ig_2d[1] = self.opt_params[self.key]

        self.key = "ai_both"
        self.hyperparams[self.key]["x0"] = ig_2d
        self._fit_param_2d(verbose=True)

    def fit(
        self,
        verbose="light",
        save_output=None,
        loss_type="mle",
        n_2d_opts=2,
        penalize_zeros=False,
    ):

        if verbose == "full":
            vf = True
        else:
            vf = False

        if self.fit_dim == 2:
            kwargs = {
                "verbose": verbose,
                "loss_type": loss_type,
                "n_optimizations": n_2d_opts,
            }
            self._fit_param_2d(**kwargs)
        elif self.fit_dim == 1:
            kwargs = {
                "verbose": verbose,
                "loss_type": loss_type,
            }
            self._fit_param(**kwargs)

        if verbose == "light":
            print("Fitting {}".format(self.key))

        if save_output is not None:
            filename = "sub_{}_img_{}_k_{}_fit_model.pkl".format(
                self.subject, self.img, self.k
            )
            savename = save_output + filename
            import_utils._pickle(self, savename)

    def _get_model_response(self):
        responses = dynamics._get_responses_from_rt_arr(
            self.rts_struct - 1, self.logits
        )
        self.responses_struct = responses
        self.model_responses = np.mean(responses, axis=(0, -1))

    def get_df(self):
        self._get_model_response()
        binarize = lambda x: True if x > 0.5 else False
        binarize = np.vectorize(binarize)
        d = {
            "model_rt": self.log_td_hat,
            "human_rt": self.log_td,
            "image_distance": self.distances[0, :, 0][self.z_exc_idxs],
            "human_response": self.human_responses.astype("float")[self.z_exc_idxs],
            "human_error": (
                np.logical_xor(
                    self.human_responses.astype("bool")[self.z_exc_idxs],
                    self.human_seg_flag.astype("bool")[self.z_exc_idxs],
                )
            ).astype("float"),
            "model_response": self.model_responses[self.z_exc_idxs],
            "model_bool": binarize(self.model_responses[self.z_exc_idxs]),
        }

        df = pd.DataFrame.from_dict(d)

        return df

    def plot_loss(self):
        fig, axs = plt.subplots(
            nrows=1, ncols=1, sharey=False, sharex=False, figsize=(10, 10)
        )
        kwargs_model = {
            "histtype": "step",
            "linewidth": 6,
            "alpha": 0.9,
            "density": True,
            "ec": "black",
            "alpha": 0.7,
        }

        kwargs_human = {
            "histtype": "step",
            "linewidth": 6,
            "ec": "red",
            "density": True,
            "alpha": 0.5,
        }

        axs.hist(self.log_td_hat, **kwargs_model, label="model")
        axs.hist(self.log_td, **kwargs_human, label="human")
        axs.legend(fontsize=20)
        axs.grid(visible=True, axis="both", which="both")
        axs.tick_params(labelsize=15)

        for axis in ["top", "bottom", "left", "right"]:
            axs.spines[axis].set_linewidth(3)

        axs.set_ylabel("density", fontsize=20)
        axs.set_xlabel("log(decision time) - rescaled", fontsize=20)


class CrossValidator(Model):

    def get_test_train_split(self, n_splits):
        self.dfs = []
        self.n_splits = n_splits
        if (self.key == "ai_lambda") or (self.key == "ai_both"):
            self.attr_to_split = ["logits", "logit_deriv", "smooth_logits", "distances"]
        elif "ei" in self.key:
            self.attr_to_split = ["logits", "sfs_t", "distances", "noise_struct"]
        else:
            self.attr_to_split = ["logits", "sfs_t", "distances"]

        self.attr_human = [
            "human_rt",
            "human_responses",
            "human_seg_flag",
        ]

        self.fold_params = {k: None for k in range(n_splits)}

        self.fold_train_loss = {k: None for k in range(n_splits)}

        self.fold_test_loss = {k: None for k in range(n_splits)}

        for attr in self.attr_to_split + self.attr_human:
            self.__dict__["parent_" + attr] = self.__dict__[attr]

        idxs = np.arange(self.parent_logits.shape[1])
        self.n_pairs = self.parent_logits.shape[1]

        kf = KFold(n_splits=n_splits)

        splits = kf.split(idxs)
        self.train_idxs = np.asarray([split[0] for split in splits])
        splits = kf.split(idxs)
        self.test_idxs = np.asarray([split[1] for split in splits])

    def _train(self, fold):
        train_idxs = self.train_idxs[fold]

        for attr in self.attr_to_split:
            self.__dict__[attr] = self.__dict__["parent_" + attr][:, train_idxs, ...]

        for attr in self.attr_human:
            self.__dict__[attr] = self.__dict__["parent_" + attr][train_idxs]

        self.fit()

        self.fold_params[fold] = self.opt_params[self.key]
        self.fold_train_loss[fold] = [self.opt_error[self.key]]

    def _test(self, fold):
        test_idxs = self.test_idxs[fold]

        for attr in self.attr_to_split:
            self.__dict__[attr] = self.__dict__["parent_" + attr][:, test_idxs, ...]

        for attr in self.attr_human:
            self.__dict__[attr] = self.__dict__["parent_" + attr][test_idxs]

        self.fold_test_loss[fold] = [self.loss(self.fold_params[fold])]

        self.dfs.append(self.get_df())

    def cross_val_summary(self):

        data = pd.concat(self.dfs, axis=0, ignore_index=True)

        self.data = data
        self.data["model_key"] = self.key
        self.data["subject"] = self.subject
        self.data["img"] = self.img
        self.data["k"] = self.k
        self.data["best_layer"] = self.best_layer

        test_loss = (
            pd.DataFrame.from_dict(self.fold_test_loss)
            .reset_index()
            .melt(
                id_vars="index",
                value_vars=list(range(self.n_splits)),
                var_name="fold",
                value_name="test_loss",
            )
            .rename({"index": "model_key"}, axis=1)
        ).set_index(["model_key", "fold"])

        train_loss = (
            pd.DataFrame.from_dict(self.fold_train_loss)
            .reset_index()
            .melt(
                id_vars="index",
                value_vars=list(range(self.n_splits)),
                var_name="fold",
                value_name="train_loss",
            )
            .rename({"index": "model_key"}, axis=1)
        ).set_index(["model_key", "fold"])

        self.lkldf = test_loss.join(train_loss, ["model_key", "fold"])

        self.lkldf = self.lkldf.reset_index()

        self.lkldf["avg_test_loss"] = self.lkldf["test_loss"] / (
            self.n_pairs // self.n_splits
        )

        self.lkldf["avg_train_loss"] = self.lkldf["train_loss"] / (
            (self.n_pairs // self.n_splits) * (self.n_splits - 1)
        )

        self.lkldf["model_key"] = self.key
        for idx in range(self.n_splits):
            self.lkldf.loc[idx, "lambda"] = self.fold_params[idx][1]
            self.lkldf.loc[idx, "b"] = self.fold_params[idx][0]

        self.lkldf["subject"] = self.subject
        self.lkldf["img"] = self.img
        self.lkldf["k"] = self.k
        self.lkldf["best_layer"] = self.best_layer

        """
        self.lkldf["param_bound"] = None
        self.lkldf["param_deriv"] = None
        for param in self.params_to_fit:
            if "logits" in param:
                for fold in range(self.n_splits):
                    if self.fold_params[fold] is not None:
                        self.lkldf.loc[
                            (self.lkldf.model_type == param)
                            & (self.lkldf.fold == fold),
                            "param_bound",
                        ] = self.fold_params[fold][param]
            elif param == "automult":
                for fold in range(self.n_splits):
                    if self.fold_params[fold] is not None:
                        self.lkldf.loc[
                            (self.lkldf.model_type == "automult")
                            & (self.lkldf.fold == fold),
                            "param_deriv",
                        ] = self.fold_params[fold][param]
            elif param == "automult_2d":
                for fold in range(self.n_splits):
                    if self.fold_params[fold] is not None:
                        self.lkldf.loc[
                            (self.lkldf.model_type == "automult_2d")
                            & (self.lkldf.fold == fold),
                            "param_deriv",
                        ] = self.fold_params[fold][param][0]
                        self.lkldf.loc[
                            (self.lkldf.model_type == "automult_2d")
                            & (self.lkldf.fold == fold),
                            "param_bound",
                        ] = self.fold_params[fold][param][1]

        self.lkldf["parent_loss"] = None
        for param in self.params_to_fit:
            self.lkldf.loc[(self.lkldf.model_type == param), "parent_loss"] = (
                self.parent_opt_error[param]
            )

        self.lkldf["avg_parent_loss"] = self.lkldf["parent_loss"] / self.n_pairs

        self.lkldf["parent_param_bound"] = None
        self.lkldf["parent_param_deriv"] = None
        for param in self.params_to_fit:
            if "logits" in param:
                self.lkldf.loc[
                    (self.lkldf.model_type == param), "parent_param_bound"
                ] = self.parent_opt_params[param]
            elif param == "automult":
                self.lkldf.loc[
                    (self.lkldf.model_type == "automult"), "parent_param_deriv"
                ] = self.parent_opt_params[param]
            elif param == "automult_2d":
                self.lkldf.loc[
                    (self.lkldf.model_type == "automult_2d"), "parent_param_deriv"
                ] = self.parent_opt_params[param][0]
                self.lkldf.loc[
                    (self.lkldf.model_type == "automult_2d"), "parent_param_bound"
                ] = self.parent_opt_params[param][1]

        self.lkldf["k"] = self.k
        return self.lkldf
        """
