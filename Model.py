import sys
import numpy as np
import seaborn as sns
import pandas as pd

sys.path.append("../src/")
import import_utils
import dynamics
import toolbox as tb

from scipy.optimize import basinhopping
from scipy.optimize import dual_annealing
from scipy.stats import iqr

from sklearn.model_selection import KFold


class Model:

    def __init__(
        self,
        ResponseObj_file,
        params_to_fit=["automult_2d", "automult", "logits", "ei_logits", "wei_logits"],
    ):
        """
        Collects only the output of segmentation probabilities per iteration of
        the Model. Image information is not included, but optimization functions
        are the same.

        Parameters:
        ------------
        ResponseObj : Response
            ResponseObj contains image information while ResultObj contains only
            numerical information.

        Attributes:
        ------------
        self.logits
        self.logits_deriv
        self.smooth_logits
        self.ei_logits
        self.wei_logits
        self.sfs_t
        """

        sns.set_context("talk")

        if type(ResponseObj_file) == str:
            assert ".pkl" in ResponseObj_file, "Can only handle pickled inputs"
            Response = import_utils._load(ResponseObj_file)
        else:
            Response = ResponseObj_file

        self.distances = np.asarray([Trial.distances for Trial in Response.Models])

        self.subject = Response.fileinfo["subject"]
        self.img = Response.fileinfo["img"]
        self.k = Response.kSeg

        self.logits = Response.logits
        self.logits_deriv = Response.logit_deriv
        self.smooth_logits = Response.smooth_logits
        self.ei_logits = Response.ei_logits
        self.wei_logits = Response.wei_logits
        self.sfs_t = Response.seg_flags

        self.human_rt = Response.reactionTime
        self.human_responses = Response.Response.astype("bool")
        self.human_seg_flag = Response.human_seg_flag

        logit_ig = iqr(self.logits.reshape(-1), rng=(1, 2))
        ei_logit_ig = iqr(self.ei_logits.reshape(-1), rng=(1, 2))
        wei_logit_ig = iqr(self.wei_logits.reshape(-1), rng=(1, 2))

        self.hyperparams = {
            "automult": {
                "x0": np.array([0.01]),
                "stepsize": 0.005,
                "T": 100,
                "niter_success": 100,
            },
            "logits": {"x0": logit_ig, "stepsize": 1, "T": 100, "niter_success": 100},
            "ei_logits": {
                "x0": ei_logit_ig,
                "stepsize": 1,
                "T": 100,
                "niter_success": 100,
            },
            "wei_logits": {
                "x0": wei_logit_ig,
                "stepsize": 1,
                "T": 0.001,
                "niter_success": 100,
            },
        }

        self.params_to_fit = params_to_fit

        self.rt_dict = {k: None for k in self.params_to_fit}

        self.opt_params = {}
        self.opt_error = {}

    def _sweep(
        self,
        value,
        param="automult",
        conv_failure="argmax",
        use_boundary=None,
        add_one=False,
    ):
        """
        Parameters:
        -----------
        value : float
        param : str
            "automult" : derivative parameter for FlexMM model
            "online_bound": bound parameter for FlexMM model
            "ei_bound": bound parameter for vanilla EI model
            "wei_bound" : bound parameter for weighted EI model

        Returns:
        ---------
        rts : array of reaction times
        """
        if "automult" in param:
            if use_boundary is not None:
                rts = dynamics._get_rt_from_deriv(
                    self.smooth_logits,
                    self.logits_deriv,
                    value,
                    return_mean=True,
                    failure_mode=conv_failure,
                    mean_axis=(0, -1),
                    use_boundary=use_boundary,
                    add_one=add_one,
                )
            else:
                rts = dynamics._get_rt_from_deriv(
                    self.smooth_logits,
                    self.logits_deriv,
                    value,
                    return_mean=True,
                    failure_mode=conv_failure,
                    mean_axis=(0, -1),
                    add_one=add_one,
                )
        if param == "logits":
            rts = dynamics._get_rt_from_boundary(
                self.logits,
                value,
                return_mean=True,
                output_flat=False,
                mean_axis=(0, -1),
                add_one=add_one,
            )
        elif param == "ei_logits":
            rts = dynamics._get_rt_from_boundary(
                self.ei_logits,
                value,
                return_mean=True,
                output_flat=False,
                mean_axis=(0, -1),
                add_one=add_one,
            )
        elif param == "wei_logits":
            rts = dynamics._get_rt_from_boundary(
                self.wei_logits,
                value,
                return_mean=True,
                output_flat=False,
                mean_axis=(0, -1),
                add_one=add_one,
            )
        return rts

    def loss(
        self, value, param, use_boundary=None, loss_type="mle", penalize_zeros=True
    ):
        if penalize_zeros:
            model_data_full = self._sweep(
                value, param, use_boundary=use_boundary, add_one=False
            )
        else:
            model_data_full = self._sweep(
                value, param, use_boundary=use_boundary, add_one=True
            )

        n_zeros = len(self.human_rt) - np.count_nonzero(model_data_full)
        if loss_type == "mle":
            n_zeros *= 1000

        human_data_full = self.human_rt
        log_hd_full = np.log(human_data_full)

        z_score = lambda x: (x - x.mean()) / x.std()

        z_hd = z_score(log_hd_full)

        log_hd_trunc = log_hd_full[z_hd < 1.282]
        model_data_trunc = model_data_full[z_hd < 1.282]
        human_responses_trunc = self.human_responses[z_hd < 1.282]

        if penalize_zeros:
            model_data = model_data_trunc[model_data_trunc != 0]
            h_log_data = log_hd_trunc[model_data_trunc != 0]
            human_responses = human_responses_trunc[model_data_trunc != 0]
        else:
            model_data = model_data_trunc
            h_log_data = log_hd_trunc
            human_responses = human_responses_trunc

        m_log_data = np.log(model_data)

        _max = np.max(m_log_data)

        model_resc = (m_log_data - np.min(m_log_data)) / np.ptp(m_log_data)
        human_resc = (h_log_data - np.min(h_log_data)) / np.ptp(h_log_data)

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
        param,
        loss_type="mle",
        annealing_step=True,
        penalize_zeros=True,
        verbose=False,
    ):
        if param != "automult_2d":
            loss_ = lambda x: self.loss(
                x,
                param=param,
                loss_type=loss_type,
                use_boundary=None,
                penalize_zeros=penalize_zeros,
            )
        else:
            pass
        cbf = lambda x, f, accept: True if (f < (1e-3)) and (accept) else False

        opt_res = basinhopping(
            loss_,
            **self.hyperparams[param],
            minimizer_kwargs={"bounds": [(1e-4, 20)]},
            disp=verbose,
            callback=cbf,
        )

        if annealing_step:
            if verbose:
                cbf_anneal = print
            else:
                cbf_anneal = None

            if param != "automult":
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
            else:
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

        self.opt_error[param] = res.fun
        self.opt_params[param] = res.x[0]

        self.is_fit_1d = True

    def _fit_param_2d(
        self,
        loss_type="mle",
        annealing_step=True,
        penalize_zeros=True,
        verbose=False,
        n_optimizations=2,
    ):

        self.n_2d_opts = n_optimizations

        assert self.is_fit_1d, "Must fit 1d params before 2d"

        loss_ = lambda x: self.loss(
            x[0],
            param="automult2d",
            loss_type=loss_type,
            use_boundary=x[1],
            penalize_zeros=penalize_zeros,
        )

        self.opt_params["automult_2d"] = np.array(
            [self.opt_params["automult"], self.opt_params["logits"] * 3]
        )

        cbf = lambda x, f, accept: True if (f < (1e-3)) and (accept) else False

        hyperparams = {
            "x0": self.opt_params["automult_2d"],
            "stepsize": 0.1,
            "T": 1,
            "niter_success": 100,
        }

        for i in range(n_optimizations):
            hyperparams = {
                "x0": self.opt_params["automult_2d"],
                "stepsize": 0.1,
                "T": 100,
                "niter_success": 100,
            }

            opt_res = basinhopping(
                loss_,
                **hyperparams,
                minimizer_kwargs={"bounds": [(1e-4, 1), (1e-4, 10)]},
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

            self.opt_params["automult_2d"] = res.x

        self.opt_params["automult_2d"] = res.x
        self.opt_error["automult_2d"] = res.fun

        self.is_fit_2d = True

    def fit(
        self,
        verbose="light",
        save_output=None,
        loss_type="mle",
        n_2d_opts=2,
        penalize_zeros=True,
    ):

        if verbose == "full":
            vf = True
        else:
            vf = False

        for param in self.params_to_fit[1:]:
            if verbose == "light":
                print("Fitting {}".format(param))
                print(self.opt_params)

            self._fit_param(
                param, loss_type=loss_type, verbose=vf, penalize_zeros=penalize_zeros
            )

        if verbose == "light":
            print("Fitting automult_2d")

        self._fit_param_2d(
            verbose=vf,
            loss_type=loss_type,
            n_optimizations=n_2d_opts,
            penalize_zeros=penalize_zeros,
        )

        if save_output is not None:
            filename = "sub_{}_img_{}_k_{}_fit_model.pkl".format(
                self.subject, self.img, self.k
            )
            savename = save_output + filename
            import_utils._pickle(self, savename)

    def plot_loss_histograms(self):
        pass


class CrossValidator(Model):

    def get_test_train_split(self, n_splits):
        self.n_splits = n_splits
        self.attr_to_split = [
            "logits",
            "logits_deriv",
            "smooth_logits",
            "ei_logits",
            "wei_logits",
            "sfs_t",
        ]

        self.attr_human = [
            "human_rt",
            "human_responses",
            "human_seg_flag",
        ]

        self.fold_params = {k: None for k in range(n_splits)}

        self.fold_train_loss = {
            k: {param: None for param in self.params_to_fit} for k in range(n_splits)
        }
        self.fold_test_loss = {
            k: {param: None for param in self.params_to_fit} for k in range(n_splits)
        }

        for attr in self.attr_to_split + self.attr_human:
            self.__dict__["parent_" + attr] = self.__dict__[attr]

        idxs = np.arange(self.parent_logits.shape[1])
        self.n_pairs = self.parent_logits.shape[1]

        kf = KFold(n_splits=n_splits)

        splits = kf.split(idxs)
        self.train_idxs = np.asarray([split[0] for split in splits])
        splits = kf.split(idxs)
        self.test_idxs = np.asarray([split[1] for split in splits])

    def set_opt_params(self, opt_params):
        self.parent_opt_params = opt_params
        for params in list(self.hyperparams.keys()):
            self.hyperparams[params]["x0"] = np.array([self.parent_opt_params[params]])

    def set_opt_error(self, opt_error):
        self.parent_opt_error = opt_error

    def _train(self, fold):
        train_idxs = self.train_idxs[fold]

        for attr in self.attr_to_split:
            self.__dict__[attr] = self.__dict__["parent_" + attr][:, train_idxs, ...]

        for attr in self.attr_human:
            self.__dict__[attr] = self.__dict__["parent_" + attr][train_idxs]

        self.fit()

        self.fold_params[fold] = self.opt_params
        self.fold_train_loss[fold] = self.opt_error

    def _test(self, fold):
        test_idxs = self.test_idxs[fold]

        for attr in self.attr_to_split:
            self.__dict__[attr] = self.__dict__["parent_" + attr][:, test_idxs, ...]

        for attr in self.attr_human:
            self.__dict__[attr] = self.__dict__["parent_" + attr][test_idxs]

        for param in self.params_to_fit:
            if param != "automult_2d":
                self.fold_test_loss[fold][param] = self.loss(
                    self.fold_params[fold][param], param
                )
            else:
                self.fold_test_loss[fold][param] = self.loss(
                    self.fold_params[fold][param][0],
                    param,
                    use_boundary=self.fold_params[fold][param][1],
                )

    def cross_val_summary(self):

        test_loss = (
            pd.DataFrame.from_dict(self.fold_test_loss)
            .reset_index()
            .melt(
                id_vars="index",
                value_vars=list(range(self.n_splits)),
                var_name="fold",
                value_name="test_loss",
            )
            .rename({"index": "model_type"}, axis=1)
        ).set_index(["model_type", "fold"])

        train_loss = (
            pd.DataFrame.from_dict(self.fold_train_loss)
            .reset_index()
            .melt(
                id_vars="index",
                value_vars=list(range(self.n_splits)),
                var_name="fold",
                value_name="train_loss",
            )
            .rename({"index": "model_type"}, axis=1)
        ).set_index(["model_type", "fold"])

        self.lkldf = test_loss.join(train_loss, ["model_type", "fold"])

        self.lkldf = self.lkldf.reset_index()

        self.lkldf["avg_test_loss"] = self.lkldf["test_loss"] / (
            self.n_pairs // self.n_splits
        )

        self.lkldf["avg_train_loss"] = self.lkldf["train_loss"] / (
            (self.n_pairs // self.n_splits) * (self.n_splits - 1)
        )

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

        self.lkldf["subject"] = self.subject
        self.lkldf["img"] = self.img
        self.lkldf["k"] = self.k
        return self.lkldf
