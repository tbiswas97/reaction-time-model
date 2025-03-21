import sys
import numpy as np

sys.path.append("../src/")
import import_utils
import dynamics
import toolbox as tb

from scipy.optimize import basinhopping
from scipy.optimize import dual_annealing


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

        assert ".pkl" in ResponseObj_file, "Can only handle pickled inputs"

        Response = import_utils._load(ResponseObj_file)

        self.logits = Response.logits
        self.logits_deriv = Response.logit_deriv
        self.smooth_logits = Response.smooth_logits
        self.ei_logits = Response.ei_logits
        self.wei_logits = Response.wei_logits
        self.sfs_t = Response.sfs_t

        self.human_rt = Response.reactionTime
        self.human_responses = Response.Response.astype("bool")
        self.hyperparams = {
            "automult": {
                "x0": np.array([0.01]),
                "stepsize": 0.005,
                "T": 100,
                "niter_success": 100,
            },
            "online_rt": {"x0": 5, "stepsize": 0.1, "T": 100, "niter_success": 100},
            "ei_rt": {"x0": 5, "stepsize": 0.1, "T": 100, "niter_success": 100},
            "wei_rt": {"x0": 2, "stepsize": 1, "T": 0.001, "niter_success": 100},
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
        pass
        if param == "automult":
            if use_boundary is not None:
                rts = dynamics._get_rt_from_deriv(
                    self.smooth_logits,
                    self.logit_deriv,
                    value,
                    return_mean=True,
                    failure_mode=conv_failure,
                    mean_axis=(0, -1),
                    use_boundary=use_boundary,
                )
            else:
                rts = dynamics._get_rt_from_deriv(
                    self.smooth_logits,
                    self.logit_deriv,
                    value,
                    return_mean=True,
                    failure_mode=conv_failure,
                    mean_axis=(0, -1),
                )
        if param == "logits":
            rts = dynamics._get_rt_from_boundary(
                self.logits,
                value,
                return_mean=True,
                output_flat=False,
                mean_axis=(0, -1),
            )
        elif param == "ei_logits":
            rts = dynamics._get_rt_from_boundary(
                self.ei_logits,
                value,
                return_mean=True,
                output_flat=False,
                mean_axis=(0, -1),
            )
        elif param == "wei_logits":
            rts = dynamics._get_rt_from_boundary(
                self.wei_logits,
                value,
                return_mean=True,
                output_flat=False,
                mean_axis=(0, -1),
            )
        return rts

    def loss(
        self, value, param, use_boundary=None, loss_type="mle", penalize_zeros=True
    ):
        model_data_full = self._sweep(value, param, use_boundary=use_boundary)

        n_zeros = len(self.human_rt) - np.count_nonzero(model_data_full)
        if loss_type == "mle":
            n_zeros *= 1000

        human_data_full = self.human_rt
        log_hd_full = np.log(human_data_full)

        z_score = lambda x: (x - x.mean()) / x.std()

        z_hd = z_score(log_hd_full)

        log_hd_trunc = log_hd_full[z_hd < 1.282]
        model_data_trunc = model_data[z_hd < 1.282]
        human_responses_trunc = self.human_responses[z_hd < 1.282]

        model_data = model_data_trunc[model_data_trunc != 0]
        h_log_data = log_hd_trunc[model_data_trunc != 0]
        human_responses = human_responses_trunc[model_data_trunc != 0]

        m_log_data = np.log(model_data)

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
            minimizer_kwargs={"bounds": [(1e-4, 10)]},
            disp=verbose,
            callback=cbf
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

        assert self.is_fit_1d, "Must fit 1d params before 2d"

        loss_ = lambda x: self.loss(
            x[0],
            param="automult2d",
            loss_type=loss_type,
            use_boundary=x[1],
            penalize_zeros=penalize_zeros,
        )

        self.opt_params["automult_2d"] = np.array(
            [self.opt_params["automult"], self.opt_params["logits"]]
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
                "T": 1,
                "niter_success": 100,
            }

            opt_res = basinhopping(
                loss_,
                hyperparams,
                minimizer_kwargs={"bounds": [(1e-4, 10)]},
                disp=verbose,
                callback=cbf,
            )

            self.opt_params["automult_2d"] = opt_res.x

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
        self.opt_error["automult_2d"] = res.fun

    def fit(self, verbose="light"):

        if verbose == "full":
            vf = True

        for param in self.params_to_fit[1:]:
            if verbose == "light":
                print("Fitting {}".format(param))

            self._fit_param(param, verbose=vf)

        self._fit_param_2d(verbose=vf)
