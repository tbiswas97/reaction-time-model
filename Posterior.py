import numpy as np
import import_utils as io
import Response
from scipy.special import gamma
from scipy.stats import entropy
import toolbox as tb
import pandas as pd


class Posterior:
    def __init__(self, ResponseObj, n_r=500):
        """
        Saves Student-T parameters from FlexMM
        """
        R = ResponseObj
        self.R = R
        self.points = R.points
        self.Model = R.Models[0]
        self.T = self.Model.means_t.shape[0]
        self.K = self.Model.means_t.shape[1]
        self.means_t = self.Model.means_t[:, :, 6]
        self.vars_t = self.Model.covars_t[:, :, 6, 6]
        self.dofs_t = self.Model.degrees_t

        self.n_r = n_r
        self.r = np.linspace(0, 10, self.n_r)
        self.sc = 1

    def set_spike_count_multiplier(self, c):
        self.sc = c

    def analytical_mean(self, point_idx, freeze_params=False, mix=True, _map=False):
        coord = self.points[point_idx]
        x = self.Model.data_pca[coord[0], coord[1], 6]
        beta = self.dofs_t / 2
        x = x - self.means_t
        gamma_term = gamma((self.dofs_t / 2) + 1) / gamma((self.dofs_t / 2) + 1 / 2)

        out = (((beta / x**2) + (1 / (2 * self.vars_t))) ** (-1 / 2)) * gamma_term

        if mix:
            pis = self.Model.weights_t[:, coord[0], coord[1], :]

            if freeze_params:
                out = (pis @ out.T)[:, -1]
            else:
                out = np.diag(pis @ out.T)
        else:
            if _map:
                pis = self.Model.weights_t[:, coord[0], coord[1], :]
                out = np.diag(out[:, pis.argmax(1)])

        return out * self.sc

    def _analytical_mean(self, coord, freeze_params=False, mix=True, _map=False):
        x = self.Model.data_pca[coord[0], coord[1], 6]
        beta = self.dofs_t / 2
        x = x - self.means_t
        gamma_term = gamma((self.dofs_t / 2) + 1) / gamma((self.dofs_t / 2) + 1 / 2)

        out = (((beta / x**2) + (1 / (2 * self.vars_t))) ** (-1 / 2)) * gamma_term

        if mix:
            pis = self.Model.weights_t[:, coord[0], coord[1], :]

            if freeze_params:
                out = (pis @ out.T)[:, -1]
            else:
                out = np.diag(pis @ out.T)
        else:
            if _map:
                pis = self.Model.weights_t[:, coord[0], coord[1], :]
                out = np.diag(out[:, pis.argmax(1)])

        return out * self.sc

    def _analytical_var(self, coord, freeze_params=False, mix=True, _map=False):
        x = self.Model.data_pca[coord[0], coord[1], 6]
        beta = self.dofs_t / 2
        x = x - self.means_t

        first_term = ((beta / x**2) + (1 / (2 * self.vars_t))) ** (-1)
        gamma_term1 = gamma((self.dofs_t / 2) + (3 / 2)) / gamma(
            (self.dofs_t / 2) + (1 / 2)
        )
        gamma_term2 = (
            gamma((self.dofs_t / 2) + 1) / gamma((self.dofs_t / 2) + 1 / 2)
        ) ** 2

        out = first_term * (gamma_term1 - (gamma_term2)) * (self.sc**2)

        if mix:
            means = self.analytical_mean(
                point_idx, freeze_params=freeze_params, mix=False
            )

            pis = self.Model.weights_t[:, coord[0], coord[1], :]

            if freeze_params:
                sq_means = (pis @ (means).T[:, -1]) ** 2
                means_sq = pis @ (means**2).T[:, -1]
                mean_vars = (pis @ out.T)[:, -1]
            else:
                sq_means = (np.diag(pis @ (means).T)) ** 2
                means_sq = np.diag(pis @ (means**2).T)
                mean_vars = np.diag(pis @ out.T)

            out = mean_vars + means_sq - sq_means
        else:
            if _map:
                pis = self.Model.weights_t[:, coord[0], coord[1], :]
                out = np.diag(out[:, pis.argmax(1)])

        return out

    def analytical_var(self, point_idx, freeze_params=False, mix=True, _map=False):
        coord = self.points[point_idx]
        x = self.Model.data_pca[coord[0], coord[1], 6]
        beta = self.dofs_t / 2
        x = x - self.means_t

        first_term = ((beta / x**2) + (1 / (2 * self.vars_t))) ** (-1)
        gamma_term1 = gamma((self.dofs_t / 2) + (3 / 2)) / gamma(
            (self.dofs_t / 2) + (1 / 2)
        )
        gamma_term2 = (
            gamma((self.dofs_t / 2) + 1) / gamma((self.dofs_t / 2) + 1 / 2)
        ) ** 2

        out = first_term * (gamma_term1 - (gamma_term2)) * (self.sc**2)

        if mix:
            means = self.analytical_mean(
                point_idx, freeze_params=freeze_params, mix=False
            )

            pis = self.Model.weights_t[:, coord[0], coord[1], :]

            if freeze_params:
                sq_means = (pis @ (means).T[:, -1]) ** 2
                means_sq = pis @ (means**2).T[:, -1]
                mean_vars = (pis @ out.T)[:, -1]
            else:
                sq_means = (np.diag(pis @ (means).T)) ** 2
                means_sq = np.diag(pis @ (means**2).T)
                mean_vars = np.diag(pis @ out.T)

            out = mean_vars + means_sq - sq_means
        else:
            if _map:
                pis = self.Model.weights_t[:, coord[0], coord[1], :]
                out = np.diag(out[:, pis.argmax(1)])

        return out

    def pointwise_posterior(self, point_idx, freeze_params=False):
        coord = self.points[point_idx]
        out = np.zeros((self.n_r, self.T))
        if freeze_params:
            vec = self._pointwise_posterior_t(point_idx, self.T - 1)
            for t in range(self.T):
                pis = self.Model.weights_t[t, coord[0], coord[1], :]
                out[:, t] = pis @ vec.T

        else:
            self.pis = []
            for t in range(self.T):
                vec = self._pointwise_posterior_t(point_idx, t)
                pis = self.Model.weights_t[t, coord[0], coord[1], :]
                self.pis.append(pis)
                out[:, t] = pis @ vec.T

        return out

    def _pointwise_posterior_t(self, point_idx, t):
        coord = self.points[point_idx]
        x = self.Model.data_pca[coord[0], coord[1], 6]

        p_r_cond_x = np.zeros((self.n_r, self.K))

        for k in range(self.K):
            p_r_cond_x[:, k] = np.asarray(
                [
                    self._f_r_cond_x(
                        r, x, self.means_t[t, k], self.vars_t[t, k], self.dofs_t[t, k]
                    )
                    for r in self.r
                ]
            )

        return p_r_cond_x

    def _coord_posterior(self, coord):
        out = np.zeros((self.n_r, self.T))
        for t in range(self.T):
            vec = self._coord_posterior_t(coord, t)
            pis = self.Model.weights_t[t, coord[0], coord[1], :]
            out[:, t] = pis @ vec.T

        return out

    def _coord_posterior_t(self, coord, t):
        x = self.Model.data_pca[coord[0], coord[1], 6]

        p_r_cond_x = np.zeros((self.n_r, self.K))

        for k in range(self.K):
            p_r_cond_x[:, k] = np.asarray(
                [
                    self._f_r_cond_x(
                        r, x, self.means_t[t, k], self.vars_t[t, k], self.dofs_t[t, k]
                    )
                    for r in self.r
                ]
            )

        return p_r_cond_x

    def _f_r_cond_x(self, r, x, _mu, _var, _dof):
        g = r
        alpha = _dof / 2
        beta = _dof / 2
        x = x - _mu

        g_term = (np.abs(g)) ** (2 * alpha)
        exp_constant = (beta / np.abs(x) ** 2) + (1 / (2 * _var))

        numer = (
            (exp_constant ** (alpha + 0.5)) * g_term * np.exp(-(g**2) * exp_constant)
        )

        denom = gamma(alpha + 0.5)

        y = numer / denom

        return y

    def get_pointwise_df(self):
        d = {}
        R = self.R

        d["k"] = [R.kSeg] * len(R.points) * self.T
        d["img"] = [R.fileinfo["img"]] * len(R.points) * self.T
        d["sub"] = [R.fileinfo["subject"]] * len(R.points) * self.T
        d["t"] = list(range(self.T)) * len(R.points)
        d["point_idx"] = np.asarray(range(len(R.points))).repeat(self.T)
        d["mu_mixed"] = np.concatenate(
            [
                self.analytical_mean(i, mix=True, _map=False)
                for i in range(len(R.points))
            ]
        )
        d["mu_MAP"] = np.concatenate(
            [
                self.analytical_mean(i, mix=False, _map=True)
                for i in range(len(R.points))
            ]
        )
        d["var_mixed"] = np.concatenate(
            [self.analytical_var(i, mix=True, _map=False) for i in range(len(R.points))]
        )
        d["var_MAP"] = np.concatenate(
            [self.analytical_var(i, mix=False, _map=True) for i in range(len(R.points))]
        )

        entropies = entropy(R.fit_pmap)
        entropy_flat = entropies.T.ravel()

        d["entropy_flat"] = entropy_flat.repeat(self.T)

        df = pd.DataFrame.from_dict(d)

        if "random" in R.filename:
            df["key"] = "random"
        elif "unsmooth" in R.filename:
            df["key"] = "unsmooth"
        else:
            df["key"] = "hmap"

        return df
