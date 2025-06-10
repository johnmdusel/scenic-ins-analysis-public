from itertools import product
from pathlib import Path
from typing import Iterable

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import scipy

from src.analysis import McmcAnalysis
from src.effects_comparisons import posterior_summary_and_effects
import src.prior as prior
from src.hpd import hpd

pytensor.config.floatX = "float32"
pytensor.config.blas__ldflags = "-llapack -lblas -lcblas"


class PcAnalysis(McmcAnalysis):
    """
    Analysis of empirical or predicted proportion correct.
    Subclasses `McmcAnalysis`.

    Methods
    -------
    run_mcmc
        Perform MCMC analysis.
        Runtime for default MCMC settings is tens-of-minutes
        on DELL XPS 15 9500 (i7-10875H CPU @ 2.30GHz x 16, 64 GiB memory).
    analyze_posterior
        See description for superclass.
    credibility_analysis
        Computes 95% HDI for estimated proportion correct in each joint level of
        experimental manipulation. Reports percentage of empirical p(c) lying
        in the HDI.

    Notes
    -----
    All paths/filenames are set in `config.yaml`.
    """

    def __init__(
        self,
        savedir: Path,
        rng: np.random.Generator,
        pc_data_path: Path,
        rt_response_data_path: Path,
        prior_data_paths: Iterable[Path],
        prior_plot_name: str | None
    ):
        super().__init__(savedir, rng)
        self.pc_data_path = pc_data_path
        self.rt_data_path = rt_response_data_path
        self.prior_data_paths = prior_data_paths
        self.prior_plot_name = prior_plot_name

    def run_mcmc(
        self, output_mcmc_chains_filename: str, output_mcmc_summary_filename: str
    ):
        df = pd.read_csv(self.pc_data_path)
        x1 = np.fromiter(
            map(lambda lvl: {"AA": 0, "AB": 1}[lvl], df["Cond"]), dtype=int
        )
        x2 = np.fromiter(
            map(lambda lvl: {"Acc": 0, "Spd": 1}[lvl], df["Ins"]), dtype=int
        )
        xSubj = df["participant_id"].to_numpy()
        y = df["pc"].to_numpy() - 0.0001

        Nx1Lvl, Nx2Lvl, NxSubjLvl = [len(np.unique(x)) for x in (x1, x2, xSubj)]

        prior_params = prior.get_pc_analysis_prior(
            self.prior_data_paths, self.rng, self.savedir, self.prior_plot_name
        )

        with pm.Model() as model:
            a0 = pm.Normal(
                name="a0",
                mu=prior_params["a0"]["mu"],
                sigma=prior_params["a0"]["sigma"],
            )
            sigma2_a1 = pm.InverseGamma(name="a1Var",
                                    alpha=prior_params["a1Var"]["alpha"],
                                    beta=prior_params["a1Var"]["beta"], )
            a1 = pm.Normal(
                name="a1",  # Cond (similarity)
                mu=0,
                sigma=pm.math.sqrt(sigma2_a1),
                shape=Nx1Lvl,
            )
            sigma2_a2 = pm.InverseGamma(name="a2Var",
                                    alpha=prior_params["a2Var"]["alpha"],
                                    beta=prior_params["a2Var"]["beta"], )
            a2 = pm.Normal(
                name="a2",  # Ins (instructions)
                mu=0,
                sigma=pm.math.sqrt(sigma2_a2),
                shape=Nx2Lvl,
            )
            sigma2_Subj = pm.InverseGamma(name="aSubjVar",
                                    alpha=prior_params["aSubjVar"]["alpha"],
                                    beta=prior_params["aSubjVar"]["beta"], )
            aSubj = pm.Normal(
                name="aSubj",
                mu=0,
                sigma=pm.math.sqrt(sigma2_Subj),
                shape=NxSubjLvl,
            )
            sigma2_a1a2 = pm.InverseGamma(name="a1a2Var",
                                    alpha=prior_params["a1a2Var"]["alpha"],
                                    beta=prior_params["a1a2Var"]["beta"], )
            a1a2 = pm.Normal(
                name="a1a2",
                mu=0,
                sigma=pm.math.sqrt(sigma2_a1a2),
                shape=(Nx1Lvl, Nx2Lvl),
            )

            # prior for p(c) standard deviation
            pc_sigma = pm.Uniform(
                "pc_sigma",
                lower=prior_params["pc_sigma"]["lower"],
                upper=prior_params["pc_sigma"]["upper"],
            )

            # general linear model with logistic link function
            mu_latent = a0 + a1[x1] + a2[x2] + aSubj[xSubj] + a1a2[x1, x2]
            mu = pm.math.invlogit(mu_latent)  # equiv to scipy.special.expit
            kappa_plus_1 = mu * (1 - mu) / pc_sigma**2

            # likelihood
            alpha = mu * kappa_plus_1
            beta = (1 - mu) * kappa_plus_1
            pc_obs = pm.Beta("pc_obs", alpha=alpha, beta=beta, observed=y)

            idata = pm.sample(
                cores=self.mcmc_config_state["cores"],
                chains=self.mcmc_config_state["chains"],
                tune=self.mcmc_config_state["tune"],
                draws=self.mcmc_config_state["draws"],
                nuts_sampler="pymc",  # equiv "mcmc" in HssmAnalysis.run_mcmc
                random_seed=self.rng,
                return_inferencedata=True,
            )

        idata.to_netcdf(Path(self.savedir, output_mcmc_chains_filename))

        # summarize chains for parameters of interest
        var_names = ("a0", "a1", "a2", "aSubj", "a1a2", "pc_sigma")
        for vn in var_names:
            az.plot_trace(data=idata, var_names=(vn,))
            plt.title(f"Traces for {vn}")
            plt.savefig(Path(self.savedir, f"trace-{vn}.png"))
            plt.close()
        summary = pm.stats.summary(idata, hdi_prob=0.95)
        summary.to_csv(Path(self.savedir, output_mcmc_summary_filename))

    def analyze_posterior(
        self,
        mcmc_chains_filename: str,
        output_posterior_summary_filename: str,
        output_effects_analysis_filename: str,
    ):
        idata = az.from_netcdf(Path(self.savedir, mcmc_chains_filename))

        a0, a1, a2, aSubj, a1a2 = [
            idata.posterior[k].values for k in ("a0", "a1", "a2", "aSubj", "a1a2")
        ]

        mu_latent = self._marginal_glm_samples(a0, a1, a2, aSubj, a1a2)
        mu = scipy.special.expit(mu_latent)  # in p(c) units
        posterior_summary_rows, effects_comparisons_rows = (
            posterior_summary_and_effects(param_name="p(c)", params=mu)
        )

        postsum_path = Path(self.savedir, output_posterior_summary_filename)
        print(f"Summary of posterior: {postsum_path}")
        pd.DataFrame(posterior_summary_rows).to_csv(postsum_path, index=False)

        posteff_path = Path(self.savedir, output_effects_analysis_filename)
        print(f"Experimental effects analysis: {posteff_path}")
        pd.DataFrame(effects_comparisons_rows).to_csv(posteff_path, index=False)

    def credibility_analysis(
        self,
        pc_data_empirical_path: Path,
        pc_data_postpred_filename: str,
        output_df_filename: str,
    ):
        df_pc_emp = pd.read_csv(pc_data_empirical_path)
        df_pc_hssm = pd.read_csv(Path(self.savedir, pc_data_postpred_filename))

        rows = []

        for inslvl, simlvl in product(
            df_pc_hssm.Ins.unique(), df_pc_hssm.Cond.unique()
        ):
            idx_jntlvl = df_pc_hssm.index[
                (df_pc_hssm.Ins == inslvl) & (df_pc_hssm.Cond == simlvl)
            ]
            pc_emp_jntlvl = df_pc_emp.loc[idx_jntlvl].pc
            pc_pred_jntlvl = df_pc_hssm.iloc[idx_jntlvl].pc
            pc_pred_mode = scipy.stats.mode(pc_pred_jntlvl.round(3))[0]
            pc_pred_mean = pc_pred_jntlvl.mean()
            L, U = hpd(pc_pred_jntlvl)
            pct_credible = np.mean((L < pc_emp_jntlvl) & (pc_emp_jntlvl < U))
            rows.append(
                {
                    "Condition": f"{inslvl}:{simlvl}",
                    "Mean": pc_pred_mean,
                    "Mode": pc_pred_mode,
                    "L": L,
                    "U": U,
                    "Percent Credible": pct_credible,
                }
            )
            fig, ax = plt.subplots()
            ax.hist(pc_emp_jntlvl, bins="auto", density=True, alpha=0.5, label="Data")
            ax.hist(pc_pred_jntlvl, bins="auto", density=True, alpha=0.5, label="HSSM")
            ax.vlines(x=(L, U), ymin=0, ymax=1, colors="red")
            ax.set_xlabel("Proportion correct")
            ax.set_xlim(0, 1)
            ax.legend()
            simlvl_semantic = self.simlvl_semantic[simlvl]
            inslvl_semantic = self.inslvl_semantic[inslvl]
            ax.set_title(
                f"Proportion Correct: Data vs. HSSM Prediction for {simlvl_semantic}/{inslvl_semantic}"
            )
            plt.savefig(Path(self.savedir, f"p(c)_hist_{inslvl}-{simlvl}.png"))

        df_pc_credibility = pd.DataFrame(rows)
        df_pc_credibility.to_csv(Path(self.savedir, output_df_filename), index=False)
