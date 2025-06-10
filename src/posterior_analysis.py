from pathlib import Path
from itertools import product
from time import time
from typing import Iterator

import arviz as az
import numpy as np
import pandas as pd

from src.analysis import Analysis
from src.rt_analysis import estimate_rt_quantiles
from src.hssm_analysis import declare_model


class PosteriorPredictiveAnalysis(Analysis):
    """
    Posterior predictive analyses for an HSSM model,
    after `HssnAnalysis.run_mcmc` has been called.

    Methods
    -------
    generate_posterior_predictions
        Interface to `hssm.HSSM.sample_posterior_predictons. Creates CSV of
        predicted response time / response accuracy.
    process_posterior_predictions
        Process posterior predictions of response time / response accuracy into
        CSV files used by downstream analyses.

    Notes
    -----
    All paths/filenames are set in `config.yaml`.
    """

    def __init__(
        self,
        savedir: Path,
        rng: np.random.Generator,
        pc_data_path: Path,
        hssm_prior_data_path: Path,
        rt_response_data_path: Path,
        hssm_mcmc_chains_path: Path,
        pc_prior_data_paths: Iterator[Path],
    ):
        super().__init__(savedir)
        self.rng = rng
        self.pc_data_path = pc_data_path
        self.prior_data_path = hssm_prior_data_path
        self.rt_response_data_path = rt_response_data_path
        self.hssm_mcmc_chains_path = hssm_mcmc_chains_path
        self.pc_prior_data_paths = pc_prior_data_paths
        self.n_trials_per_pc = 16
        self.n_rt_samples_per_quantile = 1000
        self.quantiles = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

    def generate_posterior_predictions(self, output_filename: str):
        idata = az.from_netcdf(str(self.hssm_mcmc_chains_path))
        df_pc = pd.read_csv(self.pc_data_path)

        n_draws_per_chain = len(idata.posterior.draw)

        # this many p(c) samples per chain
        n_pc_per_chain = n_draws_per_chain // self.n_trials_per_pc

        print(
            f"Generating {n_pc_per_chain} pred p(c) samples per chain per joint level..."
        )
        model = declare_model(
            prior_data_path=self.prior_data_path,
            rt_response_data_path=self.rt_response_data_path,
            savedir=None,
            rng=self.rng,
        )

        # sample rt,response from posterior predictive
        start = time()
        idata_postpred = model.sample_posterior_predictive(
            idata=idata,
            data=df_pc,
            inplace=False,
            kind="response",
            draws=16 * n_pc_per_chain,
            # random_seed=self.rng  # unsupported?
        )
        end = time()
        print(f"\t...{end - start} sec")
        output_filename = Path(self.savedir, output_filename)
        idata_postpred.to_netcdf(str(output_filename))

    def process_posterior_predictions(
        self,
        mcmc_chains_filename: str,
        output_pc_postpred_filename: str,
        output_rt_response_postpred_filename: str,
        output_rt_quantile_estimates_filename: str,
    ):
        """
        Process predicted response time / response accuracy into
        CSV files used by downstream analyses.

        Parameters
        ----------
        mcmc_chains_filename : str
            Name of file containing MCMC samples of GLM parameters.
        output_pc_postpred_filename : str
            Name of output file containing predicted proportion correct.
        output_rt_response_postpred_filename : str
            Name of output file containing predicted rt/response accuracy.
        output_rt_quantile_estimates_filename : str
            Name of output file containing predicted response time quantiles.

        Returns
        -------
        None

        Notes
        -----
        All filenames are set in `config.yaml`.
        """
        df_pc = pd.read_csv(self.pc_data_path)
        idata_postpred = az.from_netcdf(str(Path(self.savedir, mcmc_chains_filename)))
        n_chains = len(idata_postpred.posterior.chain)
        n_draws_per_chain = len(idata_postpred.posterior.draw)
        n_pidxcell = len(df_pc)  # number of PID x joint levels

        rt_pred = idata_postpred.posterior_predictive["rt,response"].values[
            :, :, :, 0  # all chains  # all draws  # all cell/PID  # just rt
        ]
        response_pred = idata_postpred.posterior_predictive["rt,response"].values[
            :, :, :, 1  # all chains, all draws, all cell/PID, just response
        ]  # values are -1, +1

        # compute and save predicted pc for effects/credibility analyses
        #     manually aggregate predicted responses into predicted p(c)
        #     https://github.com/lnccbrown/HSSM/discussions/647
        response_acc_pred = (1 + response_pred) / 2  # values are 0, 1
        shape_for_pc_calc = (-1, self.n_trials_per_pc, n_pidxcell)
        pc_pred_all = np.mean(
            response_acc_pred.reshape(shape_for_pc_calc), axis=1
        )  # -> shape (n_chains * n_pc_per_chain, len(df_pc))
        df_pc["pc"] = pc_pred_all.mean(0)  # report mean pc, one per i,j,k

        df_pc.to_csv(Path(self.savedir, output_pc_postpred_filename), index=False)

        # save predicted rt,response
        df_pc_repeated = pd.concat(
            [df_pc] * (n_chains * n_draws_per_chain), ignore_index=True
        )
        chain = np.tile(np.repeat(np.arange(n_chains), n_draws_per_chain), n_pidxcell)
        draw = np.tile(np.arange(n_draws_per_chain), n_chains * n_pidxcell)
        df_rt_response_pred = df_pc_repeated.drop(columns=["pc"])
        df_rt_response_pred["chain"] = chain
        df_rt_response_pred["draw"] = draw
        df_rt_response_pred["rt"] = rt_pred.reshape(
            -1,
        )
        df_rt_response_pred["response"] = response_pred.reshape(
            -1,
        )
        df_rt_response_pred.to_csv(output_rt_response_postpred_filename, index=False)

        # estimate rt quantiles for effects/credibility analyses
        rt_quantile_estimates = []
        for inslvl, simlvl, resplvl in product(
            df_rt_response_pred.Ins.unique(),
            df_rt_response_pred.Cond.unique(),
            df_rt_response_pred.response.unique().astype(int),
        ):
            df_jntlvl = df_rt_response_pred[
                (df_rt_response_pred.Ins == inslvl)
                & (df_rt_response_pred.Cond == simlvl)
                & (df_rt_response_pred.response == resplvl)
            ]
            rt_pred_jntlvl = df_jntlvl["rt"].to_numpy()
            n_sim_quantiles, modes, Q_hdi = estimate_rt_quantiles(
                self.quantiles, rt_pred_jntlvl, self.n_rt_samples_per_quantile
            )
            rt_quantile_estimates.extend(
                [
                    {
                        "Ins": inslvl,
                        "Cond": simlvl,
                        "response": resplvl,
                        "Num RT Total": len(rt_pred_jntlvl),
                        "Num RT Per Qs Estimate": self.n_rt_samples_per_quantile,
                        "Num Qs Estimates": n_sim_quantiles,
                        "Q": Q,
                        "Estimated Mode": modes[idxQ],
                        "95% HDI Lower": Q_hdi[idxQ, 0],
                        "95% HDI Upper": Q_hdi[idxQ, 1],
                    }
                    for idxQ, Q in enumerate(self.quantiles)
                ]
            )
        df_rt_quantile_estimates = pd.DataFrame(rt_quantile_estimates)
        df_rt_quantile_estimates.to_csv(
            Path(self.savedir, output_rt_quantile_estimates_filename), index=False
        )
