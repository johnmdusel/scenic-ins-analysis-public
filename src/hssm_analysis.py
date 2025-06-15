from pathlib import Path
from typing import Union

import arviz as az
import hssm
import numpy as np
import pandas as pd
import pytensor

from src.analysis import McmcAnalysis
from src.prior import get_hssm_prior
from src.effects_comparisons import posterior_summary_and_effects

hssm.set_floatX("float32")
pytensor.config.floatX = "float32"
pytensor.config.blas__ldflags = "-llapack -lblas -lcblas"


def declare_model(
    prior_data_path: Path,
    rt_response_data_path: Path,
    savedir: Union[Path, None],
    rng: np.random.Generator,
    prior_plot_name: str,
) -> hssm.HSSM:
    """
    Construct HSSM model with an informed prior, using point estimates of
    diffusion model parameters from the SCENIC study.

    Parameters
    ----------
    prior_data_path : Path
        Path to `ScenicParams.csv`.
    rt_response_data_path : Path
        Path to `SCENICINS_FullDat.csv`.
    savedir : path
        Where to save the analysis outputs.
    rng : np.random.Generator
        Random state, for reproducibility.
    prior_plot_name : str
        Filename for plot showing prior distributions.

    Returns
    -------
    hssm.HSSM
        Diffusion model with variable parameters

        - a: Boundary separation
        - v: Drift rate
        - z: Normalized starting point
        - t: Nondecision time

        Lapse probability disabled

    Notes
    -----
    All paths and filenames are set in `config.yaml`.
    """
    prior = get_hssm_prior(prior_data_path, savedir, rng, prior_plot_name)
    model = hssm.HSSM(
        data=pd.read_csv(rt_response_data_path),
        model="ddm",
        include=prior["include"],
        p_outlier=None,
        lapse=None,
    )
    return model


def param_key(param_symb, term):
    return {
        "p0": f"{param_symb}_Intercept",
        "p1": f"{param_symb}_1|Cond",
        "p2": f"{param_symb}_1|Ins",
        "p3": f"{param_symb}_1|participant_id",
        "p1p2": f"{param_symb}_1|Cond:Ins",
    }[term]


class HssmAnalysis(McmcAnalysis):
    """
    Diffusion model analysis for empirical response time and response accuracy.

    Methods
    -------
    run_mcmc
        Perform MCMC analysis.
        Runtime for default settings is ~36 hours
        on DELL Precision 7920 (Xeon Gold 6138 CPU @ 2.00GHz × 80, 125.5 GiB memory).
    analyze_posterior
        See description for superclass.

    Notes
    -----
    All paths/filenames are set in `config.yaml`.
    """

    def __init__(
        self,
        savedir: Path,
        rng: np.random.Generator,
        prior_type: str,
        rt_response_data_path: Path,
        prior_data_path: Path,
        prior_plot_name: str,
    ):
        super().__init__(savedir, rng)
        self.prior_type = prior_type
        self.rt_response_data_path = rt_response_data_path
        self.prior_data_path = prior_data_path
        self.prior_plot_name = prior_plot_name

    def run_mcmc(
        self, output_mcmc_chains_filename: str, output_mcmc_summary_filename: str
    ):
        model = declare_model(
            prior_data_path=self.prior_data_path,
            rt_response_data_path=self.rt_response_data_path,
            savedir=self.savedir,
            rng=self.rng,
            prior_plot_name=self.prior_plot_name,
        )

        model.sample(
            cores=self.mcmc_config_state["cores"],
            chains=self.mcmc_config_state["chains"],
            draws=self.mcmc_config_state["draws"],
            tune=self.mcmc_config_state["tune"],
            sampler="mcmc",  # equiv "pymc" in PcAnalysis.run_mcmc
            random_seed=self.rng,
        )  # populates model.traces

        model.traces.to_netcdf(Path(self.savedir, output_mcmc_chains_filename))

        # summarize results
        summary = model.summary()
        # summary = az.summary(model.traces)  # extra details
        summary.to_csv(Path(self.savedir, output_mcmc_summary_filename))

    def analyze_posterior(
        self,
        mcmc_chains_filename: str,
        output_posterior_summary_filename: str,
        output_effects_analysis_filename: str,
    ):
        posterior_summary_rows, effects_comparisons_rows = [], []
        param_name = {"v": "Delta", "a": "Alpha", "t": "Tau", "z": "Beta"}
        inferencedata = az.from_netcdf(Path(self.savedir, mcmc_chains_filename))

        for param_symb in ("v", "a", "t", "z"):
            p0, p1, p2, p3, p1p2 = [
                inferencedata.posterior[param_key(param_symb, k)].values
                for k in ("p0", "p1", "p2", "p3", "p1p2")
            ]
            p1p2 = p1p2.reshape((*p1p2.shape[:2], 2, 2))
            params = self._marginal_glm_samples(p0, p1, p2, p3, p1p2)

            p_postsumrows, p_posteffrows = posterior_summary_and_effects(
                param_name=param_name[param_symb], params=params
            )
            posterior_summary_rows.extend(p_postsumrows)
            effects_comparisons_rows.extend(p_posteffrows)

        output_posterior_summary_filename = Path(
            self.savedir, output_posterior_summary_filename
        )
        print(f"Saving posterior summary: {output_posterior_summary_filename}")
        pd.DataFrame(posterior_summary_rows).to_csv(output_posterior_summary_filename)

        output_effects_analysis_filename = Path(
            self.savedir, output_effects_analysis_filename
        )
        print(f"Saving posterior effects: {output_effects_analysis_filename}")
        pd.DataFrame(effects_comparisons_rows).to_csv(output_effects_analysis_filename)
