from pathlib import Path
import yaml

import numpy as np

from src.pc_analysis import PcAnalysis
from src.rt_analysis import RtAnalysis

if __name__ == "__main__":

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    rng = np.random.default_rng(config["seed"])

    # setup for empirical p(c) data analysis
    pc_analysis = PcAnalysis(
        savedir=Path(config["savedir"]),
        rng=rng,
        pc_data_path=Path(config["input_paths"]["pc_data"]),
        rt_response_data_path=Path(config["input_paths"]["rt_response_data"]),
        prior_data_paths=[Path(p) for p in config["input_paths"]["pc_prior_data"]],
        prior_plot_name=config["pc_analysis"]["prior_plot"]
    )

    # mcmc sampling
    pc_analysis.run_mcmc(
        output_mcmc_chains_filename=config["pc_analysis"]["mcmc_chains"],
        output_mcmc_summary_filename=config["pc_analysis"]["mcmc_summary"],
    )

    # summarize posterior distribution of GLM parameters
    # run experimental effects anslysis on GLM parameters
    pc_analysis.analyze_posterior(
        mcmc_chains_filename=config["pc_analysis"]["mcmc_chains"],
        output_posterior_summary_filename=config["pc_analysis"]["posterior_summary"],
        output_effects_analysis_filename=config["pc_analysis"]["posterior_effects"],
    )

    # setup for empirical rt analysis
    rt_analysis = RtAnalysis(savedir=Path(config["savedir"]))

    # quantile-probability plot
    rt_analysis.get_response_proportion_rt_quantiles(
        rt_response_data_path=Path(config["input_paths"]["rt_response_data"]),
        output_csv_filename=config["rt_analysis"]["response_prop_quantiles"],
    )
    rt_analysis.quantile_probability_plot(
        input_data_filename=str(
            Path(config["savedir"], config["rt_analysis"]["response_prop_quantiles"])
        ),
        output_fig_filename=config["rt_analysis"]["response_prop_quantiles_fig"],
    )

    # descriptive plot showing experimental effects
    rt_analysis.effects_plot_empirical(
        rt_response_data_path=Path(config["input_paths"]["rt_response_data"]),
        output_df_filename=config["rt_analysis"]["quantiles"],
        output_plot_filename=config["rt_analysis"]["effects_plot"],
    )
