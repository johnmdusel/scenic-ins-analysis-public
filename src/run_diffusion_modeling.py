from pathlib import Path
import yaml

import numpy as np

from src.hssm_analysis import HssmAnalysis
from src.pc_analysis import PcAnalysis
from src.posterior_analysis import PosteriorPredictiveAnalysis
from src.rt_analysis import RtAnalysis


if __name__ == "__main__":

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    rng = np.random.default_rng(config["seed"])

    # setup for diffusion model run
    dm_analysis = HssmAnalysis(
        savedir=Path(config["savedir"]),
        rng=rng,
        prior_type=config["hssm_analysis"]["prior_type"],
        rt_response_data_path=Path(config["input_paths"]["rt_response_data"]),
        prior_data_path=Path(config["hssm_analysis"]["prior_data_path"]),
        prior_plot_name=config["hssm_analysis"]["prior_plot"],
    )

    # mcmc sampling (see README for timing)
    dm_analysis.run_mcmc(
        output_mcmc_chains_filename=config["hssm_analysis"]["mcmc_chains"],
        output_mcmc_summary_filename=config["hssm_analysis"]["mcmc_summary"],
    )

    # summarize posterior distribution of GLM parameters
    #  run experimental effects analysis on GLM parameters
    dm_analysis.analyze_posterior(
        mcmc_chains_filename=config["hssm_analysis"]["mcmc_chains"],
        output_posterior_summary_filename=config["hssm_analysis"]["posterior_summary"],
        output_effects_analysis_filename=config["hssm_analysis"]["posterior_effects"],
    )

    # setup for posterior predictive analyses
    postpred_analysis = PosteriorPredictiveAnalysis(
        savedir=Path(config["savedir"]),
        rng=rng,
        pc_data_path=Path(config["input_paths"]["pc_data"]),
        hssm_prior_data_path=Path(config["hssm_analysis"]["prior_data_path"]),
        rt_response_data_path=Path(config["input_paths"]["rt_response_data"]),
        hssm_mcmc_chains_path=Path(
            config["savedir"], config["hssm_analysis"]["mcmc_chains"]
        ),
        pc_prior_data_paths=(Path(p) for p in config["input_paths"]["pc_prior_data"]),
    )

    # create mcmc chains augmented with rt,response predictions
    postpred_analysis.generate_posterior_predictions(
        output_filename=config["postpred_analysis"]["mcmc_chains"]
    )

    # create CSV files used as input for the analyses
    postpred_analysis.process_posterior_predictions(
        mcmc_chains_filename=config["postpred_analysis"]["mcmc_chains"],
        output_pc_postpred_filename=config["postpred_analysis"]["pc_data"],
        output_rt_response_postpred_filename=config["postpred_analysis"][
            "rt_response_data"
        ],
        output_rt_quantile_estimates_filename=config["postpred_analysis"][
            "rt_quantiles"
        ],
    )

    # experimental effects on predicted p(c)
    hssm_pc_analysis = PcAnalysis(
        savedir=Path(config["savedir"]),
        rng=rng,
        pc_data_path=Path(config["savedir"], config["postpred_analysis"]["pc_data"]),
        rt_response_data_path=Path(
            config["savedir"], config["postpred_analysis"]["rt_response_data"]
        ),
        prior_data_paths=[Path(p) for p in config["input_paths"]["pc_prior_data"]],
        prior_plot_name=None,
    )
    hssm_pc_analysis.run_mcmc(
        output_mcmc_chains_filename=config["postpred_analysis"]["pc_mcmc_chains"],
        output_mcmc_summary_filename=config["postpred_analysis"]["pc_mcmc_summary"],
    )
    hssm_pc_analysis.analyze_posterior(
        mcmc_chains_filename=config["postpred_analysis"]["pc_mcmc_chains"],
        output_posterior_summary_filename=config["postpred_analysis"][
            "pc_posterior_summary"
        ],
        output_effects_analysis_filename=config["postpred_analysis"][
            "pc_posterior_effects"
        ],
    )

    # credibility analysis of empirical p(c) according to predicted p(c)
    hssm_pc_analysis.credibility_analysis(
        pc_data_empirical_path=config["input_paths"]["pc_data"],
        pc_data_postpred_filename=config["postpred_analysis"]["pc_data"],
        output_df_filename=config["postpred_analysis"]["pc_credibility"],
    )

    hssm_rt_analysis = RtAnalysis(savedir=Path(config["savedir"]))
    hssm_rt_analysis.effects_plot_predicted(
        rt_quantile_estimates_filename=config["postpred_analysis"]["rt_quantiles"],
        output_plot_filename=config["postpred_analysis"]["rt_effects_plot"],
    )
    hssm_rt_analysis.credibility_analysis(
        rt_response_data_path=config["input_paths"]["rt_response_data"],
        rt_quantiles_postpred_filename=config["postpred_analysis"]["rt_quantiles"],
        output_df_filename=config["postpred_analysis"]["rt_credibility"],
        output_summary_filename=config["postpred_analysis"]["rt_credibility_summary"],
    )
    hssm_rt_analysis.credibility_plot(
        rt_credibility_data_filename=config["postpred_analysis"]["rt_credibility"],
        output_plot_filename=config["postpred_analysis"]["rt_credibility_plot"],
    )
