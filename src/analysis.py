"""
Analysis classes for the SCENIC-INS study.

This module contains classes for setting up and running analyses on the SCENIC-INS dataset.
Classes help create output directories, define experimental variables, and manage MCMC sampling.

Classes
-------
Analysis
    Sets up the output directory (`savedir`) and defines semantics of the experimental variables.

McmcAnalysis
    Sets MCMC sampler parameters and provides a standard way to create MCMC samples
    for use in later analyses.

Notes
-----
These classes are for single-use data analyses in the SCENIC-INS study.
They are not meant to be reused or packaged.
"""

import os
from pathlib import Path

import numpy as np


class Analysis:
    """
    Base class for all analyses.
    Subclassed by `McmcAnalysis` and `RtAnalysis` and `PosteriorPredictiveAnalysis`.
    All outputs will be saved into the directory `savedir`, which is created
    if necessary.

    Notes
    -----
    All paths/filenames are set in `config.yaml`.
    """

    def __init__(self, savedir: Path):
        self.savedir = savedir
        self.resplvl_semantic = {-1: "Error", 1: "Correct"}
        self.inslvl_semantic = {
            "Acc": "Accuracy Instructions",
            "Spd": "Speed Instructions",
        }
        self.simlvl_semantic = {"AA": "AA' Condition", "AB": "AB' Condition"}
        os.makedirs(savedir, exist_ok=True)


class McmcAnalysis(Analysis):
    """
    Base class for MCMC analysis.
    Subclassed by `PcAnalysis` and `HssmAnalysis`.
    Sets MCMC sampler defaults.

    Methods
    -------
    _marginal_glm_samples
        Construct MCMC samples for the output of the GLM predictor.
    analyze_posterior
        Analyze experimental effects and summarize the posterior on GLM parameters

    Notes
    -----
    All paths/filenames are set in `config.yaml`.
    """

    def __init__(self, savedir: Path, rng: np.random.Generator):
        super().__init__(savedir)
        self.rng = rng
        self.mcmc_config_state = {
            "cores": 16,
            "chains": 16,
            "tune": 1000,
            "draws": 2000,
        }

    def _marginal_glm_samples(
        self,
        p0: np.ndarray,
        p1: np.ndarray,
        p2: np.ndarray,
        p3: np.ndarray,
        p1p2: np.ndarray,
    ) -> np.ndarray:
        """
        Construct MCMC samples for the output of the general linear model.

        This function combines MCMC samples of terms from a general linear model, marginalizing out
        participant effects. It is used in experimental effects analyses for p(c) and diffusion model parameters.

        Parameters
        ----------
        p0 : ndarray, shape (c, d, 1)
            Intercept term.
        p1 : ndarray, shape (c, d, 2)
            Main effect for similarity (Cond).
        p2 : ndarray, shape (c, d, 2)
            Main effect for instructions (Ins).
        p3 : ndarray, shape (c, d, 120)
            Participant ID effect.
        p1p2 : ndarray, shape (c, d, 2, 2)
            Interaction between Cond and Ins.

        Returns
        -------
        samples_unlinked : ndarray, shape (c*d, 2, 2)
            Output of the general linear model with participant ID marginalized out.
            No link function applied.
        """
        samples_unlinked = (
            # dimensions are chain, draw, cond, ins, pid
            p0[:, :, np.newaxis, np.newaxis, np.newaxis]  # broadcast over all
            + p1[:, :, :, np.newaxis, np.newaxis]  # broadcast over ins/pid
            + p2[:, :, np.newaxis, :, np.newaxis]  # broadcast over cond/pid
            + p3[:, :, np.newaxis, np.newaxis, :]  # broadcast over cond/ins
            + p1p2[:, :, :, :, np.newaxis]  # broadcast over pid
        )
        samples_unlinked = samples_unlinked.mean(axis=-1)  # marginalize out pid
        samples_unlinked = samples_unlinked.reshape(-1, 2, 2)
        return samples_unlinked

    def analyze_posterior(
        self,
        mcmc_chains_filename: str,
        output_posterior_summary_filename: str,
        output_effects_analysis_filename: str,
    ):
        """
        Analyze experimental effects and summarize the posterior on GLM parameters.

        This method performs an analysis of the experimental effects
        on GLM parameters. It also computes a summary of the posterior distribution
        for these parameters.

        Parameters
        ----------
        mcmc_chains_filename : str
            Name of file containing MCMC samples of GLM parameters.
        output_posterior_summary_filename : str
            Name of file that will contain summary of the posterior distribution.
        output_effects_analysis_filename : str
            Name of file that will contain experimental effects analysis.

        Returns
        -------
        None

        Notes
        -----
        All filenames are set in `config.yaml`.
        """
        pass
