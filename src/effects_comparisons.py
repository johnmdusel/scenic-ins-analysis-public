"""
This module provides functions for comparing experimental effects in hierarchical models.

The main function, `posterior_summary_and_effects`, summarizes posterior samples and computes effect comparisons for
key parameters. It is essential for reporting results in this repository.
"""

from statistics import mode
from typing import Dict, List, Tuple

import numpy as np

from src.hpd import hpd

PARAM_NAMES = {
    "Alpha": {"unicode": "\u0251", "semantic": "Boundary Separation"},
    "Beta": {"unicode": "\u03b2", "semantic": "Starting Point"},
    "Delta": {"unicode": "\u03b4", "semantic": "Drift Rate"},
    "Tau": {"unicode": "\u03c4", "semantic": "Nondecision Time"},
    "p(c)": {"unicode": "p(c)", "semantic": "Proportion Correct"},
}


def mean_standardized_difference(param_diffs: np.typing.NDArray[np.float32]) -> float:
    standardized_diffs = param_diffs / param_diffs.std()
    msd = standardized_diffs.mean()
    return msd


def posterior_summary_and_effects(
    param_name: str, params: np.ndarray
) -> Tuple[List[Dict], List[Dict]]:
    """
    Summarize posterior samples and compute experimental effect comparisons.

    This function generates summary statistics and effect comparisons for a model parameter.
    It assumes posterior samples are a 3D array: samples, condition, and instruction.

    Parameters
    ----------
    param_name : str
        Name of the parameter. Used in output tables.
    params : np.ndarray
        Posterior samples. Shape (n_samples, 2, 2).
        Axis 0: sample (chains * draws).
        Axis 1: condition (AA, AB).
        Axis 2: instruction (Acc, Spd).

    Returns
    -------
    posterior_summary_rows : List of dict
        Lower 95% HDI limit, mode, upper 85% HDI limit in each joint level.
    effects_comparisons_rows : List of dict
        Main effects, simple effects, and interaction comparisons.
        Each comparison contains:
            - difference being compared
            - lower/upper limits of 95% HDI
            - probability of difference >= 0
            - probability of difference < 0
            - mean standard difference (effect size proxy)
            - mean, median, mode

        Modes are computed from rounding :params: to 3 decimal places.
    """
    params = np.round(params, 3)  # remove excess, unstable precision

    # regression coefficient samples by condition
    AA = np.mean(params[:, 0, :], axis=-1)  # marginalize out ins
    AB = np.mean(params[:, 1, :], axis=-1)
    Acc = np.mean(params[:, :, 0], axis=-1)  # marginalize out cond
    Spd = np.mean(params[:, :, 1], axis=-1)
    AA_Acc = params[:, 0, 0]
    AB_Acc = params[:, 1, 0]
    AA_Spd = params[:, 0, 1]
    AB_Spd = params[:, 1, 1]

    # posterior summary table
    AA_Acc_mode = mode(AA_Acc)
    AB_Acc_mode = mode(AB_Acc)
    AA_Spd_mode = mode(AA_Spd)
    AB_Spd_mode = mode(AB_Spd)
    AA_Acc_hpd = hpd(AA_Acc)
    AB_Acc_hpd = hpd(AB_Acc)
    AA_Spd_hpd = hpd(AA_Spd)
    AB_Spd_hpd = hpd(AB_Spd)
    posterior_summary_rows = [
        {
            "parameter": f"{param_name} LL",
            "AA:Acc": AA_Acc_hpd[0],
            "AB:Acc": AB_Acc_hpd[0],
            "AA:Spd": AA_Spd_hpd[0],
            "AB:Spd": AB_Spd_hpd[0],
        },
        {
            "parameter": f"{param_name} Mode",
            "AA:Acc": AA_Acc_mode,
            "AB:Acc": AB_Acc_mode,
            "AA:Spd": AA_Spd_mode,
            "AB:Spd": AB_Spd_mode,
        },
        {
            "parameter": f"{param_name} UL",
            "AA:Acc": AA_Acc_hpd[1],
            "AB:Acc": AB_Acc_hpd[1],
            "AA:Spd": AA_Spd_hpd[1],
            "AB:Spd": AB_Spd_hpd[1],
        },
    ]

    # effects comparison table
    Cond = AA - AB
    Ins = Acc - Spd
    CondAtAcc = AA_Acc - AB_Acc
    CondAtSpd = AA_Spd - AB_Spd
    interaction = CondAtAcc - CondAtSpd

    Cond_hpd = hpd(Cond)
    Ins_hpd = hpd(Ins)
    CondAtAcc_hpd = hpd(CondAtAcc)
    CondAtSpd_hpd = hpd(CondAtSpd)
    interaction_hpd = hpd(interaction)

    effects_comparisons_rows = [
        {
            "Parameter": param_name,
            "Comparison": "AA - AB",
            "LL": Cond_hpd[0],
            "UL": Cond_hpd[1],
            "Prob >= 0": (Cond >= 0).mean(),
            "Prob < 0": (Cond < 0).mean(),
            "Mean": Cond.mean(),
            "Median": np.median(Cond),
            "Mode": mode(Cond),
            "MSD": mean_standardized_difference(Cond),
        },
        {
            "Parameter": param_name,
            "Comparison": "Acc - Spd",
            "LL": Ins_hpd[0],
            "UL": Ins_hpd[1],
            "Prob >= 0": (Ins >= 0).mean(),
            "Prob < 0": (Ins < 0).mean(),
            "Mean": Ins.mean(),
            "Median": np.median(Ins),
            "Mode": mode(Ins),
            "MSD": mean_standardized_difference(Ins),
        },
        {
            "Parameter": param_name,
            "Comparison": "AA:Acc - AB:Acc",
            "LL": CondAtAcc_hpd[0],
            "UL": CondAtAcc_hpd[1],
            "Prob >= 0": (CondAtAcc >= 0).mean(),
            "Prob < 0": (CondAtAcc < 0).mean(),
            "Mean": CondAtAcc.mean(),
            "Median": np.median(CondAtAcc),
            "Mode": mode(CondAtAcc),
            "MSD": mean_standardized_difference(CondAtAcc),
        },
        {
            "Parameter": param_name,
            "Comparison": "AA:Spd - AB:Spd",
            "LL": CondAtSpd_hpd[0],
            "UL": CondAtSpd_hpd[1],
            "Prob >= 0": (CondAtSpd >= 0).mean(),
            "Prob < 0": (CondAtSpd < 0).mean(),
            "Mean": CondAtSpd.mean(),
            "Median": np.median(CondAtSpd),
            "Mode": mode(CondAtSpd),
            "MSD": mean_standardized_difference(CondAtSpd),
        },
        {
            "Parameter": param_name,
            "Comparison": "(AA:Acc - AB:Acc) - (AA:Spd - AB:Spd)",
            "LL": interaction_hpd[0],
            "UL": interaction_hpd[1],
            "Prob >= 0": (interaction >= 0).mean(),
            "Prob < 0": (interaction < 0).mean(),
            "Mean": interaction.mean(),
            "Median": np.median(interaction),
            "Mode": mode(interaction),
            "MSD": mean_standardized_difference(interaction),
        },
    ]

    return posterior_summary_rows, effects_comparisons_rows
