import itertools
import math
from pathlib import Path
from typing import Dict, Iterable

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import scipy
import pymc as pm


QUANTILES = np.linspace(0.05, 0.95, 19)

N_SAMPLES_DEFLECTION_VAR = 10**4

PARAM_INFO = {
    "alpha": {"hssm_name": "a", "dist": {"scipy": scipy.stats.gamma, "pymc": "Gamma"}},
    "beta": {"hssm_name": "z", "dist": {"scipy": scipy.stats.beta, "pymc": "Beta"}},
    "delta": {"hssm_name": "v", "dist": {"scipy": scipy.stats.norm, "pymc": "Normal"}},
    "tau": {"hssm_name": "t", "dist": {"scipy": scipy.stats.gamma, "pymc": "Gamma"}},
}

link_function_lookup = {"alpha": np.log, "delta": lambda x: x, "tau": np.log}

link_function_name_lookup = {"alpha": "log", "delta": "identity", "tau": "log"}

invlink_function_lookup = {
    "alpha": pm.math.exp,
    "delta": lambda x: x,
    "tau": pm.math.exp,
}


def inv_gamma_alpha_beta_from_mean_var(mean: float, variance: float) -> Dict:
    """Compute alpha (shape) and beta (scale) parameters for a SciPy/PyMC
    inverse gamma distribution.

    Parameters
    ----------
    mean : float
        Mean (mu) of the inverse gamma distribution.
    variance : float
        Variance of the inverse gamma distribution.

    Returns
    -------
    {"alpha": alpha, "beta": beta}
    """
    alpha = (mean / variance) ** 2 + 2
    beta = mean * (alpha - 1)
    return {"alpha": alpha, "beta": beta}


def invgamma_alpha_beta_from_mode_std(mode: float, std: float) -> Dict:
    """Compute alpha (shape) and beta (scale) parameters for a SciPy/PyMC
    inverse gamma distribution.

    Parameters
    ----------
    mode : float
        Mode of the inverse gamma distribution.
    std : float
        Standard deviation of the inverse gamma distribution.

    Returns
    -------
    {"alpha": alpha, "beta": beta}

    Notes
    -----
    See formulas in appendix A.1.
    """
    R = (mode + math.sqrt(mode**2 + 4 * std**2)) / (2 * std**2)
    alpha = 1 + mode * R
    beta = 1 / R
    return {"alpha": alpha, "beta": beta}


def load_scenic_data(prior_data_paths: Iterable[Path]) -> pd.DataFrame:
    dfs = tuple(pd.read_csv(pdp) for pdp in prior_data_paths)
    # tribal knowledge: which path goes with Expt1c
    dfs[0]["Expt"] = "1a"
    dfs[1]["Expt"] = "1c"
    dfs[1].Cond = tuple(map(lambda c: ["AA", "AB"][c - 1], dfs[1].Cond))
    df = pd.concat(objs=dfs, ignore_index=True)
    dfpc = pd.DataFrame(
        [
            {
                "SubjNum": pid,
                "Cond": sim,
                "Expt": expt,  # notional proxy for instructions manipulation
                "pc": df[
                    (df.SubjNum == pid) & (df.Cond == sim) & (df.Expt == expt)
                ].acc.mean(),
            }
            for pid, sim, expt in itertools.product(
                sorted(df.SubjNum.unique()),
                sorted(df.Cond.unique()),
                sorted(df.Expt.unique()),
            )
        ]
    )
    dfpc = dfpc.dropna()
    return dfpc


def plot_informative_prior(
    param_values: np.ndarray,
    x_pdf_int: np.ndarray,
    y_pdf_int: np.ndarray,
    param_quantile_scores: np.ndarray,
    x_quantiles_int: np.ndarray,
    y_quantiles_int: np.ndarray,
    dispersion_samples: np.ndarray,
    x_pdf_dsp: np.ndarray,
    y_pdf_dsp: np.ndarray,
    dsp_quantile_scores: np.ndarray,
    x_quantiles_dsp: np.ndarray,
    y_quantiles_dsp: np.ndarray,
    param_name: str,
    dispersion_type: str,
    savedir: Path,
    filename: str
):
    assert dispersion_type in ("Variance", "Standard Deviation")

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10,6))
    fig.suptitle(f"Informative Prior for GLM Predictor of {param_name}")

    ax[0, 0].set_title("Distribution of Values for Intercept Term")
    ax[0, 0].set_xlabel("Value of Intercept Term")
    ax[0, 0].set_ylabel("Density")
    ax[0, 0].hist(param_values,  bins="auto", density=True, label="Data", color="gray")
    ax[0, 0].plot(x_pdf_int, y_pdf_int, label="Prior PDF", c="k")
    ax[0, 0].legend()

    ax[0, 1].set_title("Quantile Plot for Intercept Term")
    ax[0, 1].set_xlabel("Quantile")
    ax[0, 1].set_ylabel("Value of Intercept Term")
    ax[0, 1].scatter(x_quantiles_int, param_quantile_scores, label="Data", c="gray")
    ax[0, 1].plot(x_quantiles_int, y_quantiles_int, label="Prior Inverse CDF", c="k")
    ax[0, 1].legend()

    ax[1, 0].set_title(f"Samples of {dispersion_type} of Deflection")
    ax[1, 0].set_xlabel(f"Value of {dispersion_type}")
    ax[1, 0].set_ylabel("Density")
    ax[1, 0].hist(dispersion_samples, bins="auto", density=True, label="Data", color="gray")
    ax[1, 0].plot(x_pdf_dsp, y_pdf_dsp, label="Prior PDF", c="k")
    ax[1, 0].legend()

    ax[1, 1].set_title(f"Quantile Plot for {dispersion_type} of Deflection")
    ax[1, 1].set_xlabel("Quantile")
    ax[1, 1].set_ylabel(f"Value of {dispersion_type}")
    ax[1, 1].scatter(x_quantiles_dsp, dsp_quantile_scores, label="Data", c="gray")
    ax[1, 1].plot(x_quantiles_dsp, y_quantiles_dsp, label="Prior Inverse CDF", c="k")
    ax[1, 1].legend()

    fig.tight_layout()
    savepath = Path(savedir, filename)
    print(f"Prior plot for {param_name} saved to {savepath}")
    plt.savefig(savepath)
    plt.close()


def get_hssm_prior(
        prior_data_path: Path,
    savedir: Path,
    rng: np.random.Generator,
        prior_plot_name: str
):
    df = pd.read_csv(prior_data_path)
    output = {}
    fixed = []
    to_be_included = []
    for param in PARAM_INFO:
        hssm_name = PARAM_INFO[param]["hssm_name"]
        first_order = "(1|participant_id) + (1|Cond) + (1|Ins)"
        second_order = "(1|Cond:Ins)"
        prior_entry = {
            "name": hssm_name,
            "formula": f"{hssm_name} ~ 1 + {first_order} + {second_order}",
        }
        param_values = df.loc[df["params"] == param]["par"]
        x_pdf_int = np.linspace(param_values.min(), param_values.max(), 100)
        pv_mean = param_values.mean()
        pv_std = param_values.std()
        param_quantile_scores = np.quantile(param_values, QUANTILES)
        distrib = PARAM_INFO[param]["dist"]["scipy"]
        if distrib == scipy.stats.gamma:
            shape_guess = pv_mean**2 / pv_std**2
            scale_guess = pv_std ** 2 / pv_mean
            shape, scale = fit_ppf(
                quantiles=QUANTILES,
                scores=param_quantile_scores,
                candidate_ppf=lambda x, a, scale: distrib.ppf(
                    x, a=a, scale=scale
                ),
                params_guess=(shape_guess, scale_guess)
            )
            intercept_mu = shape * scale
            intercept_sigma = scale * np.sqrt(shape)
            y_pdf_int = distrib.pdf(x_pdf_int, a=shape, scale=scale)
            y_quantiles_int = distrib.ppf(QUANTILES, a=shape, scale=scale)
        elif distrib == scipy.stats.beta:
            pv_var = np.var(param_values)
            alpha_guess = pv_mean * ((pv_mean * (1 - pv_mean)) / pv_var - 1)
            beta_guess = (1 - pv_mean) * (
                        (pv_mean * (1 - pv_mean)) / pv_var - 1)
            alpha, beta = fit_ppf(
                quantiles=QUANTILES,
                scores=param_quantile_scores,
                candidate_ppf=lambda x, a, b: distrib.ppf(x, a=a, b=b),
                params_guess=(alpha_guess, beta_guess)
            )
            intercept_mu = alpha / (alpha + beta)
            intercept_sigma = np.sqrt(
                alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1))
            )
            y_pdf_int = distrib.pdf(x_pdf_int, a=alpha, b=beta)
            y_quantiles_int = distrib.ppf(QUANTILES, a=alpha, b=beta)
        else:  # distib == scipy.stats.norm
            intercept_mu, intercept_sigma = fit_ppf(
                quantiles=QUANTILES,
                scores=param_quantile_scores,
                candidate_ppf=lambda x, loc, scale: distrib.ppf(
                    x, loc=loc, scale=scale
                ),
                params_guess=(pv_mean, pv_std)
            )
            y_pdf_int = distrib.pdf(x_pdf_int, loc=intercept_mu, scale=intercept_sigma)
            y_quantiles_int = distrib.ppf(QUANTILES, loc=intercept_mu, scale=intercept_sigma)
        deflections_std_samples = rng.choice(
            a=param_values - pv_mean,
            size=(len(param_values) // 10, N_SAMPLES_DEFLECTION_VAR),
        ).std(axis=0)
        x_pdf_dsp = np.linspace(deflections_std_samples.min(),
                                deflections_std_samples.max(), 100)
        dsp_quantile_scores = np.quantile(deflections_std_samples, QUANTILES)
        invgamma_params_init = inv_gamma_alpha_beta_from_mean_var(
            mean=np.mean(deflections_std_samples),
            variance=np.var(deflections_std_samples),
        )  # alpha, beta corresponding to MLEs
        deflection_shape, deflection_scale = fit_ppf(
            quantiles=QUANTILES,
            scores=dsp_quantile_scores,
            candidate_ppf=lambda x, alpha, beta: scipy.stats.invgamma.ppf(
                x, a=alpha, scale=beta
            ),
            params_guess=tuple(invgamma_params_init.values()),
        )
        y_pdf_dsp = scipy.stats.invgamma.pdf(x_pdf_dsp,
                                             a=deflection_shape,
                                             scale=deflection_scale)
        y_quantiles_dsp = scipy.stats.invgamma.ppf(QUANTILES,
                                                   a=deflection_shape,
                                                   scale=deflection_scale)
        prior_entry.update(
            {
                "prior": {
                    **{
                        "Intercept": {
                            "name": PARAM_INFO[param]["dist"]["pymc"],
                            "mu": np.float32(intercept_mu),
                            "sigma": np.float32(intercept_sigma),
                        }
                    },
                    **{
                        k: {
                            "name": "Normal",
                            "mu": 0,
                            "sigma": {
                                "name": "InverseGamma",
                                "alpha": np.float32(deflection_shape),
                                "beta": np.float32(deflection_scale),
                            },
                        }
                        for k in (
                            "1|participant_id",
                            "1|Cond",
                            "1|Ins",
                            "1|Cond:Ins",
                        )
                    },
                }
            }
        )
        to_be_included.append(prior_entry)
        plot_informative_prior(
            param_values=param_values,
            x_pdf_int=x_pdf_int,
            y_pdf_int=y_pdf_int,
            param_quantile_scores=param_quantile_scores,
            x_quantiles_int=QUANTILES,
            y_quantiles_int=y_quantiles_int,
            dispersion_samples=deflections_std_samples,
            x_pdf_dsp=x_pdf_dsp,
            y_pdf_dsp=y_pdf_dsp,
            dsp_quantile_scores=dsp_quantile_scores,
            x_quantiles_dsp=QUANTILES,
            y_quantiles_dsp=y_quantiles_dsp,
            param_name=param,
            dispersion_type="Standard Deviation",
            savedir=savedir,
            filename=f"{prior_plot_name}_{param}.png"
        )
    output["fixed"] = fixed
    output["include"] = to_be_included
    return output


def get_pc_analysis_prior(
        prior_data_paths: Iterable[Path],
        rng: np.random.Generator,
        savedir: Path,
        filename: str | None
):
    data = load_scenic_data(prior_data_paths)
    pc = data.pc.values - 0.0001
    pc_std_overall = pc.std()
    pc_linked = scipy.special.logit(pc)
    pc_linked_mean_overall = pc_linked.mean()

    quantile_scores_a0 = np.quantile(pc_linked, QUANTILES)
    a0_prior_params = fit_ppf(
        quantiles=QUANTILES,
        scores=quantile_scores_a0,
        candidate_ppf=lambda x, mu, sigma: scipy.stats.norm.ppf(
            x, loc=mu, scale=sigma
        ),
        params_guess=(pc_linked_mean_overall, pc_linked.std()),
    )

    variance_samples = rng.choice(
        a=pc_linked - pc_linked_mean_overall,
        size=(len(pc_linked) // 10, N_SAMPLES_DEFLECTION_VAR),
        replace=True
    ).var(axis=0)
    quantile_scores_deflection_var = np.quantile(variance_samples, QUANTILES)
    deflection_prior_params = fit_ppf(
        quantiles=QUANTILES,
        scores=quantile_scores_deflection_var,
        candidate_ppf=lambda x, alpha, beta: scipy.stats.invgamma.ppf(
            x, a=alpha, scale=beta
        ),
        params_guess=tuple(inv_gamma_alpha_beta_from_mean_var(
            mean=variance_samples.mean(),
            variance=variance_samples.var()
        ).values())
    )

    params = {
        "a0": {"mu": np.float32(a0_prior_params[0]),
               "sigma": np.float32(a0_prior_params[1])},
        "pc_sigma": {"lower": np.float32(pc_std_overall / 10),
                     "upper": np.float32(min(0.5,  # sup of std for beta RV
                                             pc_std_overall * 10))},
        **{k: {"alpha": np.float32(deflection_prior_params[0]),
               "beta": np.float32(deflection_prior_params[1])}
           for k in ("a1Var", "a2Var", "aSubjVar", "a1a2Var")}
    }

    if savedir and filename:
        x_pdf_int = np.linspace(pc_linked.min(), pc_linked.max(), 100)
        x_pdf_dsp = np.linspace(variance_samples.min(), variance_samples.max(), 100)
        plot_informative_prior(
            param_values=pc_linked,
            x_pdf_int=x_pdf_int,
            y_pdf_int=scipy.stats.norm.pdf(x_pdf_int,
                                           loc=params["a0"]["mu"],
                                           scale=params["a0"]["sigma"]),
            param_quantile_scores=quantile_scores_a0,
            x_quantiles_int=QUANTILES,
            y_quantiles_int=scipy.stats.norm.ppf(QUANTILES,
                                 loc=params["a0"]["mu"],
                                 scale=params["a0"]["sigma"]),
            dispersion_samples=variance_samples,
            x_pdf_dsp=x_pdf_dsp,
            y_pdf_dsp=scipy.stats.invgamma.pdf(x_pdf_dsp,
                                     a=params["a1Var"]["alpha"],
                                     scale=params["a1Var"]["beta"]),
            dsp_quantile_scores=quantile_scores_deflection_var,
            x_quantiles_dsp=QUANTILES,
            y_quantiles_dsp=scipy.stats.invgamma.ppf(QUANTILES,
                                                a=params["a1Var"]["alpha"],
                                                scale=params["a1Var"]["beta"]),
            param_name="p(c)",
            dispersion_type="Variance",
            savedir=savedir,
            filename=filename
        )

    return params

def fit_ppf(quantiles, scores, candidate_ppf, params_guess):
    fit_params_ppf, _fit_covariances = scipy.optimize.curve_fit(
        candidate_ppf, xdata=quantiles, ydata=scores, p0=params_guess
    )
    return fit_params_ppf
