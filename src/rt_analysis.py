"""
This module provides experimental effects analyses for empirical or predicted response time.

For predicted response time, it provides quantile estimation and also provides a credibility analysis.
"""

from itertools import product
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

from src.analysis import Analysis
from src.hpd import hpd


def estimate_rt_quantiles(
    p_values, rt_data: np.ndarray, n_samples_per_quantile: int
) -> Tuple[int, np.ndarray, np.ndarray]:
    truncated_len = n_samples_per_quantile * (
        rt_data.shape[0] // n_samples_per_quantile
    )
    rt_data = rt_data[:truncated_len].reshape(-1, n_samples_per_quantile)
    n_sim_quantiles = rt_data.shape[0]
    Q_pred = np.quantile(rt_data, p_values, axis=1)
    y = scipy.stats.mode(Q_pred.round(3), axis=1)[0]
    Q_hdi = np.array([hpd(q) for q in Q_pred])
    return n_sim_quantiles, y, Q_hdi


class RtAnalysis(Analysis):
    def __init__(self, savedir: Path):
        super().__init__(savedir)
        self.quantiles = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

    def get_response_proportion_rt_quantiles(
        self, rt_response_data_path: Path, output_csv_filename: str
    ):
        rt_response_data = pd.read_csv(rt_response_data_path)
        rt_response_data.response = (1 + rt_response_data.response) // 2
        rows = []
        for (
            inslvl,
            simlvl,
        ) in product(
            sorted(rt_response_data.Ins.unique()),
            sorted(rt_response_data.Cond.unique()),
        ):
            response_jntlvl = rt_response_data[
                (rt_response_data.Ins == inslvl) & (rt_response_data.Cond == simlvl)
            ].response.to_numpy()
            rt_cor_jntlvl = rt_response_data[
                (rt_response_data.Ins == inslvl)
                & (rt_response_data.Cond == simlvl)
                & (rt_response_data.response == 1)
            ].rt.to_numpy()
            rt_err_jntlvl = rt_response_data[
                (rt_response_data.Ins == inslvl)
                & (rt_response_data.Cond == simlvl)
                & (rt_response_data.response == 0)
            ].rt.to_numpy()
            q_cor = np.quantile(rt_cor_jntlvl, self.quantiles)
            q_err = np.quantile(rt_err_jntlvl, self.quantiles)
            response_prop_cor = response_jntlvl.mean()
            response_prop_err = 1 - response_prop_cor
            rows.extend(
                [
                    {
                        "Ins": inslvl,
                        "Cond": simlvl,
                        "Response": "Correct",
                        "Response Proportion": response_prop_cor,
                        "p value": p,
                        "RT Quantile": q_cor[i],
                        "Number of Responses": len(rt_cor_jntlvl),
                    }
                    for i, p in enumerate(self.quantiles)
                ]
            )
            rows.extend(
                [
                    {
                        "Ins": inslvl,
                        "Cond": simlvl,
                        "Response": "Error",
                        "Response Proportion": response_prop_err,
                        "p value": p,
                        "RT Quantile": q_err[i],
                        "Number of Responses": len(rt_err_jntlvl),
                    }
                    for i, p in enumerate(self.quantiles)
                ]
            )
        output_csv_filename = Path(self.savedir, output_csv_filename)
        pd.DataFrame(rows).to_csv(output_csv_filename)
        print(
            f"Empirical response proportion-RT quantiles saved to {output_csv_filename}"
        )

    def quantile_probability_plot(
        self, input_data_filename: str, output_fig_filename: str
    ):
        data_emp = pd.read_csv(input_data_filename)
        fig, ax = plt.subplot_mosaic(
            mosaic=[[0, 1]], sharey=True, figsize=(8, 6), layout="constrained"
        )
        fig.suptitle("Quantile-Probability Plot")
        fig.supxlabel("Response Proportion")
        fig.supylabel("Response Time Quantile (ms)")
        emp_cor = data_emp[data_emp.Response == "Correct"]
        emp_err = data_emp[data_emp.Response == "Error"]
        ax[0].set_title("Error Response")
        ax[0].set_xticks(emp_err["Response Proportion"].to_list())
        ax[0].set_xticklabels(emp_err["Response Proportion"].round(3).to_list())
        ax[0].grid(visible=True, axis="y")
        ax[1].set_title("Correct Response")
        ax[1].set_xticks(emp_cor["Response Proportion"].to_list())
        ax[1].set_xticklabels(emp_cor["Response Proportion"].round(3).to_list())
        ax[1].grid(visible=True, axis="y")

        markers = {"Acc": {"AA": "1", "AB": "2"}, "Spd": {"AA": "3", "AB": "4"}}
        for inslvl, simlvl in product(data_emp.Ins.unique(), data_emp.Cond.unique()):
            inslabel = self.inslvl_semantic[inslvl]
            simlabel = self.simlvl_semantic[simlvl]
            emp_cor_jntlvl = data_emp[
                (data_emp.Response == "Correct")
                & (data_emp.Ins == inslvl)
                & (data_emp.Cond == simlvl)
            ]
            emp_err_jntlvl = data_emp[
                (data_emp.Response == "Error")
                & (data_emp.Ins == inslvl)
                & (data_emp.Cond == simlvl)
            ]
            ax[0].scatter(
                x=emp_err_jntlvl["Response Proportion"],
                y=emp_err_jntlvl["RT Quantile"] * 1e3,
                marker=markers[inslvl][simlvl],
                c="k",
                s=75,
                label=f"{inslabel}, {simlabel}",
            )
            ax[1].scatter(
                x=emp_cor_jntlvl["Response Proportion"],
                y=emp_cor_jntlvl["RT Quantile"] * 1e3,
                marker=markers[inslvl][simlvl],
                c="k",
                s=75,
            )
        ax[0].legend(
            bbox_to_anchor=(0.0, 1.02, 2.0, 0.102),
            loc="lower left",
            ncols=2,
            mode="expand",
            borderaxespad=2.0,
        )
        output_fig_filename = Path(self.savedir, output_fig_filename)
        plt.savefig(output_fig_filename)
        plt.close()
        print(f"Quantile-probability plot saved to {output_fig_filename}")

    def _setup_effects_plot_axes(self):
        fig, ax = plt.subplots(
            nrows=1, ncols=len(self.resplvl_semantic), figsize=(12, 4), sharey=True
        )
        for idx, resplvl in enumerate(self.resplvl_semantic.values()):
            ax[idx].set_title(f"{resplvl} Response")
            ax[idx].set_ylabel("Response Time (ms)")
            ax[idx].set_xlabel("p-value")
            ax[idx].grid(visible=True, axis="y")
        return fig, ax

    def effects_plot_empirical(
        self,
        rt_response_data_path: Path,
        output_df_filename: str,
        output_plot_filename: str,
    ):
        data = pd.read_csv(rt_response_data_path)
        fig, ax = self._setup_effects_plot_axes()
        fig.suptitle("Response Time Quantiles by Experimental Condition")
        rt_quantile_rows = []
        for inslvl, simlvl in product(
            self.inslvl_semantic,
            self.simlvl_semantic,
        ):
            inslabel = self.inslvl_semantic[inslvl]
            simlabel = self.simlvl_semantic[simlvl]
            for idx, resplvl in enumerate(self.resplvl_semantic):
                data_jntlvl = data[
                    (data.Ins == inslvl)
                    & (data.Cond == simlvl)
                    & (data.response == resplvl)
                ]
                rt_jntlvl = data_jntlvl.rt.to_numpy()
                q_scores = np.quantile(rt_jntlvl, self.quantiles)
                q_scores *= 1e3  # convert to ms
                rt_quantile_rows.extend(
                    [
                        {
                            "Ins": inslvl,
                            "Cond": simlvl,
                            "response": resplvl,
                            "p": p,
                            "q": q_scores[idxp],
                        }
                        for idxp, p in enumerate(self.quantiles)
                    ]
                )
                ax[idx].plot(
                    self.quantiles - {"AA": -0.005, "AB": 0.005}[simlvl],
                    q_scores,
                    label=f"{inslabel}, {simlabel}",
                    marker=".",
                    c={"Acc": "k", "Spd": "gray"}[inslvl],
                    linestyle={"AA": "solid", "AB": "dashed"}[simlvl],
                )
                ax[idx].legend()
        output_df_filename = Path(self.savedir, output_df_filename)
        pd.DataFrame(rt_quantile_rows).to_csv(output_df_filename)
        plt.tight_layout()
        output_plot_filename = Path(self.savedir, output_plot_filename)
        plt.savefig(output_plot_filename)
        print(f" RT quantiles (empirical data) from data saved to {output_df_filename}")
        print(f"RT effects plot (empirical data) saved to {output_plot_filename}")

    def effects_plot_predicted(
        self, rt_quantile_estimates_filename: str, output_plot_filename: str
    ):
        data = pd.read_csv(Path(self.savedir, rt_quantile_estimates_filename))
        fig, ax = self._setup_effects_plot_axes()
        fig.suptitle(
            "Predicted Response Time Quantiles by Experimental Condition"
            + " (Mode / 95% HDI)"
        )
        for inslvl, simlvl in product(
            self.inslvl_semantic,
            self.simlvl_semantic,
        ):
            inslabel = self.inslvl_semantic[inslvl]
            simlabel = self.simlvl_semantic[simlvl]
            for idx, resplvl in enumerate(self.resplvl_semantic):
                data_jntlvl = data[
                    (data.Ins == inslvl)
                    & (data.Cond == simlvl)
                    & (data.response == resplvl)
                ]
                x = self.quantiles + {"AA": -0.005, "AB": 0.005}[simlvl]
                y = data_jntlvl["Estimated Mode"].to_numpy()
                L = data_jntlvl["95% HDI Lower"].to_numpy()
                U = data_jntlvl["95% HDI Upper"].to_numpy()
                yerr = np.abs(np.row_stack((L, U)) - y[np.newaxis, ...])
                ax[idx].errorbar(
                    x=x,
                    y=y,
                    yerr=yerr,
                    label=f"{inslabel}, {simlabel}",
                    c={"Acc": "k", "Spd": "gray"}[inslvl],
                    linestyle={"AA": "solid", "AB": "dashed"}[simlvl],
                    linewidth=1,
                )
                ax[idx].legend()
                ax[idx].grid(visible=True, axis="y")
                ax[idx].set_xticks(self.quantiles)
                ax[idx].set_xticklabels(self.quantiles)
        plt.tight_layout()
        output_plot_filename = Path(self.savedir, output_plot_filename)
        plt.savefig(output_plot_filename)
        print(f"Predicted RT effects saved to {output_plot_filename}")

    def credibility_analysis(
        self,
        rt_response_data_path: Path,
        rt_quantiles_postpred_filename: str,
        output_df_filename: str,
        output_summary_filename: str,
    ):
        rt_response_data = pd.read_csv(rt_response_data_path)
        rt_quantiles_postpred = pd.read_csv(
            Path(self.savedir, rt_quantiles_postpred_filename)
        )
        resp_lvls = rt_quantiles_postpred.response.unique()
        ins_lvls = rt_quantiles_postpred.Ins.unique()
        sim_lvls = rt_quantiles_postpred.Cond.unique()
        rt_credibility_rows = []
        rt_credibility_summary_rows = []
        for resplvl, inslvl, simlvl in product(resp_lvls, ins_lvls, sim_lvls):
            rt_emp_resplvl_jntlvl = rt_response_data[
                (rt_response_data.response == resplvl)
                & (rt_response_data.Ins == inslvl)
                & (rt_response_data.Cond == simlvl)
            ].rt.to_numpy()
            Q_data_jntlvl = np.quantile(rt_emp_resplvl_jntlvl, self.quantiles)
            rt_quantiles_jntlvl = rt_quantiles_postpred[
                (rt_quantiles_postpred.response == resplvl)
                & (rt_quantiles_postpred.Ins == inslvl)
                & (rt_quantiles_postpred.Cond == simlvl)
            ]
            hdi = rt_quantiles_jntlvl[["95% HDI Lower", "95% HDI Upper"]].values
            Q_model_jntlvl = rt_quantiles_jntlvl["Estimated Mode"].values
            credibility_qwise = (hdi[:, 0] <= Q_data_jntlvl) & (
                Q_data_jntlvl <= hdi[:, 1]
            )
            rt_credibility_rows.extend(
                [
                    {
                        "Ins": inslvl,
                        "Cond": simlvl,
                        "response": resplvl,
                        "Quantile": q,
                        "Q Data": Q_data_jntlvl[i],
                        "Q Model": Q_model_jntlvl[i],
                        "95% HDI Lower": hdi[i, 0],
                        "95% HDI Upper": hdi[i, 1],
                        "Credible": int(credibility_qwise[i]),
                    }
                    for i, q in enumerate(self.quantiles)
                ]
            )
            rt_credibility_summary_rows.append(
                {
                    "Ins": inslvl,
                    "Cond": simlvl,
                    "response": resplvl,
                    "Percent Credible:": credibility_qwise.mean(),
                }
            )
        df_rt_credibility = pd.DataFrame(rt_credibility_rows)
        output_df_filename = Path(self.savedir, output_df_filename)
        df_rt_credibility.to_csv(output_df_filename, index=False)
        df_rt_credibility_summary = pd.DataFrame(rt_credibility_summary_rows)
        output_summary_filename = Path(self.savedir, output_summary_filename)
        df_rt_credibility_summary.to_csv(output_summary_filename)

    def credibility_plot(
        self, rt_credibility_data_filename: str, output_plot_filename: str
    ):
        data = pd.read_csv(Path(self.savedir, rt_credibility_data_filename))
        fig, ax = plt.subplots(
            nrows=2, ncols=2, sharex=False, sharey=False, figsize=(12, 8)
        )
        fig.suptitle("Posterior Credibility Check for RT quantiles")
        for idx_ins, inslvl in enumerate(data.Ins.unique()):
            inslabel = self.inslvl_semantic[inslvl]
            for idx_sim, simlvl in enumerate(data.Cond.unique()):
                simlabel = self.simlvl_semantic[simlvl]
                ax[idx_ins, idx_sim].set_title(f"{inslabel}, {simlabel}")
                ax[idx_ins, idx_sim].set_xlabel("p-Value")
                ax[idx_ins, idx_sim].set_ylabel("Response Time (sec)")
                for resplvl in data.response.unique():
                    data_jntlvl = data[
                        (data.Ins == inslvl)
                        & (data.Cond == simlvl)
                        & (data.response == resplvl)
                    ]
                    color = {1: "k", -1: "gray"}[resplvl]
                    x = self.quantiles + {-1: -0.005, 1: 0.005}[resplvl]
                    y_data = data_jntlvl["Q Data"].values
                    y_model = data_jntlvl["Q Model"].values
                    L = data_jntlvl["95% HDI Lower"].to_numpy()
                    U = data_jntlvl["95% HDI Upper"].to_numpy()
                    yerr = np.abs(np.row_stack((L, U)) - y_model[np.newaxis, ...])
                    num_credible = data_jntlvl.Credible.sum()
                    dataplotlabel = f"Data"
                    dataplotlabel += f", {self.resplvl_semantic[resplvl]} Response"
                    dataplotlabel += f", {num_credible}/{len(self.quantiles)} Credible"
                    ax[idx_ins, idx_sim].scatter(
                        x=x, y=y_data, label=dataplotlabel, c=color, linewidth=1
                    )
                    ax[idx_ins, idx_sim].errorbar(
                        x=x,
                        y=y_model,
                        yerr=yerr,
                        label=f"HSSM 95% HDI, {self.resplvl_semantic[resplvl]} Response",
                        c=color,
                    )
                    ax[idx_ins, idx_sim].legend()
                    ax[idx_ins, idx_sim].grid(visible=True, axis="y"),
                    ax[idx_ins, idx_sim].set_xticks(self.quantiles)
                    ax[idx_ins, idx_sim].set_xticklabels(self.quantiles)
        plt.tight_layout()
        output_plot_filename = Path(self.savedir, output_plot_filename)
        plt.savefig(output_plot_filename)
        print(f"RT credibility plot saved to {output_plot_filename}")
