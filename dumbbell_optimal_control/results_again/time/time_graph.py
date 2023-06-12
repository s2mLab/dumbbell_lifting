from enum import Enum
import os

import pandas as pd
import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "Times New Roman"

# data
class ResultFile(Enum):
    NMPC_FATIGUE_TORQUE = "CONDITIONS_FATIGUE_TORQUE_$TauXia_all$.csv"
    NMPC_ONLY_FATIGUE = "CONDITIONS_ONLY_FATIGUE_$TauXia_mf$.csv"
    NMPC_ONLY_TORQUE = "CONDITIONS_ONLY_TORQUE_$TauXia_tau$.csv"
    FULL_WINDOW_FATIGUE_TORQUE = "FULL_WINDOW_FATIGUE_TORQUE_$TauXia_all$.csv"
    FULL_WINDOW_ONLY_FATIGUE = "FULL_WINDOW_ONLY_FATIGUE_$TauXia_mf$.csv"
    FULL_WINDOW_ONLY_TORQUE = "FULL_WINDOW_ONLY_TORQUE_$TauXia_tau$.csv"


class CostFunctonLatex(Enum):
    FATIGUE_TORQUE = r'$\Phi^{m_f,\tau}$'
    ONLY_FATIGUE = r'$\Phi^{m_f}$'
    ONLY_TORQUE = r'$\Phi^{\tau}$'


class Colors(Enum):
    BLUE = "tab:blue"
    ORANGE = "tab:orange"
    GREEN = "tab:green"


class Style(Enum):
    FINITE_HORIZON_FATIGUE_TORQUE = ("tab:blue", "--", r'Finite horizon $\int \tau^2 + {m_f}^2 + {q_0}² \; dt$')
    FINITE_HORIZON_ONLY_FATIGUE = ("tab:orange", "--", r'Finite horizon $\int {m_f}^2 + {q_0}² \; dt$')
    FINITE_HORIZON_ONLY_TORQUE = ("tab:green", "--", r'Finite horizon $\int \tau^2 + {q_0}² \; dt$')
    FULL_WINDOW_FATIGUE_TORQUE = ("tab:blue", "-", r'Full horizon $\int \tau^2 + {m_f}^2 + {q_0}² \; dt$')
    FULL_WINDOW_ONLY_FATIGUE = ("tab:orange", "-", r'Full horizon $\int {m_f}^2 + {q_0}² \; dt$')
    FULL_WINDOW_ONLY_TORQUE = ("tab:green", "-", r'Full horizon $\int \tau^2 + {q_0}² \; dt$')


# load data
def load_data(result_file) -> dict:
    return pd.read_csv(result_file.value, sep=",")


def export_matplotlib_figure(fig,  name):
    # in png, eps, pdf and svg
    fig.savefig(f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{name}.eps", dpi=300, bbox_inches="tight")
    fig.savefig(f"{name}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(f"{name}.svg", dpi=300, bbox_inches="tight")


def plot_clustered_stacked(dfall, labels=None, title="multiple stacked bar plot", H="/", **kwargs):
    """Given a list of dataframes, with identical columns and index, create a clustered stacked bar plot.
    labels is a list of the names of the dataframe, used for the legend
    title is a string for the title of the plot
    H is the hatch used for identification of the different dataframe"""

    n_df = len(dfall)
    n_col = len(dfall[0].columns)
    n_ind = len(dfall[0].index)
    axe = plt.subplot(111)

    for df, color in zip(dfall, Colors):  # for each data frame
        axe = df.plot(kind="bar",
                      linewidth=0,
                      stacked=True,
                      ax=axe,
                      legend=False,
                      grid=False,
                      color=color.value,
                      alpha=0.95,
                      **kwargs)  # make bar plots

    h, l = axe.get_legend_handles_labels()  # get the handles we want to modify
    for i in range(0, n_df * n_col, n_col):  # len(h) = n_col * n_df
        for j, pa in enumerate(h[i:i + n_col]):
            for rect in pa.patches:  # for each index
                rect.set_x(rect.get_x() + 1 / float(n_df + 1) * 1.05 * i / float(n_col))
                # rect.set_hatch(H * int(i / n_col))  # edited part
                rect.set_width(1 / float(n_df + 1))
                # add egdes around
                rect.set_edgecolor("black")
                rect.set_linewidth(0.25)
                # display number in the middle of rectangles
                # only if is not zero
                if rect.get_height() != 0:
                    # only if this is the last rectangle of the stack
                    if j == n_col - 15 or j == n_col - 14:
                        axe.text(
                            rect.get_x() + rect.get_width() / 2.,
                            rect.get_y() + rect.get_height() / 2.,
                            f"{j + 1}-{j+3}th", ha='center', va='center')


    axe.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.)
    axe.set_xlim(-0.5, n_ind)
    axe.set_xticklabels(df.index, rotation=0)
    axe.set_title(title)

    # # Add invisible data to add another legend
    n = []
    for df, color in zip(dfall, Colors):
        n.append(axe.bar(0, 0, color=color.value))

    if labels is not None:
        l2 = plt.legend(n, labels,
                        # loc=[1.01, 0.1],
                        loc="best",
                        title="Cost function",
                        )

    return axe

normalized_by_cycle = True
remove_last_window = False

data_df1 = np.zeros((2, 32))
data_df2 = np.zeros((2, 32))
data_df3 = np.zeros((2, 32))
for i, result_file in enumerate(ResultFile):
    print(result_file.name)
    data = load_data(result_file)
    # fill a column with the condition name
    data["result_file"] = result_file.name
    if "FULL_WINDOW" in result_file.name:
        data["WINDOW"] = "Full-horizon"
        data_slice = slice(0, 1)
        row_idx = 0
        nb_cycles = 32
    if "CONDITIONS" in result_file.value:
        data["WINDOW"] =  "Sliding-horizon"
        data_slice = slice(0, data.__len__() if not remove_last_window else data.__len__() - 1)
        row_idx = 1
        nb_cycles = data["time"].__len__() + 2
        if remove_last_window:
            nb_cycles -= 1
    try:
        if "FATIGUE_TORQUE" in result_file.value:
            data["COST"] = "FATIGUE_TORQUE"
            # time by total cycles
            if normalized_by_cycle:
                data_time = data["time"][data_slice]
                data_df1[row_idx, data_slice] = data_time / nb_cycles
                print(data_time)
            else:
                data_df1[row_idx, data_slice] = data["time"]

        if "ONLY_FATIGUE" in result_file.value:
            data["COST"] = "ONLY_FATIGUE"
            # time by total cycles
            if normalized_by_cycle:
                data_time = data["time"][data_slice]
                data_df2[row_idx, data_slice] = data_time / nb_cycles
                print(data_time)
            else:
                data_df2[row_idx, data_slice] = data["time"] / 60

        if "ONLY_TORQUE" in result_file.value:
            data["COST"] = "ONLY_TORQUE"
            # time by total cycles
            if normalized_by_cycle:
                data_time = data["time"][data_slice]
                data_df3[row_idx, data_slice] = data_time / nb_cycles
                print(data_time)
            else:
                data_df3[row_idx, data_slice] = data["time"] / 60

    except:
        print(f"Error with {result_file}")


df1 = pd.DataFrame(data_df1,
                   index=["Full-horizon", "Sliding-horizon"],
                   )
df2 = pd.DataFrame(data_df2,
                    index=["Full-horizon", "Sliding-horizon"],
                   )
df3 = pd.DataFrame(data_df3,
                   index=["Full-horizon", "Sliding-horizon"],
                   )

# Then, just call :
axe = plot_clustered_stacked([df1, df2, df3],
                             title=None,
                             labels=[
                                 CostFunctonLatex.FATIGUE_TORQUE.value,
                                 CostFunctonLatex.ONLY_FATIGUE.value,
                                 CostFunctonLatex.ONLY_TORQUE.value,
                             ],
                             )
axe.set_xlabel("Optimal control problems")
if normalized_by_cycle:
    axe.set_ylabel("Time to solve (s/cycle)")
else:
    axe.set_ylabel("Time to solve (min)")
# axe.set_ylabel("Time to solve (s)")
# grid behind bars
axe.set_axisbelow(True)
axe.grid(color="lightgray", linestyle="--", linewidth=1)
# font size 15 everywhere
axe.tick_params(axis="both", which="major", labelsize=15)
axe.set_title(axe.get_title(), fontdict={"size": 15})
axe.set_xlabel(axe.get_xlabel(), fontdict={"size": 15})
axe.set_ylabel(axe.get_ylabel(), fontdict={"size": 15})
# set latex and times new roman on legend
axe.legend_.get_title().set_fontsize(15)
for text in axe.legend_.get_texts():
    text.set_fontsize(15)
    text.set_fontname("Times New Roman")


# get fig from axe

# # set "plt.rcParams["text.usetex"] = True"
# # and set "plt.rcParams["font.family"] = "Times New Roman""
# # to use latex
# # get plt
# # plt = fig.axes[0].get_figure()
#
# # set latex
# axe.set_xlabel(axe.get_xlabel(), fontdict={"family": "Times New Roman", "size": 12}, text)
# axe.set_ylabel(axe.get_ylabel(), fontdict={"family": "Times New Roman", "size": 12})
# axe.set_title(axe.get_title(), fontdict={"family": "Times New Roman", "size": 12})
# axe.tick_params(axis="both", which="major", labelsize=12)
# # set latex and times new roman on legend
# axe.legend_.get_title().set_fontsize(12)
# for text in axe.legend_.get_texts():
#     text.set_fontsize(12)
#     text.set_family("Times New Roman")
# # set latex and times new roman on ticks
# for tick in axe.get_xticklabels():
#     tick.set_fontsize(12)
#     tick.set_family("Times New Roman")
# for tick in axe.get_yticklabels():
#     tick.set_fontsize(12)
#     tick.set_family("Times New Roman")

fig = axe.get_figure()
plt.show()
if normalized_by_cycle:
    export_matplotlib_figure(fig, "time_to_solve_per_cycle")
else:
    export_matplotlib_figure(fig, "time_to_solve")

