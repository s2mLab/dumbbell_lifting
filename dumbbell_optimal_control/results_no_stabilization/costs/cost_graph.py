from enum import Enum

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "Times New Roman"
# size font
plt.rcParams["font.size"] = 14


# data
class ResultFile(Enum):
    FULL_WINDOW_FATIGUE_TORQUE = "FULL_WINDOW_FATIGUE_TORQUE_$TauXia_all$.csv"
    FULL_WINDOW_ONLY_FATIGUE = "FULL_WINDOW_ONLY_FATIGUE_$TauXia_mf$.csv"
    FULL_WINDOW_ONLY_TORQUE = "FULL_WINDOW_ONLY_TORQUE_$TauXia_tau$.csv"
    NMPC_FATIGUE_TORQUE = "CONDITIONS_FATIGUE_TORQUE_$TauXia_all$.csv"
    NMPC_ONLY_FATIGUE = "CONDITIONS_ONLY_FATIGUE_$TauXia_mf$.csv"
    NMPC_ONLY_TORQUE = "CONDITIONS_ONLY_TORQUE_$TauXia_tau$.csv"


class CostFunction(Enum):
    FATIGUE_TORQUE = r'$\Phi^{m_f,\tau}$'
    ONLY_FATIGUE = r'$\Phi^{m_f}$'
    ONLY_TORQUE = r'$\Phi^{\tau}$'


class Style(Enum):
    # linecolor, linestyle, label, cost_function, facecolor
    # tab:blue in rgba: (0.12156862745098039, 0.4666666666666667, 0.7058823529411765, 1.0)
    # tab:orange in rgba: (1.0, 0.4980392156862745, 0.054901960784313725, 1.0)
    # tab:green in rgba: (0.17254901960784313, 0.6274509803921569, 0.17254901960784313, 1.0)
    # white in rgba: (1.0, 1.0, 1.0, 1.0)
    FULL_WINDOW_FATIGUE_TORQUE = ("tab:blue", "-", r'Full horizon $\int \tau^2 + {m_f}^2 + {q_0}² \; dt$', CostFunction.FATIGUE_TORQUE.value, (0.12156862745098039, 0.4666666666666667, 0.7058823529411765, 0.75))
    FULL_WINDOW_ONLY_FATIGUE = ("tab:orange", "-", r'Full horizon $\int {m_f}^2 + {q_0}² \; dt$', CostFunction.ONLY_FATIGUE.value, (1.0, 0.4980392156862745, 0.054901960784313725, 0.75))
    FULL_WINDOW_ONLY_TORQUE = ("tab:green",  "-", r'Full horizon $\int \tau^2 + {q_0}² \; dt$', CostFunction.ONLY_TORQUE.value, (0.17254901960784313, 0.6274509803921569, 0.17254901960784313, 0.75))
    FINITE_HORIZON_FATIGUE_TORQUE = ("tab:blue", "--", r'Finite horizon $\int \tau^2 + {m_f}^2 + {q_0}² \; dt$', CostFunction.FATIGUE_TORQUE.value,
                                     (1.0, 1.0, 1.0, 0.75))
    FINITE_HORIZON_ONLY_FATIGUE = ("tab:orange", "--", r'Finite horizon $\int {m_f}^2 + {q_0}² \; dt$', CostFunction.ONLY_FATIGUE.value,  (1.0, 1.0, 1.0, 0.75))
    FINITE_HORIZON_ONLY_TORQUE = ("tab:green", "--", r'Finite horizon $\int \tau^2 + {q_0}² \; dt$', CostFunction.ONLY_TORQUE.value, (1.0, 1.0, 1.0, 0.75))


# load data
def load_data(result_file) -> dict:
    return pd.read_csv(result_file.value, sep=",")


def export_matplotlib_figure(fig,  name):
    # in png, eps, pdf and svg
    fig.savefig(f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{name}.eps", dpi=300, bbox_inches="tight")
    fig.savefig(f"{name}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(f"{name}.svg", dpi=300, bbox_inches="tight")


df = pd.DataFrame()
for result_file in ResultFile:
    data = load_data(result_file)
    # fill a column with the condition name
    data["result_file"] = result_file.name
    df = pd.concat([df, data], axis=0)

data = load_data(ResultFile.NMPC_FATIGUE_TORQUE)
# for each condition, get the cost of each cycle
# subplot (1,2)
# plot the cost of each cycle
fig, ax = plt.subplots(2, 2, figsize=(8, 6))
ms = 4
h= []
for (condition, style) in zip(ResultFile, Style):
    style = style.value
    data = df[df["result_file"] == condition.name]
    x = np.arange(1, len(data["shoulder_state_cost"]) + 1)
    h.append(ax[0, 0].plot(x, data["shoulder_state_cost"], color=style[0], linestyle=style[1], label=style[3], marker="o", ms=ms,
                  markerfacecolor=style[4]),
             )
    ax[0, 1].plot(x, data["torque_cost"], color=style[0], linestyle=style[1], label=style[3], marker="o", ms=ms,
                  markerfacecolor=style[4])
    ax[1, 0].plot(x, data["tau_minus_mf_cost"], color=style[0], linestyle=style[1], label=style[3], marker="o", ms=ms,
                    markerfacecolor=style[4])
    ax[1, 1].plot(x, data["tau_plus_mf_cost"], color=style[0], linestyle=style[1], label=style[3], marker="o", ms=ms,
                    markerfacecolor=style[4])


# print final values of the cost
for (condition, style) in zip(ResultFile, Style):
    style = style.value
    data = df[df["result_file"] == condition.name]
    print(f"{condition.name} Shoulder cost : {data['shoulder_state_cost'].iloc[-1]}")
    print(f"{condition.name} Torque cost : {data['torque_cost'].iloc[-1]}")
    print(f"{condition.name} Tau minus mf cost : {data['tau_minus_mf_cost'].iloc[-1]}")
    print(f"{condition.name} Tau plus mf cost : {data['tau_plus_mf_cost'].iloc[-1]}")


# add grid
for i in range(4):
    # sub to ind 2x2
    row = i // 2
    col = i % 2
    # xaxis starts from 0.5 and xticksvalue from 1

    ax[row, col].set_xlim(0.5, 32.5)
    ax[row, col].set_xticks(np.array([1, 4, 8, 12, 16, 20, 24, 28, 32]))
    ax[row, col].grid(color="lightgray", linestyle="--", linewidth=1)

    ax[row, col].set_yscale("log")
    # add letter for scientific article labelling as annotation, IN UPPER CASE not bold, A, B, C, D, ITALIC
    ax[row, col].annotate(chr(65 + i) + ")", xy=(0.05, 0.9), xycoords="axes fraction", fontsize=12,
                            # italics
                            fontstyle="italic",
                            )

ax[1,1].set_ylim(1e-2, 1e4)
ax[1,0].set_ylim(1e-2, 1e4)
ax[0,1].set_ylim(1e2, 1e3)
# more dense grid
ax[0,1].grid(color="lightgray", linestyle="--", linewidth=1, which="minor")
# remove ticks between 1e2 et 1e3
ax[0,1].set_yticklabels([1e2, "", 1e3], minor=True)

ax[1, 0].set_xlabel("Cycle")
ax[1, 1].set_xlabel("Cycle")
ax[0,0].set_ylabel("Cost")
ax[1,0].set_ylabel("Cost")
# put the legend as a (3,2) table for each element, horizontal below the (1,4) subplots
ax[0, 1].legend(handles=[hh[0] for hh in h[:3]],
            loc="best",
             )
# plot a fake line to have a legend with marker empty and not empty
h1 = ax[1, 1].plot([], [], color="white", marker="o", ms=ms, markerfacecolor="white", label="Sliding-horizon", markeredgecolor="black")
h2 = ax[1, 1].plot([], [], color="white", marker="o", ms=ms, markerfacecolor="black", label="Full-horizon", markeredgecolor="black")
ax[1, 1].legend(handles=[h1[0], h2[0]],
            loc="lower right",
                )


# all text of size 15
plt.rc('font', size=15)
plt.rc('axes', titlesize=15)
plt.rc('axes', labelsize=15)
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.rc('legend', fontsize=15)
plt.rc('figure', titlesize=15)


plt.tight_layout()
export_matplotlib_figure(fig, "cost")


# plot the sum of cost
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ms = 4
h= []
for (condition, style) in zip(ResultFile, Style):
    style = style.value
    data = df[df["result_file"] == condition.name]
    # if nan turn to 0
    data["shoulder_state_cost"] = data["shoulder_state_cost"].fillna(0)
    data["torque_cost"] = data["torque_cost"].fillna(0)
    data["tau_minus_mf_cost"] = data["tau_minus_mf_cost"].fillna(0)
    data["tau_plus_mf_cost"] = data["tau_plus_mf_cost"].fillna(0)

    data["sum_cost"] = data["shoulder_state_cost"] + data["torque_cost"] + data["tau_minus_mf_cost"] + data["tau_plus_mf_cost"]
    # build x axis from 1 to N
    x = np.arange(1, len(data["sum_cost"]) + 1)
    h.append(ax.plot(x, data["sum_cost"], color=style[0], linestyle=style[1], label=style[3], marker="o", ms=ms,
                  markerfacecolor=style[4]),
             )

# add grid
ax.grid(color="lightgray", linestyle="--", linewidth=1)
plt.title("Sum of cost")
ax.set_xlabel("Cycle")
ax.set_ylabel("Total Cost")
ax.set_yscale("log")
ax.set_xlim(0.5, 32.5)
ax.set_xticks(np.array([1, 4, 8, 12, 16, 20, 24, 28, 32]))
# legend for cost functions
l = ax.legend(handles=[hh[0] for hh in h[:3]],
            loc="center right",
                )
# plot a fake line to have a legend with marker empty and not empty
h1 = ax.plot([], [], color="white", marker="o", ms=ms, markerfacecolor="white", label="Sliding-horizon", markeredgecolor="black")
h2 = ax.plot([], [], color="white", marker="o", ms=ms, markerfacecolor="black", label="Full-horizon", markeredgecolor="black")
ax.legend(handles=[h1[0], h2[0]],
            loc="lower right",
                )
# add artist
ax.add_artist(l)
export_matplotlib_figure(fig, "total costs")

plt.show()