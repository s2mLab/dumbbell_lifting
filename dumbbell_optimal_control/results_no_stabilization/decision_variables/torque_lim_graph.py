from enum import Enum
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "Times New Roman"
# size font
plt.rcParams["font.size"] = 14

# data
class ResultFile(Enum):
    FULL_WINDOW_FATIGUE_TORQUE = "FULL_WINDOW_FATIGUE_TORQUE_$TauXia_all$.pkl"
    FULL_WINDOW_ONLY_FATIGUE = "FULL_WINDOW_ONLY_FATIGUE_$TauXia_mf$.pkl"
    FULL_WINDOW_ONLY_TORQUE = "FULL_WINDOW_ONLY_TORQUE_$TauXia_tau$.pkl"
    NMPC_FATIGUE_TORQUE = "CONDITIONS_FATIGUE_TORQUE_$TauXia_all$.pkl"
    NMPC_ONLY_FATIGUE = "CONDITIONS_ONLY_FATIGUE_$TauXia_mf$.pkl"
    NMPC_ONLY_TORQUE = "CONDITIONS_ONLY_TORQUE_$TauXia_tau$.pkl"


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
def load_data(result_file) -> list:
    # load with pickle
    with open(result_file, "rb") as file:
        data_loaded = pickle.load(file)
    return data_loaded


def export_matplotlib_figure(fig,  name):
    # in png, eps, pdf and svg
    fig.savefig(f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{name}.eps", dpi=300, bbox_inches="tight")
    fig.savefig(f"{name}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(f"{name}.svg", dpi=300, bbox_inches="tight")


def compute_target_load(tau, tau_max):
    return tau / tau_max


def compute_tau_limit(tau_max, mf):
    return (1 - mf) * tau_max


df = pd.DataFrame()

for i, (condition, style) in enumerate(zip(ResultFile, Style)):
    style = style.value
    solution = load_data(condition.value)
    nb_cycles = solution[2]["states"].__len__()
    # opacity[0] = 1

    for j, (sol, sol_state) in enumerate(zip(solution[2]["controls"], solution[2]["states"])):
        dict_to_append = {
            "result_file": condition.name,
            "shoulder_tau_max_lim": compute_tau_limit(50, sol_state["tau_plus_mf"][0, :])[-1],
            "shoulder_tau_min_lim": compute_tau_limit(-50, sol_state["tau_minus_mf"][0, :])[-1],
            "elbow_tau_max_lim": compute_tau_limit(50, sol_state["tau_plus_mf"][1, :])[-1],
            "elbow_tau_min_lim": compute_tau_limit(-50, sol_state["tau_minus_mf"][1, :])[-1],
        }
        df = pd.concat([df, pd.DataFrame(dict_to_append, index=[0])], ignore_index=True)


# for each condition, get the cost of each cycle
# subplot (1,2)
# plot the cost of each cycle
fig, ax = plt.subplots(2, 2, figsize=(10, 8))
ms = 3.5
h =[] # for legend
for (condition, style) in zip(ResultFile, Style):
    style = style.value
    data = df[df["result_file"] == condition.name]
    nb_cycles = data["shoulder_tau_max_lim"].__len__()
    abscissa = np.arange(0, nb_cycles) + 1
    h.append(ax[0,0].plot(abscissa, data["shoulder_tau_max_lim"], color=style[0], linestyle=style[1],  label=style[3], marker="o", ms=ms, markerfacecolor=style[4]))
    ax[0,0].plot(abscissa, data["shoulder_tau_min_lim"], color=style[0], linestyle=style[1],  label=style[3], marker="o", ms=ms, markerfacecolor=style[4])
    ax[0,1].plot(abscissa, data["elbow_tau_max_lim"], color=style[0], linestyle=style[1],  label=style[3], marker="o", ms=ms, markerfacecolor=style[4])
    ax[0,1].plot(abscissa, data["elbow_tau_min_lim"], color=style[0], linestyle=style[1],  label=style[3], marker="o", ms=ms, markerfacecolor=style[4])
    # plot a zero line
    ax[0,0].plot([-1, 32], [0, 0], color="gray", linestyle="-", linewidth=0.5)
    ax[0,1].plot([-1, 32], [0, 0], color="gray", linestyle="-", linewidth=0.5)
    # plot an horizontal line at -50 and 50 in linestyle "--"
    ax[0,0].plot([-1, 32], [-50, -50], color="gray", linestyle="--", linewidth=0.5)
    ax[0,0].plot([-1, 32], [50, 50], color="gray", linestyle="--", linewidth=0.5)
    ax[0,1].plot([-1, 32], [-50, -50], color="gray", linestyle="--", linewidth=0.5)
    ax[0,1].plot([-1, 32], [50, 50], color="gray", linestyle="--", linewidth=0.5)

    ax[0,0].set_title("Shoulder", fontsize=13)
    ax[0,1].set_title("Elbow", fontsize=13)


# add grid
for i in range(2):
    ax[0,i].grid(color="lightgray", linestyle="--", linewidth=1)
    ax[0,i].set_xlabel("Cycle")
    # add letter for scientific article labelling as annotation, IN UPPER CASE not bold, A, B, C, D, ITALIC
    ax[0,i].annotate(chr(65 + i)+ ")", xy=(0.03, 0.94), xycoords="axes fraction", fontsize=12,
                   # italics
                     fontstyle="italic",
                   )
    ax[0,i].set_ylim(-60, 60)
    # set y ticks -50 to 50 with step 10
    ax[0,i].set_yticks([-60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60])
    ax[0, i].set_xlim(0.5, 32.5)
    ax[0, i].set_xticks(np.array([1, 4, 8, 12, 16, 20, 24, 28, 32]))


ax[0,0].set_ylabel("Torque limit (Nm)")

# plot a fake line to have a legend with marker empty and not empty
h1 = ax[0,0].plot([], [], color="white", marker="o", ms=ms, markerfacecolor="white", label="Sliding-horizon", markeredgecolor="black")
h2 = ax[0,0].plot([], [], color="white", marker="o", ms=ms, markerfacecolor="black", label="Full-horizon", markeredgecolor="black")
leg = ax[1,0].legend(handles=[h1[0], h2[0]],
            # oustide right bottom
                bbox_to_anchor=(0, 0),
                 # set corner
                    loc="upper left",
                     ncol=2,
                )
ax[1,0].legend(handles=[hh[0] for hh in h[:3]],
              bbox_to_anchor=(0, 0),
                loc="upper left",
               ncol=3,
             )
# "outside right bottom" set the leg legend with add_artist
ax[1,0].add_artist(leg)
# hide thrid axe
ax[1,0].axis("off")
ax[1,1].axis("off")




# put the legend as a (3,2) table for each element, horizontal below the (1,4) subplots
# ax[1].legend(loc="lower center", bbox_to_anchor=(-1.3, -0.6), ncol=2,
#              title="FULL WINDOW      &     FINITE HORIZON",
#              # transparent background
#                 facecolor="none",
#              )
# titles for each col of the legend ("Finite horizon", "Full horizon")
# remove the box of the legend
# ax[1].legend_.get_frame().set_linewidth(0.0)
# the legend as to below the subplots and displace the plots to leave space for the legend
# plt.subplots_adjust(bottom=0.35)
plt.tight_layout()

export_matplotlib_figure(fig, "tau_limit")
plt.show()
