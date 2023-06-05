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
    FULL_WINDOW_FATIGUE_TORQUE = (
        "tab:blue", "-", r'Full horizon $\int \tau^2 + {m_f}^2 + {q_0}² \; dt$', CostFunction.FATIGUE_TORQUE.value,
        "tab:blue")
    FULL_WINDOW_ONLY_FATIGUE = (
        "tab:orange", "-", r'Full horizon $\int {m_f}^2 + {q_0}² \; dt$', CostFunction.ONLY_FATIGUE.value, "tab:orange")
    FULL_WINDOW_ONLY_TORQUE = (
        "tab:green", "-", r'Full horizon $\int \tau^2 + {q_0}² \; dt$', CostFunction.ONLY_TORQUE.value, "tab:green")
    FINITE_HORIZON_FATIGUE_TORQUE = (
        "tab:blue", "--", r'Finite horizon $\int \tau^2 + {m_f}^2 + {q_0}² \; dt$', CostFunction.FATIGUE_TORQUE.value,
        "white")
    FINITE_HORIZON_ONLY_FATIGUE = (
        "tab:orange", "--", r'Finite horizon $\int {m_f}^2 + {q_0}² \; dt$', CostFunction.ONLY_FATIGUE.value, "white")
    FINITE_HORIZON_ONLY_TORQUE = (
        "tab:green", "--", r'Finite horizon $\int \tau^2 + {q_0}² \; dt$', CostFunction.ONLY_TORQUE.value, "white")


# load data
def load_data(result_file) -> list:
    # load with pickle
    with open(result_file, "rb") as file:
        data_loaded = pickle.load(file)
    return data_loaded


def export_matplotlib_figure(fig, name):
    # in png, eps, pdf and svg
    # fig.savefig(f"{name}.png", dpi=300, bbox_inches="tight")
    # fig.savefig(f"{name}.eps", dpi=300, bbox_inches="tight")
    # fig.savefig(f"{name}.pdf", dpi=300, bbox_inches="tight")
    # fig.savefig(f"{name}.svg", dpi=300, bbox_inches="tight")

    fig.savefig(f"{name}.eps", dpi=300)
    fig.savefig(f"{name}.pdf", dpi=300)
    fig.savefig(f"{name}.svg", dpi=300)
    fig.savefig(f"{name}.png", dpi=300)

# for each condition, get the cost of each cycle
# subplot (1,2)
# plot the cost of each cycle
fig, ax = plt.subplots(2, 2, figsize=(7.5, 7.5))
ms = 3
for (condition, style) in zip(ResultFile, Style):
    style = style.value
    solution = load_data(condition.value)
    ax[0, 0].plot(
        solution[2]["time"][0],
        solution[2]["states"][0]["q"][0, :] * 180 / np.pi,
        label=style[3], linestyle=style[1], color=style[0], markerfacecolor=style[4], marker="o", markersize=ms)

    ax[0, 1].plot(
        solution[2]["time"][0],
        solution[2]["states"][-1]["q"][0, :] * 180 / np.pi,
        label=style[3], linestyle=style[1], color=style[0], markerfacecolor=style[4], marker="o", markersize=ms)

    ax[1, 0].plot(
        solution[2]["time"][0],
        solution[2]["states"][0]["q"][1, :] * 180 / np.pi,
        label=style[3], linestyle=style[1], color=style[0], markerfacecolor=style[4], marker="o", markersize=ms)

    ax[1, 1].plot(
        solution[2]["time"][0],
        solution[2]["states"][-1]["q"][1, :] * 180 / np.pi,
        label=style[3], linestyle=style[1], color=style[0], markerfacecolor=style[4], marker="o", markersize=ms)

ax[0, 0].set_title("First cycle")
ax[0, 1].set_title("Last cycle")
ax[0, 0].set_xlim(0, 1)
ax[0, 0].set_ylabel("Shoulder Angle (deg) Flexion(+)/Extension(-)")
ax[1, 0].set_ylabel("Elbow Angle (deg) Flexion(+)/Extension(-)")
ax[1, 0].set_xlabel("Time (s)")
ax[1, 1].set_xlabel("Time (s)")
ax[0, 0].grid(color="lightgray", linestyle="--", linewidth=1)
ax[0, 1].grid(color="lightgray", linestyle="--", linewidth=1)
ax[1, 0].grid(color="lightgray", linestyle="--", linewidth=1)
ax[1, 1].grid(color="lightgray", linestyle="--", linewidth=1)

for i in range(4):
    row = i // 2
    col = i % 2
    ax[row, col].annotate(chr(65 + i) + ")", xy=(0.05, 0.95), xycoords="axes fraction", fontsize=12,
                          # italics
                          fontstyle="italic",
                          )

plt.legend(loc="best", ncol=2,
           title="FULL WINDOW      &     FINITE HORIZON",
           facecolor="none",
           )
# no box around the legend linewidth=0
plt.legend().get_frame().set_linewidth(0.0)
# don't leave space btw subplots horizontally
plt.subplots_adjust(wspace=0)
plt.tight_layout()
# export_matplotlib_figure(fig, "first_and_last_cycles_q")

##figure 2
fig, ax = plt.subplots(4, 3, figsize=(9, 9))
ms = 0.5
row, col = np.unravel_index(np.arange(6), (2, 3))
for i, (condition, style) in enumerate(zip(ResultFile, Style)):
    style = style.value
    solution = load_data(condition.value)
    nb_cycles = solution[2]["states"].__len__()
    linewidth = (np.exp(-1 + np.linspace(0.5, 1, nb_cycles)) ** 5 + np.flip(
        np.exp(-1 + np.linspace(0.5, 1, nb_cycles)) ** 5)) / 2 * 1 / 0.5410425 + 0.25
    # linewidth[0] = 1.0
    opacity = (np.exp(-1 + np.linspace(0.5, 1, nb_cycles)) ** 5 + np.flip(
        np.exp(-1 + np.linspace(0.5, 1, nb_cycles)) ** 5)) / 2 * 1 / 0.5410425
    # opacity[0] = 1

    for j, sol in enumerate(solution[2]["states"]):
        ax[row[i], col[i]].plot(
            solution[2]["time"][0],
            sol["q"][0, :] * 180 / np.pi,
            label=style[3], linestyle=style[1], color=style[0], markerfacecolor=style[4], marker="o", markersize=ms,
            linewidth=linewidth[j], alpha=opacity[j])
        ax[row[i], col[i]].set_ylim(-60, 20)
        # box off and remove axis
        ax[row[i], col[i]].spines["top"].set_visible(False)
        ax[row[i], col[i]].spines["right"].set_visible(False)
        # ax[row[i], col[i]].spines["bottom"].set_visible(False)
        # ax[row[i], col[i]].spines["left"].set_visible(False)
        # # remove ticks
        # ax[row[i], col[i]].tick_params(axis="both", which="both", bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

    # ANNOTATE first and last curve
    idx = int(0.05 * 50)
    x = solution[2]["time"][0][idx]
    y = solution[2]["states"][0]["q"][0, idx] * 180 / np.pi
    an1 = ax[row[i], col[i]].annotate(
        # '1st cycle',
        '',
        xy=(x, y),
        xycoords='data',
        xytext=(0.25, 0.95),
        textcoords='axes fraction',
        arrowprops=dict(arrowstyle="->", facecolor='gray', edgecolor='gray'))

    idx = int(0.7 * 50)
    x = solution[2]["time"][0][idx]
    y = solution[2]["states"][-1]["q"][0, idx] * 180 / np.pi
    an2 = ax[row[i], col[i]].annotate(
        # '1st cycle',
        '',
        xy=(x, y),
        xycoords='data',
        xytext=(0.8, 0.1),
        textcoords='axes fraction',
        arrowprops=dict(arrowstyle="->", facecolor='black'))

    # get the last handles
    handles, labels = ax[row[i], col[i]].get_legend_handles_labels()
    # get the last two handles
    arrow_handle = handles[-2:]

    for j, sol in enumerate(solution[2]["states"]):
        ax[row[i] + 2, col[i]].plot(
            solution[2]["time"][0],
            sol["q"][1, :] * 180 / np.pi,
            label=style[3], linestyle=style[1], color=style[0], markerfacecolor=style[4], marker="o", markersize=ms,
            linewidth=linewidth[j], alpha=opacity[j])
        ax[row[i] + 2, col[i]].set_ylim(-1, 159)
        # show xticksvalues
        # 0, and 150
        ax[row[i] + 2, col[i]].set_yticks([0, 150])
        # box off and remove axis
        ax[row[i] + 2, col[i]].spines["top"].set_visible(False)
        ax[row[i] + 2, col[i]].spines["right"].set_visible(False)
        # ax[row[i]+ 2, col[i]].spines["bottom"].set_visible(False)
        # ax[row[i]+ 2, col[i]].spines["left"].set_visible(False)
        # # remove ticks
        # ax[row[i]+ 2, col[i]].tick_params(axis="both", which="both", bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

    idx = int(0.15 * 50)
    x = solution[2]["time"][0][idx]
    y = solution[2]["states"][0]["q"][1, idx] * 180 / np.pi
    ax[row[i] + 2, col[i]].annotate(
        # '1st cycle',
        '',
        xy=(x, y),
        xycoords='data',
        xytext=(0.05, 0.4),
        textcoords='axes fraction',
        arrowprops=dict(arrowstyle="->", facecolor='gray', edgecolor='gray'))

    idx = int(0.7 * 50)
    x = solution[2]["time"][0][idx]
    y = solution[2]["states"][-1]["q"][1, idx] * 180 / np.pi
    ax[row[i] + 2, col[i]].annotate(
        # '1st cycle',
        '',
        xy=(x, y),
        xycoords='data',
        xytext=(0.85, 0.5),
        textcoords='axes fraction',
        arrowprops=dict(arrowstyle="->", facecolor='black'))

plt.suptitle("Joint Angles")
# global legend in x label
fig.text(0.5, 0.21, 'Time (s)', ha='center', va='center')
# make subplots closers in x and y
# plt.subplots_adjust(wspace=0, hspace=0)
# join label for ax[0,0] and ax[1, 0] in y axis
# write "Shoulder angle" centered for ax[0,0] and ax[1, 0]
fig.text(0.04, 0.75, 'Shoulder angle (deg)', ha='center', va='center', rotation='vertical')
# write "Elbow angle" centered for ax[2,0] and ax[3, 0]
fig.text(0.04, 0.4, 'Elbow angle (deg)', ha='center', va='center', rotation='vertical')

# add legend
handle_list = []
for i, (condition, style) in enumerate(zip(ResultFile, Style)):
    style = style.value
    # plot fake legends
    ax[-1, -1].plot([], [], color=style[0], linestyle=style[1], label=style[3], markerfacecolor=style[4], marker="o",
                    markersize=ms, )
    # get the associated legend of the last plot
    handles, labels = ax[-1, -1].get_legend_handles_labels()
    # get last handles
    handle_list.append(handles[-1])

first_legend = ax[-1, -1].legend(handles=handle_list[:3],
                                 # transparent background
                                 facecolor="none",
                                 # on top
                                 frameon=True,
                                 # below the x label
                                 bbox_to_anchor=(-1.5, -0.32),
                                 )

# add fake legend full-horizon vs sliding horizon
h_full = ax[-1, -1].plot([], [], color="black", linestyle="-", label="Full-horizon", markerfacecolor="black", marker="o",
                            markersize=ms, )[0]
h_sliding = ax[-1, -1].plot([], [], color="black", linestyle="--", label="Sliding-horizon", markerfacecolor="black",
                            marker="o", markersize=ms, )[0]
# get the associated legend of the last plot
handle_list = [h_full, h_sliding]
# second legend
second_legend = ax[-1, -1].legend(handles=handle_list,
                                    # transparent background
                                    facecolor="none",
                                    # on top
                                    frameon=True,
                                    # below the x label
                                    bbox_to_anchor=(-0.2, -0.4),
                                    )

plt.gca().add_artist(first_legend)
plt.gca().add_artist(second_legend)

plt.legend(handles=[an1.arrow_patch, an2.arrow_patch], ncol=1,
           title="Arrows",
           # transparent background
           facecolor="none",
           # on top
           frameon=True,
           # below the x label
           bbox_to_anchor=(0.9, -0.32),
           labels=["first cycle", "last cycle"]
           )

# leave space below figures to put the legend
plt.subplots_adjust(bottom=0.25)

# plt.tight_layout()
export_matplotlib_figure(fig, "all_cycles_q")

# compute max available torque


##figure 2
fig, ax = plt.subplots(4, 3, figsize=(9, 9))
ms = 0.5
row, col = np.unravel_index(np.arange(6), (2, 3))
for i, (condition, style) in enumerate(zip(ResultFile, Style)):
    style = style.value
    solution = load_data(condition.value)
    nb_cycles = solution[2]["states"].__len__()
    linewidth = (np.exp(-1 + np.linspace(0.5, 1, nb_cycles)) ** 5 + np.flip(
        np.exp(-1 + np.linspace(0.5, 1, nb_cycles)) ** 5)) / 2 * 1 / 0.5410425  # + 0.5
    # linewidth[0] = 1.0
    opacity = (np.exp(-1 + np.linspace(0.5, 1, nb_cycles)) ** 5 + np.flip(
        np.exp(-1 + np.linspace(0.5, 1, nb_cycles)) ** 5)) / 2 * 1 / 0.5410425
    # opacity[0] = 1

    for j, sol in enumerate(solution[2]["controls"]):
        ax[row[i], col[i]].plot(
            solution[2]["time"][0],
            sol["tau_minus"][0, :] + sol["tau_plus"][0, :],
            label=style[3], linestyle=style[1], color=style[0], markerfacecolor=style[4], marker="o", markersize=ms,
            linewidth=linewidth[j], alpha=opacity[j])
        # box off and remove axis
        ax[row[i], col[i]].spines["top"].set_visible(False)
        ax[row[i], col[i]].spines["right"].set_visible(False)
        ax[row[i], col[i]].set_ylim(-50, 50)
        # ax[row[i], col[i]].spines["bottom"].set_visible(False)
        # ax[row[i], col[i]].spines["left"].set_visible(False)
        # remove ticks
        # ax[row[i], col[i]].tick_params(axis="both", which="both", bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

    for j, sol in enumerate(solution[2]["controls"]):
        ax[row[i] + 2, col[i]].plot(
            solution[2]["time"][0],
            sol["tau_minus"][1, :] + sol["tau_plus"][1, :],
            label=style[3], linestyle=style[1], color=style[0], markerfacecolor=style[4], marker="o", markersize=ms,
            linewidth=linewidth[j], alpha=opacity[j])
        # box off and remove axis
        ax[row[i] + 2, col[i]].spines["top"].set_visible(False)
        ax[row[i] + 2, col[i]].spines["right"].set_visible(False)
        ax[row[i] + 2, col[i]].set_ylim(-50, 50)
        # ax[row[i]+ 2, col[i]].spines["bottom"].set_visible(False)
        # ax[row[i]+ 2, col[i]].spines["left"].set_visible(False)
        # remove ticks
        # ax[row[i]+ 2, col[i]].tick_params(axis="both", which="both", bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

plt.suptitle("Torques")
# global legend in x label
fig.text(0.5, 0.04, 'Time (s)', ha='center', va='center')
# export_matplotlib_figure(fig, "all_cycles_tau")


def compute_target_load(tau, tau_max):
    return tau / tau_max


def compute_tau_limit(tau_max, mf):
    return (1 - mf) * tau_max


## TORQUE PERCENT
fig, ax = plt.subplots(4, 3, figsize=(9, 9))
ms = 0.5
row, col = np.unravel_index(np.arange(6), (2, 3))
for i, (condition, style) in enumerate(zip(ResultFile, Style)):
    style = style.value
    solution = load_data(condition.value)
    nb_cycles = solution[2]["states"].__len__()
    linewidth = (np.exp(-1 + np.linspace(0.5, 1, nb_cycles)) ** 5 + np.flip(
        np.exp(-1 + np.linspace(0.5, 1, nb_cycles)) ** 5)) / 2 * 1 / 0.5410425  # + 0.5
    # linewidth[0] = 1.0
    opacity = (np.exp(-1 + np.linspace(0.5, 1, nb_cycles)) ** 5 + np.flip(
        np.exp(-1 + np.linspace(0.5, 1, nb_cycles)) ** 5)) / 2 * 1 / 0.5410425
    # opacity[0] = 1

    for j, (sol, sol_state) in enumerate(zip(solution[2]["controls"], solution[2]["states"])):

        tau_max_lim = compute_tau_limit(50, sol_state["tau_plus_mf"][0, :])
        tau_min_lim = compute_tau_limit(-50, sol_state["tau_minus_mf"][0, :])
        tau_plus_percent = sol["tau_plus"][0, :] / tau_max_lim * 100
        tau_minus_percent = sol["tau_minus"][0, :] / tau_min_lim * 100

        ax[row[i], col[i]].plot(
            solution[2]["time"][0],
            tau_plus_percent + tau_minus_percent,
            label=style[3], linestyle=style[1], color=style[0], markerfacecolor=style[4], marker="o", markersize=ms,
            linewidth=linewidth[j], alpha=opacity[j])
        # box off and remove axis
        ax[row[i], col[i]].spines["top"].set_visible(False)
        ax[row[i], col[i]].spines["right"].set_visible(False)
        ax[row[i], col[i]].set_ylim(-100, 100)
        # ax[row[i], col[i]].spines["bottom"].set_visible(False)
        # ax[row[i], col[i]].spines["left"].set_visible(False)
        # remove ticks
        # ax[row[i], col[i]].tick_params(axis="both", which="both", bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

    for j,  (sol, sol_state) in enumerate(zip(solution[2]["controls"], solution[2]["states"])):
        tau_max_lim = compute_tau_limit(50, sol_state["tau_plus_mf"][1, :])
        tau_min_lim = compute_tau_limit(-50, sol_state["tau_minus_mf"][1, :])
        tau_plus_percent = sol["tau_plus"][1, :] / tau_max_lim * 100
        tau_minus_percent = sol["tau_minus"][1, :] / tau_min_lim * 100

        ax[row[i] + 2, col[i]].plot(
            solution[2]["time"][0],
            sol["tau_minus"][1, :] + sol["tau_plus"][1, :],
            label=style[3], linestyle=style[1], color=style[0], markerfacecolor=style[4], marker="o", markersize=ms,
            linewidth=linewidth[j], alpha=opacity[j])
        # box off and remove axis
        ax[row[i] + 2, col[i]].spines["top"].set_visible(False)
        ax[row[i] + 2, col[i]].spines["right"].set_visible(False)
        ax[row[i] + 2, col[i]].set_ylim(-100, 100)
        # ax[row[i]+ 2, col[i]].spines["bottom"].set_visible(False)
        # ax[row[i]+ 2, col[i]].spines["left"].set_visible(False)
        # remove ticks
        # ax[row[i]+ 2, col[i]].tick_params(axis="both", which="both", bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

plt.suptitle("Torques percent")
# global legend in x label
fig.text(0.5, 0.04, 'Time (s)', ha='center', va='center')
# export_matplotlib_figure(fig, "other")



## TORQUE PERCENT
fig, ax = plt.subplots(4, 3, figsize=(9, 9))
ms = 0.5
row, col = np.unravel_index(np.arange(6), (2, 3))
for i, (condition, style) in enumerate(zip(ResultFile, Style)):
    style = style.value
    solution = load_data(condition.value)
    nb_cycles = solution[2]["states"].__len__()
    linewidth = (np.exp(-1 + np.linspace(0.5, 1, nb_cycles)) ** 5 + np.flip(
        np.exp(-1 + np.linspace(0.5, 1, nb_cycles)) ** 5)) / 2 * 1 / 0.5410425  # + 0.5
    # linewidth[0] = 1.0
    opacity = (np.exp(-1 + np.linspace(0.5, 1, nb_cycles)) ** 5 + np.flip(
        np.exp(-1 + np.linspace(0.5, 1, nb_cycles)) ** 5)) / 2 * 1 / 0.5410425
    # opacity[0] = 1

    for j, (sol, sol_state) in enumerate(zip(solution[2]["controls"], solution[2]["states"])):

        tau_max_lim = compute_tau_limit(50, sol_state["tau_plus_mf"][0, :])
        tau_min_lim = compute_tau_limit(-50, sol_state["tau_minus_mf"][0, :])
        tau_plus_percent = sol["tau_plus"][0, :] / tau_max_lim * 100
        tau_minus_percent = sol["tau_minus"][0, :] / tau_min_lim * 100

        ax[row[i], col[i]].plot(
            solution[2]["time"][0],
            tau_max_lim,
            label=style[3], linestyle=style[1], color=style[0], markerfacecolor=style[4], marker="o", markersize=ms,
            linewidth=linewidth[j], alpha=opacity[j])
        ax[row[i], col[i]].plot(
            solution[2]["time"][0],
            tau_min_lim,
            label=style[3], linestyle=style[1], color=style[0], markerfacecolor=style[4], marker="o", markersize=ms,
            linewidth=linewidth[j], alpha=opacity[j])
        # box off and remove axis
        ax[row[i], col[i]].spines["top"].set_visible(False)
        ax[row[i], col[i]].spines["right"].set_visible(False)
        ax[row[i], col[i]].set_ylim(-100, 100)
        # ax[row[i], col[i]].spines["bottom"].set_visible(False)
        # ax[row[i], col[i]].spines["left"].set_visible(False)
        # remove ticks
        # ax[row[i], col[i]].tick_params(axis="both", which="both", bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

    for j,  (sol, sol_state) in enumerate(zip(solution[2]["controls"], solution[2]["states"])):
        tau_max_lim = compute_tau_limit(50, sol_state["tau_plus_mf"][1, :])
        tau_min_lim = compute_tau_limit(-50, sol_state["tau_minus_mf"][1, :])
        tau_plus_percent = sol["tau_plus"][1, :] / tau_max_lim * 100
        tau_minus_percent = sol["tau_minus"][1, :] / tau_min_lim * 100

        ax[row[i] + 2, col[i]].plot(
            solution[2]["time"][0],
            tau_max_lim,
            label=style[3], linestyle=style[1], color=style[0], markerfacecolor=style[4], marker="o", markersize=ms,
            linewidth=linewidth[j], alpha=opacity[j])
        ax[row[i] + 2, col[i]].plot(
            solution[2]["time"][0],
            tau_min_lim,
            label=style[3], linestyle=style[1], color=style[0], markerfacecolor=style[4], marker="o", markersize=ms,
            linewidth=linewidth[j], alpha=opacity[j])
        # box off and remove axis
        ax[row[i] + 2, col[i]].spines["top"].set_visible(False)
        ax[row[i] + 2, col[i]].spines["right"].set_visible(False)
        ax[row[i] + 2, col[i]].set_ylim(-100, 100)
        # ax[row[i]+ 2, col[i]].spines["bottom"].set_visible(False)
        # ax[row[i]+ 2, col[i]].spines["left"].set_visible(False)
        # remove ticks
        # ax[row[i]+ 2, col[i]].tick_params(axis="both", which="both", bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

plt.suptitle("Torques Limits")
# global legend in x label
fig.text(0.5, 0.04, 'Time (s)', ha='center', va='center')

plt.show()

