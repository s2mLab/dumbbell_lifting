from enum import Enum
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle


plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "Times New Roman"
# size font
plt.rcParams["font.size"] = 15

# data
class ResultFile(Enum):
    FULL_WINDOW_FATIGUE_TORQUE = "FULL_WINDOW_FATIGUE_TORQUE_NO_STABILIZATION_20_$TauXia_all$.pkl"
    FULL_WINDOW_ONLY_FATIGUE = "FULL_WINDOW_ONLY_FATIGUE_NO_STABILIZATION_17_$TauXia_mf$.pkl"
    FULL_WINDOW_ONLY_TORQUE = "FULL_WINDOW_ONLY_TORQUE_NO_STABILIZATION_14_$TauXia_tau$.pkl"
    NMPC_FATIGUE_TORQUE = "CONDITIONS_FATIGUE_TORQUE_$TauXia_all$.pkl"
    NMPC_ONLY_FATIGUE = "CONDITIONS_ONLY_FATIGUE_$TauXia_mf$.pkl"
    NMPC_ONLY_TORQUE = "CONDITIONS_ONLY_TORQUE_$TauXia_tau$.pkl"


class CostFunction(Enum):
    FATIGUE_TORQUE = r'$\Phi^{m_f,\tau}$'
    ONLY_FATIGUE = r'$\Phi^{m_f}$'
    ONLY_TORQUE = r'$\Phi^{\tau}$'


class CustomColor:
    Green = "#00cc96"
    Yellow = "#ffa15a"
    Red = "#ef553b"
    Gray = "tab:gray"


class Style(Enum):
    # linecolor, linestyle, label, cost_function, facecolor
    FULL_WINDOW_FATIGUE_TORQUE = ("tab:blue", "-", r'Full horizon $\int \tau^2 + {m_f}^2 + {q_0}² \; dt$', CostFunction.FATIGUE_TORQUE.value, "tab:blue")
    FULL_WINDOW_ONLY_FATIGUE = ("tab:orange", "-", r'Full horizon $\int {m_f}^2 + {q_0}² \; dt$', CostFunction.ONLY_FATIGUE.value, "tab:orange")
    FULL_WINDOW_ONLY_TORQUE = ("tab:green",  "-", r'Full horizon $\int \tau^2 + {q_0}² \; dt$', CostFunction.ONLY_TORQUE.value, "tab:green")
    FINITE_HORIZON_FATIGUE_TORQUE = ("tab:blue", "--", r'Finite horizon $\int \tau^2 + {m_f}^2 + {q_0}² \; dt$', CostFunction.FATIGUE_TORQUE.value,
                                     "white")
    FINITE_HORIZON_ONLY_FATIGUE = ("tab:orange", "--", r'Finite horizon $\int {m_f}^2 + {q_0}² \; dt$', CostFunction.ONLY_FATIGUE.value, "white")
    FINITE_HORIZON_ONLY_TORQUE = ("tab:green", "--", r'Finite horizon $\int \tau^2 + {q_0}² \; dt$', CostFunction.ONLY_TORQUE.value, "white")


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


def plot_pool(result_file: ResultFile, export:bool, show:bool):
    data = load_data(result_file.value)
    sol = data[2]
    keys_set = [["tau_plus_ma", "tau_plus_mr", "tau_plus_mf"],
                ["tau_minus_ma", "tau_minus_mr", "tau_minus_mf"]]
    n_dof = 2
    nb_cycles = sol["time"].__len__()
    colors = (CustomColor.Green, CustomColor.Yellow, CustomColor.Red)

    # concatenate arrays in sol["time"][0] and add the time of previous iterations to the next one like in cumsum
    time = np.array([])
    for i in range(len(sol["time"])):
        time = np.concatenate((time, sol["time"][i] + time[-1] if time.size else sol["time"][i]))

    # same for key sets
    states = {}
    for key in ["tau_plus_ma", "tau_plus_mr", "tau_plus_mf", "tau_minus_ma", "tau_minus_mr", "tau_minus_mf"]:
        for i in range(len(sol["states"])):
            if i == 0:
                states[key] = sol["states"][i][key]
            else:
                states[key] = np.concatenate((states[key], sol["states"][i][key]), axis=1)

    # same for key sets
    controls = {}
    for key in ["tau_minus", "tau_plus"]:
        for i in range(len(sol["controls"])):
            if i == 0:
                controls[key] = sol["controls"][i][key]
            else:
                controls[key] = np.concatenate((controls[key], sol["controls"][i][key]), axis=1)
        # find nan locations
        nan_locations = np.where(np.isnan(controls[key]))
        # deduce previous index of nans
        next_index_row = nan_locations[0]
        next_index_col = nan_locations[1] + 1
        # replace nan with previous value
        for i in range(len(next_index_row)):
            if next_index_col[i] < controls[key].shape[1]:
                print(controls[key][nan_locations[0][i], nan_locations[1][i]], controls[key][next_index_row[i], next_index_col[i]])
                controls[key][nan_locations[0][i], nan_locations[1][i]] = controls[key][next_index_row[i], next_index_col[i]]
                print(controls[key][nan_locations[0][i], nan_locations[1][i]])
            else:
                controls[key][nan_locations[0][i], nan_locations[1][i]] = controls[key][
                    next_index_row[i], nan_locations[1][i] - 1]



    fig, ax = plt.subplots(2, n_dof)
    fig.set_size_inches(16, 9)
    font_size = 16

    for j in range(n_dof):
        print(j, n_dof)
        for i, keys in enumerate(keys_set):
            tau_key = "tau_plus" if i == 0 else "tau_minus"
            tau_max = 50 if i == 0 else -50
            print(j, i,  keys)
            ax[i, j].stackplot(np.array(time, dtype=np.float64), np.vstack([states[key][j, :] * 100 for key in keys]), colors=colors,
                               alpha=0.4)
            ax[i, j].plot(np.array(time, dtype=np.float64),
                          controls[tau_key][j, :] * 100 / tau_max,
                          color="tab:blue",
                          linewidth=1.5,
                          )
            ax[i, j].set_ylim([0, 101])
            ax[i, j].set_xlim([0, time[-1]])
            ax[i, j].spines["top"].set_visible(False)
            ax[i, j].spines["right"].set_visible(False)
            ax[i, j].tick_params(axis="both", labelsize=font_size)

    ax[0, 1].legend(
        (
            r"$m_a$",
            r"$m_r$",
            r"$m_f$",
            r'$\tilde{\tau}^{+,-}$',
            "_",
            "_",
            "_",
            "_",
            "_",
            "_",
            "_",
            "_",
            "_",
            "_",
            "_",
            "_",
            "_",
            "_",
            "_",
            r"$TL$",
        ),
        loc="upper right",
        fontsize=font_size,
        framealpha=0.9,
        title_fontsize=20,
    )

    ax[0, 0].set_title(r"Shoulder", fontsize=font_size)
    ax[0, 1].set_title(r"Elbow", fontsize=font_size)

    ax[1, 0].set_xlabel('Time (s)', fontsize=font_size)
    ax[1, 1].set_xlabel('Time (s)', fontsize=font_size)
    ax[0, 0].set_ylabel('Flexion Actuation Level (\%)', fontsize=font_size)
    ax[1, 0].set_ylabel('Extension Actuation Level (\%)', fontsize=font_size)


    # plot vline each cycles for each dof
    for j in range(n_dof):
        for jj in range(2):
            for i in range(1, nb_cycles):
                vline_x = i
                ax[j, jj].axvline(x=vline_x, color="k", linestyle="--", linewidth=0.5)

    # show only ints on xticks
    # import MaxNLocator from matplotlib.ticker
    from matplotlib.ticker import MaxNLocator
    for j in range(n_dof):
        for jj in range(2):
            ax[j, jj].xaxis.set_major_locator(MaxNLocator(integer=True))

    if export:
        export_matplotlib_figure(fig, "pools" + result_file.name)
    if show:
        plt.show()

    return ax

def plot_sum_pool(result_file, show=True, export=False, ax=None, style=None):
    data = load_data(result_file.value)
    sol = data[2]
    keys_set = [["tau_plus_ma", "tau_plus_mr", "tau_plus_mf"],
                ["tau_minus_ma", "tau_minus_mr", "tau_minus_mf"]]
    n_dof = 2
    nb_cycles = sol["time"].__len__()
    colors = (CustomColor.Green, CustomColor.Yellow, CustomColor.Red)

    # concatenate arrays in sol["time"][0] and add the time of previous iterations to the next one like in cumsum
    time = np.array([])
    for i in range(len(sol["time"])):
        time = np.concatenate((time, sol["time"][i] + time[-1] if time.size else sol["time"][i]))

    # same for key sets
    states = {}
    for key in ["tau_plus_ma", "tau_plus_mr", "tau_plus_mf", "tau_minus_ma", "tau_minus_mr", "tau_minus_mf"]:
        for i in range(len(sol["states"])):
            if i == 0:
                states[key] = sol["states"][i][key]
            else:
                states[key] = np.concatenate((states[key], sol["states"][i][key]), axis=1)

    # same for key sets
    controls = {}
    for key in ["tau_minus", "tau_plus"]:
        for i in range(len(sol["controls"])):
            if i == 0:
                controls[key] = sol["controls"][i][key]
            else:
                controls[key] = np.concatenate((controls[key], sol["controls"][i][key]), axis=1)
        # find nan locations
        nan_locations = np.where(np.isnan(controls[key]))
        # deduce previous index of nans
        next_index_row = nan_locations[0]
        next_index_col = nan_locations[1] + 1
        # replace nan with previous value
        for i in range(len(next_index_row)):
            if next_index_col[i] < controls[key].shape[1]:
                # print(controls[key][nan_locations[0][i], nan_locations[1][i]],
                #       controls[key][next_index_row[i], next_index_col[i]])
                controls[key][nan_locations[0][i], nan_locations[1][i]] = controls[key][
                    next_index_row[i], next_index_col[i]]
                # print(controls[key][nan_locations[0][i], nan_locations[1][i]])
            else:
                controls[key][nan_locations[0][i], nan_locations[1][i]] = controls[key][
                    next_index_row[i], nan_locations[1][i] - 1]
    if ax is None:
        fig, ax = plt.subplots(2, n_dof)
        fig.set_size_inches(16, 9)
    else:
        fig = ax[0, 0].get_figure()

    font_size = 16

    key_sets = [['tau_plus_ma', 'tau_plus_mr', 'tau_plus_mf'],['tau_minus_ma', 'tau_minus_mr', 'tau_minus_mf']]

    for j in range(n_dof):
        print(j, n_dof)
        for i, key_set in enumerate(key_sets):

            ax[i, j].plot(
                np.array(time, dtype=np.float64),
                np.abs(np.sum(np.vstack([states[key][j, :] * 100 for key in key_set]), axis=0) - 100),
                color=style[0],
                linestyle=style[1],
            )
            # ax[i, j].set_ylim([0, 101])
            # ax[i, j].set_xlim([0, time[-1]])
            ax[i, j].spines["top"].set_visible(False)
            ax[i, j].spines["right"].set_visible(False)
            ax[i, j].tick_params(axis="both", labelsize=font_size)

    ax[0, 0].set_title(r"Shoulder", fontsize=font_size)
    ax[0, 1].set_title(r"Elbow", fontsize=font_size)

    ax[1, 0].set_xlabel('Time (s)', fontsize=font_size)
    ax[1, 1].set_xlabel('Time (s)', fontsize=font_size)
    ax[0, 0].set_ylabel('Flexion Actuation Level (\%)', fontsize=font_size)
    ax[1, 0].set_ylabel('Extension Actuation Level (\%)', fontsize=font_size)

    # set y-axis to log scales
    ax[0, 0].set_yscale("log")
    ax[0, 1].set_yscale("log")
    ax[1, 0].set_yscale("log")
    ax[1, 1].set_yscale("log")

    # show only ints on xticks
    # import MaxNLocator from matplotlib.ticker
    from matplotlib.ticker import MaxNLocator
    for j in range(n_dof):
        for jj in range(2):
            ax[j, jj].xaxis.set_major_locator(MaxNLocator(integer=True))

    if export:
        export_matplotlib_figure(fig, "pools_sum" + result_file.name)
    if show:
        plt.show()

    return ax

def compute_sum_of_pools(result_file):
    data = load_data(result_file.value)
    sol = data[2]
    keys_set = [["tau_plus_ma", "tau_plus_mr", "tau_plus_mf"],
                ["tau_minus_ma", "tau_minus_mr", "tau_minus_mf"]]
    n_dof = 2

    sum_of_pools = []
    for states in sol["states"]:
        sum_of_this_cycle = np.zeros(4)
        for i, keys in enumerate(keys_set):
            for dof in range(n_dof):
                sum_of_this_cycle[dof + i * n_dof] = np.abs(np.sum(np.vstack([states[key][dof, :] * 1 for key in keys]), axis=0) - 1)[-1]

        sum_of_pools.append(sum_of_this_cycle)

    return sum_of_pools

# for result_file in ResultFile:
#     plot_pool(result_file=result_file, show=True, export=False)

# compute sum of pools
all_sum_of_pools = dict()
for (result_file, style) in zip(ResultFile, Style):
    print("##################")
    print(result_file.name)
    print("##################")
    all_sum_of_pools[result_file.name] = compute_sum_of_pools(result_file=result_file)
    print(all_sum_of_pools[result_file.name].__len__())
    print(all_sum_of_pools[result_file.name][-1])



# ax= None
# for (result_file, style) in zip(ResultFile, Style):
#     print("##################")
#     print(result_file.name)
#     print("##################")
#     ax = plot_sum_pool(result_file=result_file, show=False, export=False, ax=ax, style=style.value)
# plt.legend([ e.value[2] for e in Style])
# plt.show()
