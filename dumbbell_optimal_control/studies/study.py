from typing import Protocol
import os

from bioptim import Solution
import numpy as np
from matplotlib import pyplot as plt

from .study_configuration import StudyConfiguration
from .ocp import DataType


class CustomColor:
    Green = "#00cc96"
    Yellow = "#ffa15a"
    Red = "#ef553b"
    Gray = "tab:gray"


class Conditions(Protocol):
    name: str
    value: StudyConfiguration


class Study:
    """
    This class supports the results of a study. It can plot the results, save the results, and print the results.

    Attributes
    ----------
    name: str
        The name of the study
    _has_run: bool
        A boolean indicating if the study has been run
    _plots_are_prepared: bool
        A boolean indicating if the plots have been prepared
    _plots_are_prepared: bool
        A boolean indicating if the plots have been prepared
    conditions: StudyConfiguration
        The configuration of the study
    solution : list[Solution, ...] | tuple[Solution, list[Solution, ...], list[Solution, ...]]
        self.solution[x][y] Condition x, if y is 0: then full solution, if y is 1: then a tuple of all windows OCP
        solutions, if y is 2: then a list of all cycle solution
    """
    def __init__(self, conditions: Conditions):
        self.name = conditions.name
        self._has_run: bool = False
        self._plots_are_prepared: bool = False
        self.conditions: StudyConfiguration = conditions.value
        self.solution: list[Solution, ...] | tuple[Solution, list[Solution, ...], list[Solution, ...]] = []

    def run(self):
        for condition in self.conditions.studies:
            self.solution.append(condition.perform())
        self._has_run = True

    def plot_data_stacked_per_window(self):
        color = ["b", "g", "r", "y", "m", "c", "k"]
        import matplotlib.pyplot as plt

        # opacity and linewidth as function of cycle considered
        linewidth = np.exp(-1 + np.linspace(0.5, 1, len(self.solution[0][1]))) + 0.5
        linewidth[0] = 1.5
        opacity = np.linspace(0.3, 1, len(self.solution[0][1]))

        for key in self.solution[0][0].states.keys():
            if key == "all":
                continue
            for j, sol in enumerate(self.solution[0][1]):
                if sol.status == 1:
                    continue
                if j == 0:
                    fig, ax = plt.subplots(1, len(self.solution[0][0].states[key]))
                    fig.suptitle(f"States: {key}")
                for i, state in enumerate(sol.states[key]):
                    ax[i].plot(sol.time, state, label=f"{key}_{i}", color=color[0], linewidth=linewidth[j],
                               alpha=opacity[j])
                    ax[i].set_title(f"State {key}{i}")
                    ax[i].set_xlabel("Time (s)")

        for key in self.solution[0][0].controls.keys():
            if key == "all":
                continue
            for j, sol in enumerate(self.solution[0][1]):
                if sol.status == 1:
                    continue
                if j == 0:
                    fig, ax = plt.subplots(1, len(self.solution[0][0].controls[key]))
                    fig.suptitle(f"Control: {key}")
                for i, state in enumerate(sol.controls[key]):
                    ax[i].plot(sol.time, state, label=f"{key}_{i}", color=color[0], linewidth=linewidth[j],
                               alpha=opacity[j])
                    ax[i].set_title(f"Control {key}{i}")
                    ax[i].set_xlabel("Time (s)")

        for j, sol in enumerate(self.solution[0][1]):
            if sol.status == 1:
                continue
            if j == 0:
                fig, ax = plt.subplots(1, len(self.solution[0][0].controls[key]))
                fig.suptitle("TL + mf")
            for i, (tau, state) in enumerate(zip(sol.controls["tau_plus"], sol.states["tau_plus_mf"])):
                ax[i].plot(sol.time, state, label=f"{key}_{i}", color=color[0], linewidth=linewidth[j],
                           alpha=opacity[j])
                ax[i].plot(sol.time, tau / 50, label=f"{key}_{i}", color=color[1], linewidth=linewidth[j],
                           alpha=opacity[j])
                ax[i].plot(sol.time, tau / 50 + state, label=f"{key}_{i}", color=color[2], linewidth=linewidth[j],
                           alpha=opacity[j])
                # horizontal line at y=1
                ax[i].axhline(y=1, color="k", linestyle="--")

                ax[i].set_title(f"TL + mf {i}")
                ax[i].set_xlabel("Time (s)")

        # plt.legend()
        plt.show()

    def plot_data_stacked_per_cycle(self):
        color = ["b", "g", "r", "y", "m", "c", "k"]
        import matplotlib.pyplot as plt

        # opacity and linewidth as function of cycle considered
        linewidth = np.exp(-1 + np.linspace(0.5, 1, len(self.solution[0][1]))) + 0.5
        linewidth[0] = 1.5
        opacity = np.linspace(0.3, 1, len(self.solution[0][1]))
        cyle_solutions = self.solution[0][2]
        for key in self.solution[0][0].states.keys():
            if key == "all":
                continue
            for j, sol in enumerate(cyle_solutions):
                if sol.status == 1:
                    continue
                if j == 0:
                    fig, ax = plt.subplots(1, len(self.solution[0][0].states[key]))
                    fig.suptitle(f"States: {key}")
                for i, state in enumerate(sol.states[key]):
                    ax[i].plot(sol.time, state, label=f"{key}_{i}", color=color[0], linewidth=linewidth[j],
                               alpha=opacity[j])
                    ax[i].set_title(f"State {key}{i}")
                    ax[i].set_xlabel("Time (s)")

        for key in self.solution[0][0].controls.keys():
            if key == "all":
                continue
            for j, sol in enumerate(cyle_solutions):
                if sol.status == 1:
                    continue
                if j == 0:
                    fig, ax = plt.subplots(1, len(self.solution[0][0].controls[key]))
                    fig.suptitle(f"Control: {key}")
                for i, state in enumerate(sol.controls[key]):
                    ax[i].plot(sol.time, state, label=f"{key}_{i}", color=color[0], linewidth=linewidth[j],
                               alpha=opacity[j])
                    ax[i].set_title(f"Control {key}{i}")
                    ax[i].set_xlabel("Time (s)")

        for j, sol in enumerate(cyle_solutions):
            if sol.status == 1:
                continue
            if j == 0:
                fig, ax = plt.subplots(1, len(self.solution[0][0].controls[key]))
                fig.suptitle("TL + mf")
            for i, (tau, state) in enumerate(zip(sol.controls["tau_plus"], sol.states["tau_plus_mf"])):
                ax[i].plot(sol.time, state, label=f"{key}_{i}", color=color[0], linewidth=linewidth[j],
                           alpha=opacity[j])
                ax[i].plot(sol.time, tau / 50, label=f"{key}_{i}", color=color[1], linewidth=linewidth[j],
                           alpha=opacity[j])
                ax[i].plot(sol.time, tau / 50 + state, label=f"{key}_{i}", color=color[2], linewidth=linewidth[j],
                           alpha=opacity[j])
                # horizontal line at y=1
                ax[i].axhline(y=1, color="k", linestyle="--")

                ax[i].set_title(f"TL + mf {i}")
                ax[i].set_xlabel("Time (s)")

        # plt.legend()
        plt.show()

    def plot_first_and_last_cycles(self):

        color = ["g", "r"]
        label = ["first", "last"]
        # turn this into a dict
        options = dict(color=color, label=label)

        import matplotlib.pyplot as plt

        # opacity and linewidth as function of cycle considered
        linewidth = np.exp(-1 + np.linspace(0.5, 1, len(self.solution[0][1]))) + 0.5
        linewidth[0] = 1.5
        opacity = np.linspace(0.3, 1, len(self.solution[0][1]))

        cycle_solutions = self.solution[0][2]
        first_and_last_cycles = [cycle_solutions[0], cycle_solutions[-1]]

        for key in self.solution[0][0].states.keys():
            if key == "all":
                continue
            for j, sol in enumerate(first_and_last_cycles):
                if sol.status == 1:
                    continue
                if j == 0:
                    fig, ax = plt.subplots(1, len(self.solution[0][0].states[key]))
                    fig.suptitle(f"States: {key}")
                for i, state in enumerate(sol.states[key]):
                    ax[i].plot(sol.time, state, label=label[j], color=color[j], linewidth=linewidth[j],
                               alpha=opacity[j])
                    ax[i].set_title(f"State {key}{i}")
                    ax[i].set_xlabel("Time (s)")
            plt.legend()

        for key in self.solution[0][0].controls.keys():
            if key == "all":
                continue
            for j, sol in enumerate(first_and_last_cycles):
                if sol.status == 1:
                    continue
                if j == 0:
                    fig, ax = plt.subplots(1, len(self.solution[0][0].controls[key]))
                    fig.suptitle(f"Control: {key}")
                for i, state in enumerate(sol.controls[key]):
                    ax[i].plot(sol.time, state, label=label[j], color=color[j], linewidth=linewidth[j],
                               alpha=opacity[j])
                    ax[i].set_title(f"Control {key}{i}")
                    ax[i].set_xlabel("Time (s)")

        for j, sol in enumerate(first_and_last_cycles):
            if sol.status == 1:
                continue
            if j == 0:
                fig, ax = plt.subplots(1, len(self.solution[0][0].controls[key]))
                fig.suptitle("TL + mf")
            for i, (tau, state) in enumerate(zip(sol.controls["tau_plus"], sol.states["tau_plus_mf"])):
                ax[i].plot(sol.time, state, label=label[j], color=color[j], linewidth=linewidth[j],
                           alpha=opacity[j])
                ax[i].plot(sol.time, tau / 50, label=label[j], color=color[j], linewidth=linewidth[j],
                           alpha=opacity[j])
                ax[i].plot(sol.time, tau / 50 + state, label=label[j], color=color[j], linewidth=linewidth[j],
                           alpha=opacity[j])
                # horizontal line at y=1
                ax[i].axhline(y=1, color="k", linestyle="--")

                ax[i].set_title(f"TL + mf {i}")
                ax[i].set_xlabel("Time (s)")

        plt.show()




    @staticmethod
    def compute_target_load(tau, tau_max):
        return tau / tau_max

    @staticmethod
    def compute_tau_limit(tau_max, mf):
        return (1 - mf) * tau_max

    def plot_torques(self):
        color = ["b", "g", "r", "y", "m", "c", "k"]
        import matplotlib.pyplot as plt

        sol = self.solution[0][0]
        # subplot for each dof
        n_dof = sol.states['q'].shape[0]
        time = np.arange(0, len(sol.states["q"][0, :])) * self.solution[0][1][0].time[1]
        fig, ax = plt.subplots(1, n_dof)

        for i in range(n_dof):
            ax[i].plot(time, sol.controls["tau_plus"][i, :], color=color[0], label="tau_plus")
            tau_lim = self.compute_tau_limit(50, sol.states["tau_plus_mf"][i, :])
            ax[i].plot(time, tau_lim, label="tau_lim", linestyle="--", color="k")

            ax[i].plot(time, sol.controls["tau_minus"][i, :], color=color[1], label="tau_minus")
            tau_lim = self.compute_tau_limit(-50, sol.states["tau_minus_mf"][i, :])
            ax[i].plot(time, tau_lim, linestyle="--", color="k")

            if i == 0:
                ax[i].set_title(f"Shoulder")
            elif i == 1:
                ax[i].set_title(f"Elbow")

            ax[i].set_xlabel("Time (s)")
        ax[0].set_ylabel("Torque (Nm)")

        # plot a vertical line for each cycle
        for i in range(1, len(self.solution[0][1])):
            for j in range(n_dof):
                for jj in range(2):
                    n = self.conditions.studies[0].n_round_trips
                    t_end = self.solution[0][1][0].time[-1]
                    vline_x = [t_end/n * i for i in range(1, n+1)]
                    for vline in vline_x:
                        ax[j].axvline(x=vline, color="k", linestyle="--", linewidth=0.5)

        plt.legend()
        plt.show()

    def plot_pools(self):
        sol = self.solution[0][0]
        keys_set = [["tau_plus_ma", "tau_plus_mr", "tau_plus_mf"],
                    ["tau_minus_ma", "tau_minus_mr", "tau_minus_mf"]]
        n_dof = sol.states['q'].shape[0]

        colors = (CustomColor.Green, CustomColor.Yellow, CustomColor.Red)
        time = np.arange(0, len(sol.states["q"][0, :])) * self.solution[0][1][0].time[1]

        fig, ax = plt.subplots(2, n_dof)
        fig.set_size_inches(16, 9)

        # plt.rcParams["text.usetex"] = True
        plt.rcParams["text.latex.preamble"] = (r"\usepackage{siunitx}",)
        plt.rcParams["font.family"] = "Times New Roman"
        font_size = 12

        for j in range(n_dof):
            for i, keys in enumerate(keys_set):
                tau_key = "tau_plus" if i == 0 else "tau_minus"
                tau_max = 50 if i == 0 else -50
                ax[i, j].stackplot(time, np.vstack((sol.states[key][j, :] * 100 for key in keys)), colors=colors,
                                   alpha=0.4)
                ax[i, j].plot(time,
                              sol.controls[tau_key][j, :] * 100 / tau_max,
                              color="tab:blue",
                              linewidth=4,
                              )
                ax[i, j].set_xlabel(r"Time (\SI{}{\second})", fontsize=font_size)
                ax[i, j].set_ylabel(r"Level (\SI{}{\percent})", fontsize=font_size)
                ax[i, j].set_ylim([0, 101])
                ax[i, j].set_xlim([0, time[-1]])
                ax[i, j].spines["top"].set_visible(False)
                ax[i, j].spines["right"].set_visible(False)
                ax[i, j].tick_params(axis="both", labelsize=font_size)
                ax[i, j].legend()

        # plot a vertical line for each cycle
        for i in range(1, len(self.solution[0][1])):
            for j in range(n_dof):
                for jj in range(2):
                    ax[jj, j].axvline(x=self.solution[0][1][0].time[-1] * (i+1), color="k", linestyle="--", alpha=0.5)

        ax[0, 0].set_title(r"Shoulder \tau_plus", fontsize=font_size)
        ax[0, 1].set_title(r"Elbow \tau_plus", fontsize=font_size)
        ax[1, 0].set_title(r"Shoulder \tau_minus", fontsize=font_size)
        ax[1, 1].set_title(r"Elbow \tau_minus", fontsize=font_size)
        # ax[i, j].legend(
        #     (
        #         "$m_a$",
        #         "$m_r$",
        #         "$m_f$",
        #         "$m_a+m_r+m_f$",
        #         "_",
        #         "_",
        #         "_",
        #         "_",
        #         "$TL$",
        #         "_",
        #         "_",
        #         "_",
        #         "_",
        #         "_",
        #         "_",
        #         "_",
        #         "_",
        #         "_",
        #         "_",
        #         "_",
        #         "$TL$",
        #     ),
        #     loc="upper right",
        #     fontsize=font_size,
        #     framealpha=0.9,
        #     title_fontsize=20,
        # )

        # plot a vertical line for each cycle
        for i in range(1, len(self.solution[0][1])):
            for j in range(n_dof):
                for jj in range(2):
                    n = self.conditions.studies[0].n_round_trips
                    t_end = self.solution[0][1][0].time[-1]
                    vline_x = [t_end / n * i for i in range(1, n + 1)]
                    for vline in vline_x:
                        ax[jj, j].axvline(x=vline, color="k", linestyle="--", linewidth=0.5, alpha=0.5)

        # if maximized:
        #     plt.get_current_fig_manager().window.showMaximized()

        # if self.study.plot_options.save_name:
        #     plt.savefig(f"{self.prepare_and_get_results_dir()}/{self.study.plot_options.save_name}.png", dpi=100)
        #     plt.savefig(f"{self.prepare_and_get_results_dir()}/{self.study.plot_options.save_name}.pdf", format="pdf")
        #     plt.savefig(f"{self.prepare_and_get_results_dir()}/{self.study.plot_options.save_name}.eps", format="eps")

        plt.show()

    def print_results(self):
        print("Number of iterations")
        for study, sol in zip(self.conditions.studies, self.solution):
            print(f"\t{study.name} = {sol.iterations}")

        print("Total time to optimize")
        for study, sol in zip(self.conditions.studies, self.solution):
            print(f"\t{study.name} = {sol.real_time_to_optimize:0.3f} second")

        print("Mean time per iteration to optimize")
        for study, sol in zip(self.conditions.studies, self.solution):
            print(f"\t{study.name} = {sol.real_time_to_optimize / sol.iterations:0.3f} second")

    def plot_cost(self):

        solutions = self.solution[0][1]

        for i, sol in enumerate(solutions):
            sol.detailed_cost_values()

        torque_cost = [sol.detailed_cost[0]["cost_value_weighted"] for sol in solutions]
        tau_minus_mf_cost = [sol.detailed_cost[1]["cost_value_weighted"] for sol in solutions]
        tau_plus_mf_cost = [sol.detailed_cost[2]["cost_value_weighted"] for sol in solutions]
        shoulder_state_cost = [sol.detailed_cost[3]["cost_value_weighted"] for sol in solutions]

        plt.figure()
        plt.plot(torque_cost, label="Torque cost", marker="o")
        plt.plot(tau_minus_mf_cost, label="Tau minus cost", marker="o")
        plt.plot(tau_plus_mf_cost, label="Tau plus cost", marker="o")
        plt.plot(shoulder_state_cost, label="Shoulder state cost", marker="o")
        # x-axis only show ticks for int
        plt.xticks(np.arange(len(solutions)), np.arange(1, len(solutions) + 1))
        plt.xlabel("Cycle")
        plt.ylabel("Cost")
        plt.legend()
        plt.show()

    def generate_latex_table(self):
        if not self._has_run:
            raise RuntimeError("run() must be called before generating the latex table")

        table = (
            f"\\documentclass{{article}}\n"
            f"\n"
            f"\\usepackage{{amsmath}}\n"
            f"\\usepackage{{amssymb}}\n"
            f"\\usepackage[table]{{xcolor}}\n"
            f"\\usepackage{{threeparttable}}\n"
            f"\\usepackage{{makecell}}\n"
            f"\\definecolor{{lightgray}}{{gray}}{{0.91}}\n"
            f"\n\n"
            f"% Aliases\n"
            f"\\newcommand{{\\rmse}}{{RMSE}}\n"
            f"\\newcommand{{\\ocp}}{{OCP}}\n"
            f"\\newcommand{{\\controls}}{{\\mathbf{{u}}}}\n"
            f"\\newcommand{{\\states}}{{\\mathbf{{x}}}}\n"
            f"\\newcommand{{\\statesDot}}{{\\mathbf{{\\dot{{x}}}}}}\n"
            f"\\newcommand{{\\q}}{{\\mathbf{{q}}}}\n"
            f"\\newcommand{{\\qdot}}{{\\mathbf{{\\dot{{q}}}}}}\n"
            f"\\newcommand{{\\qddot}}{{\\mathbf{{\\ddot{{q}}}}}}\n"
            f"\\newcommand{{\\f}}{{\\mathbf{{f}}}}\n"
            f"\\newcommand{{\\taupm}}{{\\tau^{{\\pm}}}}\n"
            f"\\newcommand{{\\tauns}}{{\\tau^{{\\times}}}}\n"
            f"\n"
            f"\\newcommand{{\\condition}}{{C/}}\n"
            f"\\newcommand{{\\noFatigue}}{{\\varnothing}}\n"
            f"\\newcommand{{\\qcc}}{{4\\textsubscript{{CC}}}}\n"
            f"\\newcommand{{\\pe}}{{P\\textsubscript{{E}}}}\n"
            f"\\newcommand{{\\condTau}}{{{{\\condition}}{{\\tau}}{{}}}}\n"
            f"\\newcommand{{\\condTauNf}}{{{{\\condition}}{{\\tau}}{{\\noFatigue}}}}\n"
            f"\\newcommand{{\\condTauQcc}}{{{{\\condition}}{{\\tau}}{{\\qcc}}}}\n"
            f"\\newcommand{{\\condTauPe}}{{{{\\condition}}{{\\tau}}{{\\pe}}}}\n"
            f"\\newcommand{{\\condTaupm}}{{{{\\condition}}{{\\taupm}}{{}}}}\n"
            f"\\newcommand{{\\condTaupmQcc}}{{{{\\condition}}{{\\taupm}}{{\\qcc}}}}\n"
            f"\\newcommand{{\\condTaupmPe}}{{{{\\condition}}{{\\taupm}}{{\\pe}}}}\n"
            f"\\newcommand{{\\condTauns}}{{{{\\condition}}{{\\tauns}}{{}}}}\n"
            f"\\newcommand{{\\condTaunsQcc}}{{{{\\condition}}{{\\tauns}}{{\\qcc}}}}\n"
            f"\\newcommand{{\\condTaunsPe}}{{{{\\condition}}{{\\tauns}}{{\\pe}}}}\n"
            f"\\newcommand{{\\condAlpha}}{{{{\\condition}}{{\\alpha}}{{}}}}\n"
            f"\\newcommand{{\\condAlphaNf}}{{{{\\condition}}{{\\alpha}}{{\\noFatigue}}}}\n"
            f"\\newcommand{{\\condAlphaQcc}}{{{{\\condition}}{{\\alpha}}{{\\qcc}}}}\n"
            f"\\newcommand{{\\condAlphaPe}}{{{{\\condition}}{{\\alpha}}{{\\pe}}}}\n"
            f"\n\n"
            f"\\begin{{document}}\n"
            f"\n"
            f"\\begin{{table}}[!ht]\n"
            f" \\rowcolors{{1}}{{}}{{lightgray}}\n"
            f" \\caption{{Comparaison des métriques d'efficacité et de comportement entre les modèles de fatigue "
            f"appliqués sur une dynamique musculaire ou articulaire lors de la résolution d'un \\ocp{{}}}}\n"
            f" \\label{{table:faisabilite}}\n"
            f" \\begin{{threeparttable}}\n"
            f"  \\begin{{tabular}}{{lccccc}}\n"
            f"   \\hline\n"
            f"   \\bfseries Condition & "
            f"\\bfseries\\makecell[c]{{Nombre de\\\\variables/\\\\contraintes}} & "
            f"\\bfseries\\makecell[c]{{Nombre\\\\d'itérations}} & "
            f"\\bfseries\\makecell[c]{{Temps\\\\de calcul\\\\(s)}} & "
            f"\\bfseries\\makecell[c]{{Temps moyen\\\\par itération\\\\(s/iteration)}} & "
            f"\\bfseries\\makecell[c]{{$\\sum\\text{{\\rmse{{}}}}$\\\\pour $\\q$\\\\(rad)}}\\\\ \n"
            f"   \\hline\n"
        )

        all_has_converged = True
        for study, sol, rmse_index in zip(self.conditions.studies, self.solution, self.conditions.rmse_index):
            rmse = np.sum(self._rmse(DataType.STATES, "q", rmse_index, sol))
            rmse_str = f"{rmse:0.3e}" if rmse != 0 else "---"
            if rmse_str.find("e") >= 0:
                rmse_str = rmse_str.replace("e", "$\\times 10^{{")
                rmse_str += "}}$"
                rmse_str = rmse_str.replace("+0", "")
                rmse_str = rmse_str.replace("-0", "-")
                rmse_str = rmse_str.replace("$\\times 10^{{0}}$", "")

            nlp = study.ocp.nlp[0]
            n_var = nlp.ns * nlp.controls.shape + (nlp.ns + 1) * nlp.states.shape
            n_constraints = nlp.ns * study.ocp.nlp[0].states.shape + sum([g.bounds.shape[0] for g in nlp.g])

            study_name = study.name
            if sol.iterations == study.solver.max_iter:
                study_name += "*"
                all_has_converged = False

            table += (
                f"   {study_name} "
                f"& {n_var}/{n_constraints} "
                f"& {sol.iterations} "
                f"& {sol.real_time_to_optimize:0.3f} "
                f"& {sol.real_time_to_optimize / sol.iterations:0.3f} "
                f"& {rmse_str} \\\\\n"
            )
        table += f"   \\hline\n" f"  \\end{{tabular}}\n"

        if not all_has_converged:
            table += f"  \\begin{{tablenotes}}\n"
            table += f"   \\item * Condition n'ayant pas convergé (maximum d'itérations atteint)\n"
            table += f"  \\end{{tablenotes}}\n"

        table += f" \\end{{threeparttable}}\n"
        table += f"\\end{{table}}\n\n"
        table += f"\\end{{document}}\n"

        save_path = f"{self.prepare_and_get_results_dir()}/results.tex"

        with open(save_path, "w", encoding="utf8") as file:
            file.write(table)
        print("\n\nTex file generated in the results folder")

    def save_solutions(self):
        for study, sol in zip(self.conditions.studies, self.solution):
            study.ocp.save(sol, file_path=f"{self.prepare_and_get_results_dir()}/{study.save_name}")
            study.ocp.save(sol, file_path=f"{self.prepare_and_get_results_dir()}/{study.save_name}", stand_alone=True)

    def prepare_and_get_results_dir(self):
        try:
            os.mkdir("results")
        except FileExistsError:
            pass

        try:
            os.mkdir(f"results/{self.name}")
        except FileExistsError:
            pass
        return f"results/{self.name}"

    def prepare_plot_data(self, data_type: DataType, key: str, font_size: int = 20):
        if not self._has_run:
            raise RuntimeError("run() must be called before plotting the results")

        n_plots = getattr(self.solution[0], data_type.value)[key].shape[0]
        if sum(np.array([getattr(sol, data_type.value)[key].shape[0] for sol in self.solution]) != n_plots) != 0:
            raise RuntimeError("All the models must have the same number of dof to be plotted")
        t = np.linspace(self.solution[0].phase_time[0], self.solution[0].phase_time[1], self.solution[0].ns[0] + 1)

        plot_options = self.conditions.plot_options
        studies = self.conditions.studies

        for i in range(n_plots):
            fig = plt.figure()
            fig.set_size_inches(16, 9)
            plt.rcParams["text.usetex"] = True
            plt.rcParams["text.latex.preamble"] = (
                r"\usepackage{amssymb}"
                r"\usepackage{siunitx}"
                r"\newcommand{\condition}{C/}"
                r"\newcommand{\noFatigue}{\varnothing}"
                r"\newcommand{\qcc}{4\textsubscript{CC}}"
                r"\newcommand{\pe}{P\textsubscript{E}}"
                r"\newcommand{\taupm}{\tau^{\pm}}"
                r"\newcommand{\tauns}{\tau^{\times}}"
                r"\newcommand{\condTauNf}{{\condition}{\tau}{\noFatigue}}"
                r"\newcommand{\condTaupm}{{\condition}{\taupm}{}}"
                r"\newcommand{\condTaupmQcc}{{\condition}{\taupm}{\qcc}}"
                r"\newcommand{\condTaupmPe}{{\condition}{\taupm}{\pe}}"
                r"\newcommand{\condTauns}{{\condition}{\tauns}{}}"
                r"\newcommand{\condTaunsNf}{{\condition}{\tauns}{\noFatigue}}"
                r"\newcommand{\condTaunsQcc}{{\condition}{\tauns}{\qcc}}"
                r"\newcommand{\condTaunsPe}{{\condition}{\tauns}{\pe}}"
                r"\newcommand{\condAlpha}{{\condition}{\alpha}{}}"
                r"\newcommand{\condAlphaNf}{{\condition}{\alpha}{\noFatigue}}"
                r"\newcommand{\condAlphaQcc}{{\condition}{\alpha}{\qcc}}"
                r"\newcommand{\condAlphaPe}{{\condition}{\alpha}{\pe}}"
            )

            ax = plt.axes()
            if plot_options.title:
                ax.set_title(plot_options.title % f"{key}\\textsubscript{{{i}}}", fontsize=1.5 * font_size)
            ax.set_xlabel(r"Temps (\SI{}{\second})", fontsize=font_size)
            ax.set_ylabel(
                r"Angle (\SI{}{\degree})" if plot_options.to_degrees else r"Angle (\SI{}{\radian})", fontsize=font_size
            )
            ax.tick_params(axis="both", labelsize=font_size)

            for sol, options in zip(self.solution, plot_options.options):
                data = getattr(sol, data_type.value)[key][i, :]
                data *= 180 / np.pi if plot_options.to_degrees else 1
                plt.plot(t, data, **options)

            if plot_options.legend_indices is not None:
                legend = [study.name if idx else "_" for study, idx in zip(studies, plot_options.legend_indices)]
                ax.legend(legend, loc="lower right", fontsize=font_size, framealpha=0.9)

            if plot_options.maximize:
                plt.get_current_fig_manager().window.showMaximized()

            if plot_options.save_path is not None and plot_options.save_path[i] is not None:
                plt.savefig(f"{self.prepare_and_get_results_dir()}/{plot_options.save_path[i]}", dpi=300)

        self._plots_are_prepared = True

    def _rmse(self, data_type, key, idx_ref: int, sol: Solution):
        data_ref = getattr(self.solution[idx_ref], data_type.value)[key]
        data = getattr(sol, data_type.value)[key]

        e = data_ref - data
        se = e ** 2
        mse = np.sum(se, axis=1) / data_ref.shape[1]
        rmse = np.sqrt(mse)
        return rmse

    def plot(self):
        if not self._plots_are_prepared:
            raise RuntimeError("At least one plot should be prepared before calling plot")

        plt.show()
