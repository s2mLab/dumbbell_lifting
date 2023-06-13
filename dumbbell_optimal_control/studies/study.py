from typing import Protocol
import os

from bioptim import Solution
import numpy as np
from matplotlib import pyplot as plt
import pickle

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

    def __init__(self, conditions: Conditions = None, solution=None, result_folder="results"):
        self.name = conditions.name if conditions is not None else "debug"
        self._has_run: bool = False
        self._plots_are_prepared: bool = False
        self.conditions: StudyConfiguration = conditions.value if conditions is not None else StudyConfiguration()
        self.solution: list[Solution, ...] | tuple[
            Solution, list[Solution, ...], list[Solution, ...]] = [] if solution is None else solution
        self.result_folder = result_folder

    def run(self):
        for condition in self.conditions.studies:
            self.solution.append(condition.perform())
        self._has_run = True
        self._remove_failed_ocps()

    def save(self):
        self._cycle_cost_computation()
        self._time_to_optimize()
        self._save_decision_variables()

    def _cycle_cost_computation(self):
        """ compute the cost of each cycle for each condition and each cost function """
        self.solution_costs = []

        for i, condition in enumerate(self.conditions.studies):
            cycle_solutions = self.solution[i][2] if isinstance(self.solution[i], tuple) else self.solution

            for j, sol in enumerate(cycle_solutions):
                sol.print_cost()
            cycle_costs = cycle_solutions[0].detailed_cost

            has_torque_cost = False
            has_tau_minus_mf_cost = False
            has_tau_plus_mf_cost = False
            has_shoulder_state_cost = False
            for idx, cost in enumerate(cycle_costs):
                if cost["name"] == "Lagrange.MINIMIZE_CONTROL" and cost["params"]["key"] == "tau" and cost[
                    "derivative"] == False:
                    has_torque_cost = True
                    idx_torque_cost = idx
                if cost["name"] == "Lagrange.MINIMIZE_STATE" and cost["params"]["key"] == "tau_minus_mf":
                    has_tau_minus_mf_cost = True
                    idx_tau_minus_mf_cost = idx
                if cost["name"] == "Lagrange.MINIMIZE_STATE" and cost["params"]["key"] == "tau_plus_mf":
                    has_tau_plus_mf_cost = True
                    idx_tau_plus_mf_cost = idx
                if cost["name"] == "Lagrange.MINIMIZE_STATE" and cost["params"]["key"] == "q":
                    has_shoulder_state_cost = True
                    idx_shoulder_state_cost = idx

            self.solution_costs.append(dict(
                condition_name=self.name + '_' + condition.name,
                shoulder_state_cost=[s.detailed_cost[idx_shoulder_state_cost]["cost_value_weighted"] for s in
                                     cycle_solutions] if has_shoulder_state_cost else None,
                torque_cost=[s.detailed_cost[idx_torque_cost]["cost_value_weighted"] for s in
                             cycle_solutions] if has_torque_cost else None,
                tau_minus_mf_cost=[s.detailed_cost[idx_tau_minus_mf_cost]["cost_value_weighted"] for s in
                                   cycle_solutions] if has_tau_minus_mf_cost else None,

                tau_plus_mf_cost=[s.detailed_cost[idx_tau_plus_mf_cost]["cost_value_weighted"] for s in
                                  cycle_solutions] if has_tau_plus_mf_cost else None,
            )
            )

            if not os.path.exists(f"{self.result_folder}"):
                os.mkdir(f"{self.result_folder}")
            if not os.path.exists(f"{self.result_folder}/costs/"):
                os.mkdir(f"{self.result_folder}/costs/")
            import pandas as pd
            pd.DataFrame(self.solution_costs[i]).to_csv(f"{self.result_folder}/costs/{self.name}_{condition.name}.csv")

    def _time_to_optimize(self):
        """ export the time to optimize of each cycle for each condition """
        self.time_to_optimize = []

        for i, condition in enumerate(self.conditions.studies):

            ocp_solutions = self.solution[i][1]

            self.time_to_optimize.append(dict(
                condition_name=condition.name,
                time=[s.real_time_to_optimize for s in
                      ocp_solutions],
            )
            )
            # save the cost in a numpy array
            if not os.path.exists(f"{self.result_folder}"):
                os.mkdir(f"{self.result_folder}")
            if not os.path.exists(f"{self.result_folder}/time/"):
                os.mkdir(f"{self.result_folder}/time/")
            # Save dictionary to a CSV file
            import pandas as pd
            pd.DataFrame(self.time_to_optimize[i]).to_csv(f"{self.result_folder}/time/{self.name}_{condition.name}.csv")

    def _save_decision_variables(self):
        """ export the decision variables of each cycle for each condition """
        self.full_ocp = []
        self.windows = []
        self.cycles = []

        for i, condition in enumerate(self.conditions.studies):

            self.full_ocp.append(dict(
                condition_name=self.name + '_' + condition.name,
                states=self.solution[i][0].states,
                controls=self.solution[i][0].controls,
                parameters=self.solution[i][0].parameters,
                time=self.solution[i][0].time,
            ))
            self.windows.append(dict(
                condition_name=self.name + '_' + condition.name,
                states=[s.states for s in self.solution[i][1]],
                controls=[s.controls for s in self.solution[i][1]],
                parameters=[s.parameters for s in self.solution[i][1]],
                time=[s.time for s in self.solution[i][1]],
            ),
            )
            self.cycles.append(dict(
                condition_name=self.name + '_' + condition.name,
                states=[s.states for s in self.solution[i][2]],
                controls=[s.controls for s in self.solution[i][2]],
                parameters=[s.parameters for s in self.solution[i][2]],
                time=[s.time for s in self.solution[i][2]],
            )
            )
            if not os.path.exists(f"{self.result_folder}"):
                os.mkdir(f"{self.result_folder}")
            if not os.path.exists(f"{self.result_folder}/decision_variables/"):
                os.mkdir(f"{self.result_folder}/decision_variables/")
            # Save dictionary with pickle
            data_to_save = (self.full_ocp[i], self.windows[i], self.cycles[i])
            with open(f"{self.result_folder}/decision_variables/{self.name}_{condition.name}.pkl", "wb") as file:
                pickle.dump(data_to_save, file)

    def _remove_failed_ocps(self):
        """ remove all failed ocp from the solution list """
        for i, sol in enumerate(self.solution):
            if isinstance(sol, list):
                for j, sol_window in enumerate(sol[1]):
                    if sol_window.status == 1:
                        self.solution[i][1].pop(j)
                for i, sol_cycle in enumerate(sol[2]):
                    if sol_cycle.status == 1:
                        self.solution[i][2].pop(i)

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

    def save_solutions(self):
        for study, sol in zip(self.conditions.studies, self.solution):
            study.ocp.save(sol, file_path=f"{self.prepare_and_get_results_dir()}/{study.save_name}")
            study.ocp.save(sol, file_path=f"{self.prepare_and_get_results_dir()}/{study.save_name}", stand_alone=True)

    def prepare_and_get_results_dir(self):
        try:
            os.mkdir(f"{self.result_folder}")
        except FileExistsError:
            pass

        try:
            os.mkdir(f"{self.result_folder}/{self.name}")
        except FileExistsError:
            pass
        return f"{self.result_folder}/{self.name}"

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
