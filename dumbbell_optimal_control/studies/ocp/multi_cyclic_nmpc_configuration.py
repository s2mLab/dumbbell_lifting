from typing import Union

from bioptim import (
    MultiCyclicNonlinearModelPredictiveControl,
    Solution,
    OdeSolver,
    ObjectiveList,
    ConstraintList,
    Solver,
    MultiCyclicCycleSolutions,
)

from .enums import DynamicsFcn
from .fatigue_model import FatigueModel
from .ocp_configurations import OcpConfiguration


class MultiCyclicNmpcConfiguration(OcpConfiguration):
    def __init__(
        self,
        name: str,
        model_path: str,
        n_shoot_per_round_trip: int,
        n_simultaneous_round_trips: int,
        n_round_trips_to_advance,
        n_total_round_trips,
        round_trip_time: float,
        stop_if_fail: bool,
        x0: tuple[float, ...],
        tau_limits: tuple[float, float],
        dynamics: DynamicsFcn,
        fatigue_model: FatigueModel,
        objectives: ObjectiveList,
        constraints: ConstraintList,
        use_sx: bool,
        ode_solver: Union[OdeSolver.RK4, OdeSolver.COLLOCATION] = OdeSolver.RK4(),
        solver: Solver.Generic = Solver.ACADOS(),
        n_threads: int = 8,
    ):
        self.n_round_trips_to_advance = n_round_trips_to_advance
        self.n_total_round_trips_optimized = n_total_round_trips
        self.stop_if_fail = stop_if_fail
        super(MultiCyclicNmpcConfiguration, self).__init__(
            name=name,
            model_path=model_path,
            n_shoot_per_round_trip=n_shoot_per_round_trip,
            n_round_trips=n_simultaneous_round_trips,
            round_trip_time=round_trip_time,
            x0=x0,
            tau_limits=tau_limits,
            dynamics=dynamics,
            fatigue_model=fatigue_model,
            objectives=objectives,
            constraints=constraints,
            use_sx=use_sx,
            ode_solver=ode_solver,
            solver=solver,
            n_threads=n_threads,
        )

    def _set_generic_ocp(self):
        self.ocp = MultiCyclicNonlinearModelPredictiveControl(
            bio_model=self.model,
            dynamics=self.dynamics,
            n_cycles_simultaneous=self.n_round_trips,
            n_cycles_to_advance=self.n_round_trips_to_advance,
            cycle_len=self.n_shoot_per_round_trip,
            cycle_duration=self.round_trip_time,
            x_init=self.x_init,
            u_init=self.u_init,
            x_bounds=self.x_bounds,
            u_bounds=self.u_bounds,
            objective_functions=self.objectives,
            constraints=self.constraints,
            use_sx=isinstance(self.solver, Solver.ACADOS),
            n_threads=self.n_threads,
            # assume_phase_dynamics=True,
            assume_phase_dynamics=False,
        )
        from bioptim import CostType
        self.ocp.add_plot_penalty(CostType.ALL)

    def perform(self) -> tuple[Solution, list[Solution, ...]]:
        """

        Parameters
        ----------
        Returns
        -------
        The solution
        """

        cyclic_options = {"states": ["q", "qdot"]}

        return self.ocp.solve(
            update_function=lambda ocp, t, sol: self.nmpc_update_function(ocp, t, sol, self.n_total_round_trips_optimized),
            solver=self.solver,
            cyclic_options=cyclic_options,
            get_all_iterations=True,
            cycle_solutions=MultiCyclicCycleSolutions.ALL_CYCLES,
            max_consecutive_failing=1 if self.stop_if_fail else 0,
        )

    @staticmethod
    def nmpc_update_function(ocp, t, sol, n_cycles_total_optimized):
        if t > (n_cycles_total_optimized - ocp.n_cycles):
            print("Finished optimizing!")
            return False

        print(f"\n\nOptimizing cycle #{t + 1}..")
        return True
