from bioptim import OdeSolver, Solver
import numpy as np


class StudySetup:
    def __init__(
        self,
        model_path: str = "models/arm26.bioMod",
        n_shoot_per_round_trip: int = 50,
        round_trip_time: float = 1,
        n_round_trips: int = 5,
        n_round_trips_to_advance: int = 1,
        n_total_round_trips: int = 10,
        x0: tuple[float, ...] = (0.07, 15 * np.pi / 180, 0, 0),
        tau_limits_no_muscles: tuple[float, float] = (-50, 50),
        tau_limits_with_muscles: tuple[float, float] = (-1, 1),
        weight_fatigue: float = 1_000,
        split_controls: bool = False,
        ode_solver: OdeSolver = None,
        solver: Solver = None,
        use_sx: bool = False,
        n_thread: int = 8,
    ):
        self.model_path = model_path
        self.n_shoot_per_round_trip = n_shoot_per_round_trip
        self.n_round_trips = n_round_trips
        self.n_round_trips_to_advance = n_round_trips_to_advance
        self.n_total_round_trips = n_total_round_trips
        self.round_trip_time = round_trip_time
        self.x0 = x0
        self.tau_limits_no_muscles = tau_limits_no_muscles
        self.tau_limits_with_muscles = tau_limits_with_muscles
        self.weight_fatigue = weight_fatigue
        self.split_controls = split_controls
        self.ode_solver = OdeSolver.RK4() if ode_solver is None else ode_solver
        self.solver = solver
        if self.solver is None:
            self.solver = Solver.IPOPT(
                show_online_optim=False,
                _print_level=5,
                # _linear_solver="ma57",
                _hessian_approximation="exact",
                _max_iter=1000,
            )
        self.use_sx = use_sx
        self.n_thread = n_thread
