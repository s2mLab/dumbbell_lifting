from .ocp import OcpConfiguration, MultiCyclicNmpcConfiguration
from .programs import Program
from .study_setup import StudySetup


def get_ocp(params: Program, study_setup: StudySetup):
    name, dynamics, fatigue_model, objectives, constraints = params.value[0](study_setup)
    return OcpConfiguration(
        name=name,
        model_path=study_setup.model_path,
        n_shoot_per_round_trip=study_setup.n_shoot_per_round_trip,
        n_round_trips=study_setup.n_round_trips,
        round_trip_time=study_setup.round_trip_time,
        x0=study_setup.x0,
        tau_limits=study_setup.tau_limits_no_muscles,
        dynamics=dynamics,
        fatigue_model=fatigue_model,
        objectives=objectives,
        constraints=constraints,
        use_sx=study_setup.use_sx,
        ode_solver=study_setup.ode_solver,
        solver=study_setup.solver,
        n_threads=study_setup.n_thread,
    )


def get_nmpc(params: Program, study_setup: StudySetup):
    name, dynamics, fatigue_model, objectives, constraints = params.value[0](study_setup)
    return MultiCyclicNmpcConfiguration(
        name=name,
        model_path=study_setup.model_path,
        n_shoot_per_round_trip=study_setup.n_shoot_per_round_trip,
        n_simultaneous_round_trips=study_setup.n_round_trips,
        n_round_trips_to_advance=study_setup.n_round_trips_to_advance,
        n_total_round_trips=study_setup.n_total_round_trips,
        round_trip_time=study_setup.round_trip_time,
        x0=study_setup.x0,
        tau_limits=study_setup.tau_limits_no_muscles,
        dynamics=dynamics,
        fatigue_model=fatigue_model,
        objectives=objectives,
        constraints=constraints,
        use_sx=study_setup.use_sx,
        ode_solver=study_setup.ode_solver,
        solver=study_setup.solver,
        n_threads=study_setup.n_thread,
    )
