import datetime
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Optional
import math


def simtime_to_utc(sim_seconds: float, sim_start) -> datetime.datetime:
    return sim_start + datetime.timedelta(seconds=sim_seconds)

# ----- Simulation Config -----
@dataclass
class SimulationConfig:
    """Configuration for a simulation run."""
    capacities: Dict[str, int]                                 # resource capacities (e.g. {"erp":1, "warehouse":2})
    resource_schedules: Dict[str, Dict[str, Any]]                       # resource schedules (daily/weekly/monthly/yearly)
    phases: Dict[str, Tuple[Tuple[int, int], Dict[str, float]]]  # phases of concept shift with (duration range in seconds, event probabilities)
    arrival_distribution: Tuple[str, Dict[str, float]]             # type of arrival distribution and its parameters (e.g. ("exponential", {"mean":300}))
    arrival_modulation: Optional[Dict[str, Any]]
    arrival_schedules: Dict[str, Dict[str, Any]]                   # arrival schedules (daily/weekly/monthly/yearly)
    event_worktime: Tuple[str, Dict[str, float]]               # type of event work time distribution and its parameters (e.g. ("exponential", {"mean":300}))
    max_n_cases: int = None                                    # maximum number of cases/orders to simulate
    max_sim_seconds: int = None                               # maximum simulation time in seconds
    sim_start: datetime.datetime = datetime.datetime(2025, 1, 1, 9, 0, 0, tzinfo=datetime.timezone.utc)  # first case arrival time
    random_seed: int = 42                                      # random seed for reproducibility

    def __post_init__(self):
        # sanity checks
        if self.arrival_distribution[0] not in ["fixed", "normal", "exponential", "uniform"]:
            raise ValueError("Arrival interval distribution must be 'fixed', 'normal', 'exponential' or 'uniform'")
        if self.event_worktime[0] not in ["fixed", "normal", "exponential", "uniform"]:
            raise ValueError("Event work time distribution must be 'fixed', 'normal', 'exponential' or 'uniform'")
        if self.max_n_cases is None and self.max_sim_seconds is None:
            raise ValueError("Either max_n_cases or max_sim_seconds must be specified.")
        if self.max_sim_seconds is not None and self.max_sim_seconds <= 0:
            raise ValueError("Maximum simulation time must be positive")
        if self.max_n_cases is not None and self.max_n_cases <= 0:
            raise ValueError("Number of cases must be positive")



def get_periodic_multiplier(dt: datetime.datetime, modulation: dict) -> float:
    """Return a periodic multiplier (1 ± amplitude) depending on modulation settings."""
    if not modulation or "type" not in modulation:
        return 1.0
    
    t_type = modulation["type"]
    amp = modulation.get("amplitude")
    const = modulation.get("constant", 1.0)
    phase = modulation.get("phase")

    if t_type == "daily":
        phi = ((dt.hour * 3600 + dt.minute * 60 + dt.second) / 86400.0)
    elif t_type == "weekly":
        phi = (dt.weekday() + dt.hour / 24.0) / 7.0
    elif t_type == "monthly":
        days_in_month = 30.0  # approximate or use calendar.monthrange(dt.year, dt.month)[1]
        phi = ((dt.day - 1) + dt.hour / 24.0) / days_in_month
    elif t_type == "yearly":
        phi = ((dt.timetuple().tm_yday - 1) + dt.hour / 24.0) / 365.0
    else:
        return 1.0

    return max(0, const + amp * math.cos(2 * math.pi * (phi - phase)))


def get_random(distribution: Tuple[str, Dict[str, float]]) -> float:
    """Get a random value from a given distribution."""
    import random
    dist_type, params = distribution
    if dist_type == "fixed":
        return params["value"]
    elif dist_type == "normal":
        return max(0, random.normalvariate(params["mean"], params["std"]))
    elif dist_type == "exponential":
        return random.expovariate(1 / params["mean"])
    elif dist_type == "uniform":
        return random.uniform(params["min"], params["max"])
    else:
        raise ValueError(f"Unknown distribution type: {dist_type}")


# ----- Availability checks -----
def is_open(resource: str, dt: datetime.datetime, schedules: dict) -> bool:
    schedule = schedules[resource]

    if schedule["daily"]:
        return any(start <= dt.time() < end for start, end in schedule["daily"])
    
    if schedule["weekly"] and dt.weekday() not in schedule["weekly"]:
        return False
    
    if schedule["monthly"] and dt.day not in schedule["monthly"]:
        return False

    if schedule["yearly"] and (dt.month, dt.day) not in schedule["yearly"]:
        return False

    return True

def next_open_time(resource: str, dt: datetime.datetime, schedules: dict) -> datetime.datetime:
    step = datetime.timedelta(minutes=1)
    while not is_open(resource, dt, schedules):
        dt += step
    return dt

def do_work(env, duration: int, resource_name: str, schedules: dict, sim_start: datetime.datetime):
    """Perform work of given duration (seconds) respecting UTC schedule."""
    remaining = datetime.timedelta(seconds=duration)
    while remaining > datetime.timedelta(0):
        dt = simtime_to_utc(env.now, sim_start)

        if not is_open(resource_name, dt, schedules):
            next_dt = next_open_time(resource_name, dt, schedules)
            yield env.timeout((next_dt - dt).total_seconds())
        else:
            rules = schedules[resource_name]
            today_end = None
            for start, end in rules["daily"]:
                if start <= dt.time() < end:
                    today_end = dt.replace(hour=end.hour, minute=end.minute,
                                           second=0, microsecond=0)
                    break
            available_now = today_end - dt if today_end else remaining
            work_now = min(remaining, available_now)
            yield env.timeout(work_now.total_seconds())
            remaining -= work_now

# ----- Phase handling -----
def get_phase_probability(phases, t, param="p1"):
    for _, ((start, end), params) in phases.items():
        if start <= t: # <= end:
            return params[param]
    raise ValueError(f"No phase found for time {t}")

# ----- Event log -----
def log_event(event_log, case_id, activity, t, sim_start):
    event_log.append({
        "case:concept:name": case_id,
        "concept:name": activity,
        "time:timestamp": simtime_to_utc(t, sim_start).replace(microsecond=0)
    })
