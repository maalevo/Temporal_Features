import simpy
import random
import pandas as pd
from simulation_utils import (
    SimulationConfig, do_work, get_phase_probability, get_random, log_event,
    is_open, next_open_time, simtime_to_utc, get_periodic_multiplier
)
import datetime


class O2C_Process:
    def __init__(self, config: SimulationConfig):
        """
        Initialize an Order-to-Cash process simulation and register case arrivals.
        :param config: SimulationConfig object with capacities, schedules, etc.
        """
        self.config = config
        self.env = simpy.Environment()
        
        # Create resources from config capacities
        self.erp = simpy.Resource(self.env, capacity=config.capacities["erp"])
        self.warehouse = simpy.Resource(self.env, capacity=config.capacities["warehouse"])
        self.sales = simpy.Resource(self.env, capacity=config.capacities["sales"])
        
        self.event_log = [] 
        random.seed(self.config.random_seed)  # global seed set here
        
        # schedule case generator process
        self.env.process(self.generate_cases())

    # ----- Atomic activities -----
    def purchase_order_received(self, case_id):
        log_event(self.event_log, case_id, "Purchase Order Received", self.env.now, self.config.sim_start)
        yield self.env.timeout(0)

    def check_stock(self, case_id):
        with self.erp.request() as req:
            yield req
            yield from do_work(self.env, 1, "erp", self.config.resource_schedules, self.config.sim_start)
            log_event(self.event_log, case_id, "Check Stock Availability", self.env.now, self.config.sim_start)
        # choose probability based on current phase
        p_in_stock = get_phase_probability(self.config.phases, self.env.now, "check_stock")
        return random.random() < p_in_stock

    def check_raw_materials(self, case_id):
        with self.erp.request() as req:
            yield req
            yield from do_work(self.env, 1, "erp", self.config.resource_schedules, self.config.sim_start)
            log_event(self.event_log, case_id, "Check Raw Materials Availability", self.env.now, self.config.sim_start)

    def retrieve_from_warehouse(self, case_id):
        with self.warehouse.request() as req:
            yield req
            yield from do_work(self.env, get_random(self.config.event_worktime), "warehouse", self.config.resource_schedules, self.config.sim_start)
            log_event(self.event_log, case_id, "Retrieve Product from Warehouse", self.env.now, self.config.sim_start)

    def request_raw_materials(self, case_id):
        with self.warehouse.request() as req:
            yield req
            yield from do_work(self.env, get_random(self.config.event_worktime), "warehouse", self.config.resource_schedules, self.config.sim_start)
            log_event(self.event_log, case_id, "Request Raw Materials", self.env.now, self.config.sim_start)

    def obtain_raw_materials(self, case_id):
        with self.warehouse.request() as req:
            yield req
            yield from do_work(self.env, get_random(self.config.event_worktime), "warehouse", self.config.resource_schedules, self.config.sim_start)
            log_event(self.event_log, case_id, "Obtain Raw Materials", self.env.now, self.config.sim_start)

    def manufacture_product(self, case_id):
        with self.warehouse.request() as req:
            yield req
            yield from do_work(self.env, get_random(self.config.event_worktime), "warehouse", self.config.resource_schedules, self.config.sim_start)
            log_event(self.event_log, case_id, "Manufacture Product", self.env.now, self.config.sim_start)

    def confirm_order(self, case_id):
        with self.sales.request() as req:
            yield req
            yield from do_work(self.env, get_random(self.config.event_worktime), "sales", self.config.resource_schedules, self.config.sim_start)
            log_event(self.event_log, case_id, "Confirm Order", self.env.now, self.config.sim_start)

    def emit_invoice(self, case_id):
        with self.sales.request() as req:
            yield req
            yield from do_work(self.env, get_random(self.config.event_worktime), "sales", self.config.resource_schedules, self.config.sim_start)
            log_event(self.event_log, case_id, "Emit Invoice", self.env.now, self.config.sim_start)

    def receive_payment(self, case_id):
        with self.sales.request() as req:
            yield req
            yield from do_work(self.env, get_random(self.config.event_worktime), "sales", self.config.resource_schedules, self.config.sim_start)
            log_event(self.event_log, case_id, "Receive Payment", self.env.now, self.config.sim_start)

    def get_shipping_address(self, case_id):
        with self.warehouse.request() as req:
            yield req
            yield from do_work(self.env, get_random(self.config.event_worktime), "warehouse", self.config.resource_schedules, self.config.sim_start)
            log_event(self.event_log, case_id, "Get Shipping Address", self.env.now, self.config.sim_start)

    def ship_product(self, case_id):
        with self.warehouse.request() as req:
            yield req
            yield from do_work(self.env, get_random(self.config.event_worktime), "warehouse", self.config.resource_schedules, self.config.sim_start)
            log_event(self.event_log, case_id, "Ship Product", self.env.now, self.config.sim_start)

    def archive_order(self, case_id):
        with self.sales.request() as req:
            yield req
            yield from do_work(self.env, get_random(self.config.event_worktime), "sales", self.config.resource_schedules, self.config.sim_start)
            log_event(self.event_log, case_id, "Archive Order", self.env.now, self.config.sim_start)

    # ----- Branches -----
    def branch_shipping(self, case_id):
        yield self.env.process(self.get_shipping_address(case_id))
        yield self.env.process(self.ship_product(case_id))

    def branch_billing(self, case_id):
        yield self.env.process(self.emit_invoice(case_id))
        yield self.env.process(self.receive_payment(case_id))

    # ----- Workflow -----
    def order_process(self, case_id):
        yield self.env.process(self.purchase_order_received(case_id))
        in_stock = yield self.env.process(self.check_stock(case_id))

        if in_stock:
            yield self.env.process(self.retrieve_from_warehouse(case_id))
        else:
            yield self.env.process(self.check_raw_materials(case_id))
            yield self.env.process(self.request_raw_materials(case_id))
            yield self.env.process(self.obtain_raw_materials(case_id))
            yield self.env.process(self.manufacture_product(case_id))

        yield self.env.process(self.confirm_order(case_id))

        ship_seq = self.env.process(self.branch_shipping(case_id))
        bill_seq = self.env.process(self.branch_billing(case_id))
        yield ship_seq & bill_seq

        yield self.env.process(self.archive_order(case_id))
        yield self.env.timeout(1)
        log_event(self.event_log, case_id, "Order Fulfilled", self.env.now, self.config.sim_start)

    # ----- Case arrivals -----
    def generate_cases(self):
        """Generate cases according to arrival schedule and distribution."""
        i = 0
        while True:
            if self.config.max_sim_seconds is not None and self.env.now >= self.config.max_sim_seconds:
                break
            if self.config.max_n_cases is not None and i >= self.config.max_n_cases:
                break

            # Check if arrivals are allowed at current simulated time
            dt = simtime_to_utc(self.env.now, self.config.sim_start)
            if not is_open("arrivals", dt, self.config.arrival_schedules):
                next_dt = next_open_time("arrivals", dt, self.config.arrival_schedules)
                yield self.env.timeout((next_dt - dt).total_seconds())

            # Generate a new case
            yield from self.generate_case(i)
            i += 1

    def generate_case(self, i):
        """Generate a single case arrival."""
        width = len(str(self.config.max_n_cases or 9999))
        case_id = f"Case_{(i+1):0{width}d}"
        self.env.process(self.order_process(case_id))  

        dt = simtime_to_utc(self.env.now, self.config.sim_start)
        mult = get_periodic_multiplier(dt, self.config.arrival_modulation)

        # Smaller multiplier => higher arrival intensity
        interval = get_random(self.config.arrival_distribution) * mult
        yield self.env.timeout(interval)

    # ----- Run -----
    def run(self):
        self.env.run()
        return pd.DataFrame(self.event_log).sort_values(
            ["case:concept:name", "time:timestamp"]
        )
