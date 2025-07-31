import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import os

# Define paths for data files - ADJUST THESE PATHS AS NECESSARY
SOLAR_DATA_PATH = '/content/solar_data_full.csv'  # Make sure this path is correct
WIND_DATA_PATH = '/content/wind_data_full.csv'    # Make sure this path is correct
GRID_DATA_PATH = '/content/grid_data_full.csv'    # Make sure this path is correct

class HybridPowerPlantEnv(gym.Env):
    """
    A Reinforcement Learning environment representing a Hybrid Power Plant
    with Solar, Wind, and Battery storage, interacting with the grid.
    """
    def __init__(self, time_step_duration_hours=1, battery_capacity_kwh=10000,
                 battery_max_charge_mw=2000, battery_max_discharge_mw=2000,
                 battery_efficiency=0.95, grid_buy_price_per_kwh=0.15,
                 grid_sell_price_per_kwh=0.10, penalty_unmet_demand=1.0):
        super().__init__()

        # Load data within the __init__ method
        try:
            self.solar_data_full = pd.read_csv(SOLAR_DATA_PATH)
            self.wind_data_full = pd.read_csv(WIND_DATA_PATH)
            self.grid_data_full = pd.read_csv(GRID_DATA_PATH)

            # Ensure time index exists and is sorted
            for df in [self.solar_data_full, self.wind_data_full, self.grid_data_full]:
                 if 'time' in df.columns:
                     df['time'] = pd.to_datetime(df['time'])
                     df.set_index('time', inplace=True)
                     df.sort_index(inplace=True)
                 else:
                     print(f"Warning: 'time' column not found in a data file.")


        except FileNotFoundError as e:
            print(f"Error loading data file: {e}")
            # Handle the missing file, e.g., raise an error or use dummy data
            self.solar_data_full = pd.DataFrame()
            self.wind_data_full = pd.DataFrame()
            self.grid_data_full = pd.DataFrame()


        # Combine renewable power, filling missing values with 0 after loading
        if not self.solar_data_full.empty and not self.wind_data_full.empty:
            # Align indices before adding
            common_index = self.solar_data_full.index.union(self.wind_data_full.index)
            solar_aligned = self.solar_data_full.reindex(common_index).fillna(0)
            wind_aligned = self.wind_data_full.reindex(common_index).fillna(0)

            if 'solar_power_MW' in solar_aligned.columns and 'wind_power_MW' in wind_aligned.columns:
                 self.combined_renewable_power_mw = solar_aligned['solar_power_MW'].add(wind_aligned['wind_power_MW'], fill_value=0)
            else:
                 print("Warning: 'solar_power_MW' or 'wind_power_MW' columns not found after aligning data.")
                 self.combined_renewable_power_mw = pd.Series(0, index=common_index)

        elif not self.solar_data_full.empty and 'solar_power_MW' in self.solar_data_full.columns:
             self.combined_renewable_power_mw = self.solar_data_full['solar_power_MW'].copy()
        elif not self.wind_data_full.empty and 'wind_power_MW' in self.wind_data_full.columns:
             self.combined_renewable_power_mw = self.wind_data_full['wind_power_MW'].copy()
        else:
            print("Warning: No valid solar or wind data loaded.")
            self.combined_renewable_power_mw = pd.Series() # Empty series if no data

        # Ensure grid demand data is available
        if not self.grid_data_full.empty and 'grid_demand_MW' in self.grid_data_full.columns:
             self.grid_demand_full_mw = self.grid_data_full['grid_demand_MW'].copy()
        else:
             print("Warning: No valid grid demand data loaded.")
             self.grid_demand_full_mw = pd.Series() # Empty series

        # Align all data based on the union of all indices
        all_indices = self.combined_renewable_power_mw.index.union(self.grid_demand_full_mw.index)
        self.combined_renewable_power_mw = self.combined_renewable_power_mw.reindex(all_indices).fillna(0)
        self.grid_demand_full_mw = self.grid_demand_full_mw.reindex(all_indices).fillna(0)

        # Ensure data is not empty after loading and aligning
        if self.combined_renewable_power_mw.empty or self.grid_demand_full_mw.empty or len(self.combined_renewable_power_mw) != len(self.grid_demand_full_mw):
            raise ValueError("Data loading or alignment failed. Renewable or grid demand data is empty or has mismatched lengths.")


        self.data_len = len(self.combined_renewable_power_mw)
        self.time_step_duration_hours = time_step_duration_hours
        self.battery_capacity_kwh = battery_capacity_kwh
        self.battery_max_charge_mw = battery_max_charge_mw
        self.battery_max_discharge_mw = battery_max_discharge_mw
        self.battery_efficiency = battery_efficiency
        self.grid_buy_price_per_kwh = grid_buy_price_per_kwh
        self.grid_sell_price_per_kwh = grid_sell_price_per_kwh
        self.penalty_unmet_demand = penalty_unmet_demand

        # Define action space: [battery_charge_discharge_mw, power_to_grid_mw]
        # battery_charge_discharge_mw: positive for charge, negative for discharge
        # power_to_grid_mw: positive for selling to grid, negative for buying from grid
        # Assuming reasonable bounds for power flow. Adjust as needed.
        low_action = np.array([-self.battery_max_discharge_mw, -np.inf], dtype=np.float32) # Can discharge up to max, can buy potentially infinite power (in reality limited by grid connection)
        high_action = np.array([self.battery_max_charge_mw, np.inf], dtype=np.float32)   # Can charge up to max, can sell potentially infinite power (in reality limited by grid connection)

        # Let's refine the action space based on common practices
        # Action 0: Battery action (charge/discharge). Range [-battery_max_discharge_mw, battery_max_charge_mw]
        # Action 1: Grid interaction. Can be power bought (negative) or sold (positive).
        # A simpler approach might be two separate actions: one for battery (charge/discharge)
        # and one for grid (buy/sell). Or a single action vector.

        # Let's define a continuous action space: [battery_power_mw, grid_power_mw]
        # battery_power_mw: positive for charging, negative for discharging, bounded by max_charge/discharge
        # grid_power_mw: positive for selling to grid, negative for buying from grid.
        # We'll let the agent decide how much to buy/sell, but apply penalties for unmet demand.
        self.action_space = spaces.Box(low=np.array([-self.battery_max_discharge_mw, -np.inf], dtype=np.float32),
                                       high=np.array([self.battery_max_charge_mw, np.inf], dtype=np.float32),
                                       dtype=np.float32)


        # Define observation space: [current_time_step, renewable_output_mw, grid_demand_mw, battery_soc_kwh]
        # current_time_step: Integer representing the current index in the data
        # renewable_output_mw: Renewable power available at the current time step
        # grid_demand_mw: Grid demand at the current time step
        # battery_soc_kwh: Current state of charge of the battery
        low_obs = np.array([0, -np.inf, -np.inf, 0], dtype=np.float32)
        high_obs = np.array([self.data_len - 1, np.inf, np.inf, self.battery_capacity_kwh], dtype=np.float32)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)

        # Initialize state variables
        self.current_time_step = 0
        self.battery_soc_kwh = self.battery_capacity_kwh / 2 # Start with battery half full
        self.time_step_duration_hours = time_step_duration_hours


    def step(self, action):
        # Ensure action is within bounds
        action = np.clip(action, self.action_space.low, self.action_space.high)

        battery_action_mw = action[0]  # Positive for charging, negative for discharging
        grid_action_mw = action[1]     # Positive for selling, negative for buying

        # Get current renewable output and grid demand
        renewable_output_mw = self.combined_renewable_power_mw.iloc[self.current_time_step]
        grid_demand_mw = self.grid_demand_full_mw.iloc[self.current_time_step]

        # --- Power Flow Logic ---

        # 1. Handle battery charge/discharge attempt
        battery_charge_attempt_mw = max(0, battery_action_mw)
        battery_discharge_attempt_mw = abs(min(0, battery_action_mw))

        # Calculate maximum possible charge based on available renewable and battery capacity
        # Available power for charging = renewable_output_mw + power_bought_for_charging (if applicable)
        # Max charge power limited by battery_max_charge_mw
        # Max charge energy limited by remaining battery capacity

        # For simplicity in this step function, let's assume battery action is the agent's
        # desired charge/discharge, and we'll handle constraints.

        actual_battery_charge_mw = 0
        actual_battery_discharge_mw = 0
        battery_delta_kwh = 0

        # Attempt to charge the battery
        if battery_charge_attempt_mw > 0:
            # Power available from renewables for charging (can be used directly or via grid)
            # Let's assume charging can come from renewables or grid based on grid_action_mw
            # A more complex model would prioritize renewable charging.
            # For this action space, battery_action_mw is the *net* power to/from battery.
            # Let's assume positive battery_action_mw means drawing power from the system (renewables/grid) to charge.
            # Negative battery_action_mw means supplying power from battery to the system.

            # Let's reinterpret action[0] as the desired change in battery SOC in MW for the time step.
            # Positive action[0] means increase SOC (charge), negative means decrease SOC (discharge).

            desired_battery_power_mw = action[0] # Positive to charge, negative to discharge

            if desired_battery_power_mw > 0: # Attempt to charge
                # Power needed for charging
                power_needed_for_charge_mw = desired_battery_power_mw
                # Max power can draw for charging
                max_charge_power_mw = min(power_needed_for_charge_mw, self.battery_max_charge_mw)
                # Max energy can store in battery
                max_charge_energy_kwh = (self.battery_capacity_kwh - self.battery_soc_kwh) / self.battery_efficiency # Account for efficiency
                max_charge_power_based_on_capacity_mw = max_charge_energy_kwh / self.time_step_duration_hours

                actual_battery_charge_mw = min(max_charge_power_mw, max_charge_power_based_on_capacity_mw)
                battery_delta_kwh = actual_battery_charge_mw * self.time_step_duration_hours * self.battery_efficiency

            elif desired_battery_power_mw < 0: # Attempt to discharge
                 # Power to supply from battery
                 power_to_supply_from_battery_mw = abs(desired_battery_power_mw)
                 # Max power can discharge
                 max_discharge_power_mw = min(power_to_supply_from_battery_mw, self.battery_max_discharge_mw)
                 # Max energy can supply from battery
                 max_discharge_energy_kwh = self.battery_soc_kwh * self.battery_efficiency # Account for efficiency
                 max_discharge_power_based_on_capacity_mw = max_discharge_energy_kwh / self.time_step_duration_hours

                 actual_battery_discharge_mw = min(max_discharge_power_mw, max_discharge_power_based_on_capacity_mw)
                 battery_delta_kwh = -actual_battery_discharge_mw * self.time_step_duration_hours / self.battery_efficiency # Energy removed from battery

            self.battery_soc_kwh += battery_delta_kwh
            self.battery_soc_kwh = np.clip(self.battery_soc_kwh, 0, self.battery_capacity_kwh) # Ensure SOC is within bounds


        # 2. Calculate net power available/needed from the system (excluding grid interaction for now)
        # Net power = Renewable Output + Battery Discharge - Battery Charge
        # Positive net_system_output_mw means surplus power available
        # Negative net_system_output_mw means deficit power needed
        net_system_output_mw = renewable_output_mw + actual_battery_discharge_mw - actual_battery_charge_mw

        # 3. Handle grid interaction and meet demand
        # grid_action_mw: positive means selling to grid, negative means buying from grid
        power_to_grid_attempt_mw = max(0, grid_action_mw)
        power_from_grid_attempt_mw = abs(min(0, grid_action_mw))

        power_sold_mw = 0
        power_bought_mw = 0
        unmet_demand_mw = 0

        # If there is a surplus from the system, prioritize meeting local demand first (if any, though demand is abstracted as grid_demand_mw)
        # and then sell excess to the grid based on agent's action
        if net_system_output_mw > 0:
            # Power available to meet demand or sell
            power_available_mw = net_system_output_mw

            # How much power does the agent want to sell to the grid?
            desired_sell_to_grid_mw = power_to_grid_attempt_mw

            # Actual power sold is limited by available surplus
            power_sold_mw = min(desired_sell_to_grid_mw, power_available_mw)

            # Any remaining surplus is not used (could be curtailed in a real system)
            # power_available_after_selling = power_available_mw - power_sold_mw

            # If there is a deficit from the system, the agent must buy from the grid to meet demand
        elif net_system_output_mw < 0:
            # Power deficit that needs to be met
            power_deficit_mw = abs(net_system_output_mw)

            # How much power does the agent want to buy from the grid?
            desired_buy_from_grid_mw = power_from_grid_attempt_mw

            # The grid demand is the total power needed. The agent's 'buying' action
            # should aim to cover the deficit.

            # Let's rethink the grid action based on meeting grid_demand_mw
            # The power needed *from the grid* is grid_demand_mw - net_system_output_mw
            # If net_system_output_mw is negative, this is grid_demand_mw + |net_system_output_mw|

            # Let's simplify: the total power required at the grid connection point is grid_demand_mw.
            # The power supplied by renewables and battery is net_system_output_mw.
            # The difference must be supplied by the grid (bought) or is unmet demand.
            # Or if net_system_output_mw > grid_demand_mw, the excess can be sold to the grid.

            # Let's interpret grid_action_mw as the NET power flow at the grid connection point
            # Positive grid_action_mw means power flows TO the grid (selling)
            # Negative grid_action_mw means power flows FROM the grid (buying)

            # The total power balance must be:
            # Renewable Output + Battery Discharge - Battery Charge + Power from Grid - Power to Grid = Grid Demand
            # net_system_output_mw + Power from Grid - Power to Grid = Grid Demand
            # Power from Grid - Power to Grid = Grid Demand - net_system_output_mw

            # So, the net grid flow (Power from Grid - Power to Grid) should ideally equal Grid Demand - net_system_output_mw
            # The agent's action[1] is the desired net grid flow.

            desired_net_grid_flow_mw = grid_action_mw # Positive to sell, negative to buy

            # The required net grid flow to perfectly meet demand is grid_demand_mw - net_system_output_mw
            required_net_grid_flow_mw = grid_demand_mw - net_system_output_mw

            # If the agent's desired net grid flow is less than what's required to meet demand, there's unmet demand.
            if desired_net_grid_flow_mw < required_net_grid_flow_mw:
                unmet_demand_mw = required_net_grid_flow_mw - desired_net_grid_flow_mw
                net_grid_flow_mw = desired_net_grid_flow_mw # Agent only managed this much net flow
            else:
                # If the agent desires more grid flow than needed (e.g., wants to sell more than surplus, or buy less than deficit)
                # The actual net grid flow is limited by the required amount to meet demand or the available surplus for selling.
                # Let's assume the agent's action is a target, and the environment applies constraints.

                # Option 1: Agent's action is a target net grid flow. Environment tries to achieve it.
                # If agent wants to sell (positive desired_net_grid_flow_mw): Can only sell up to available surplus (net_system_output_mw - grid_demand_mw if positive).
                # If agent wants to buy (negative desired_net_grid_flow_mw): Needs to buy at least the deficit (grid_demand_mw - net_system_output_mw if positive deficit).

                # Let's stick to the interpretation of action[1] as desired net grid flow (positive = sell, negative = buy).

                desired_net_grid_flow_mw = action[1]

                # If desired net flow is positive (selling):
                if desired_net_grid_flow_mw > 0:
                    # Power available to sell = max(0, net_system_output_mw - grid_demand_mw)
                    # Actual power sold is limited by desired amount and available power
                    power_sold_mw = min(desired_net_grid_flow_mw, max(0, net_system_output_mw - grid_demand_mw))
                    power_bought_mw = 0 # Cannot sell and buy simultaneously (net)
                    net_grid_flow_mw = power_sold_mw # Net flow is positive (to grid)
                    # Check if meeting demand resulted in surplus or deficit
                    remaining_demand_or_surplus = net_system_output_mw - grid_demand_mw - power_sold_mw
                    unmet_demand_mw = max(0, -remaining_demand_or_surplus) # Unmet if remaining is negative

                # If desired net flow is negative (buying):
                elif desired_net_grid_flow_mw < 0:
                    # Power needed from grid = abs(desired_net_grid_flow_mw)
                    # Power needed to meet demand after using system output = max(0, grid_demand_mw - net_system_output_mw)
                    power_to_buy_mw = abs(desired_net_grid_flow_mw)

                    # Actual power bought is limited by the amount needed to meet demand + any extra the agent wants to buy (e.g., for battery charging, though not explicitly modeled here)
                    # Let's assume buying is primarily for meeting grid demand.
                    power_bought_mw = power_to_buy_mw
                    power_sold_mw = 0 # Cannot sell and buy simultaneously (net)
                    net_grid_flow_mw = -power_bought_mw # Net flow is negative (from grid)

                    # Check if buying was sufficient to meet demand
                    power_after_grid_buy = net_system_output_mw + power_bought_mw
                    unmet_demand_mw = max(0, grid_demand_mw - power_after_grid_buy)


                # If desired net flow is zero:
                else: # desired_net_grid_flow_mw == 0
                    power_sold_mw = 0
                    power_bought_mw = 0
                    net_grid_flow_mw = 0
                    # Unmet demand is the deficit if system output isn't enough
                    unmet_demand_mw = max(0, grid_demand_mw - net_system_output_mw)


        # Convert power flows (MW) to energy (kWh) for the time step
        power_bought_kwh = power_bought_mw * self.time_step_duration_hours
        power_sold_kwh = power_sold_mw * self.time_step_duration_hours
        unmet_demand_kwh = unmet_demand_mw * self.time_step_duration_hours
        renewable_output_kwh = renewable_output_mw * self.time_step_duration_hours


        # --- Calculate Reward ---
        # Reward could be based on:
        # - Cost of buying power from the grid
        # - Revenue from selling power to the grid
        # - Penalty for unmet demand
        # - Cost/benefit of battery usage (optional, but can encourage efficient use)

        reward = 0

        # Cost of buying power
        reward -= power_bought_kwh * self.grid_buy_price_per_kwh

        # Revenue from selling power
        reward += power_sold_kwh * self.grid_sell_price_per_kwh

        # Penalty for unmet demand
        reward -= unmet_demand_kwh * self.penalty_unmet_demand

        # Optional: Add a small penalty for large battery charge/discharge to encourage smoother operation
        # reward -= abs(battery_action_mw) * 0.01 # Example penalty

        # --- Update State ---
        self.current_time_step += 1

        # --- Check if episode is done ---
        terminated = self.current_time_step >= self.data_len
        truncated = False # Or define other conditions for truncation if needed

        # --- Create Info Dictionary ---
        info = {
            "current_time_step": self.current_time_step -1, # Log the step that just finished
            "renewable_output_mw": renewable_output_mw,
            "grid_demand_mw": grid_demand_mw,
            "battery_soc_kwh": self.battery_soc_kwh,
            "battery_charge_attempt_mw": battery_charge_attempt_mw,
            "battery_discharge_attempt_mw": battery_discharge_attempt_mw,
            "actual_battery_charge_mw": actual_battery_charge_mw,
            "actual_battery_discharge_mw": actual_battery_discharge_mw,
            "power_bought_mw": power_bought_mw,
            "power_sold_mw": power_sold_mw,
            "unmet_demand_mw": unmet_demand_mw,
            "reward": reward # Include the immediate reward for this step
        }

        # --- Return step results ---
        # The observation for the *next* state
        next_obs = self._get_obs()


        return next_obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset state variables
        self.current_time_step = 0
        self.battery_soc_kwh = self.battery_capacity_kwh / 2 # Reset battery to half full

        # Get initial observation
        obs = self._get_obs()

        # Return initial observation and info
        info = {
             "current_time_step": self.current_time_step,
             "renewable_output_mw": self.combined_renewable_power_mw.iloc[self.current_time_step] if not self.combined_renewable_power_mw.empty else 0,
             "grid_demand_mw": self.grid_demand_full_mw.iloc[self.current_time_step] if not self.grid_demand_full_mw.empty else 0,
             "battery_soc_kwh": self.battery_soc_kwh,
             # Include other initial info if needed, e.g., prices
         }

        return obs, info

    def _get_obs(self):
        # Return the current observation
        if self.current_time_step < self.data_len:
             renewable_output_mw = self.combined_renewable_power_mw.iloc[self.current_time_step]
             grid_demand_mw = self.grid_demand_full_mw.iloc[self.current_time_step]
        else:
             # If episode is done, return a terminal observation (e.g., zeros or last state)
             # Or handle this in the step function's termination logic
             renewable_output_mw = 0
             grid_demand_mw = 0


        obs = np.array([
            self.current_time_step,
            renewable_output_mw,
            grid_demand_mw,
            self.battery_soc_kwh
        ], dtype=np.float32)

        return obs

    # You might want to add a render method for visualization
    # def render(self):
    #     pass

    # You might want to add a close method for cleanup
    # def close(self):
    #     pass


# Example usage (optional, for testing the environment independently)
if __name__ == '__main__':
    # Create an instance of the environment
    env = HybridPowerPlantEnv()

    # Reset the environment to get the initial state
    obs, info = env.reset()
    print("Initial Observation:", obs)
    print("Initial Info:", info)

    # Take a random action (example)
    random_action = env.action_space.sample()
    print("\nTaking random action:", random_action)

    # Step the environment
    next_obs, reward, terminated, truncated, info = env.step(random_action)
    print("Next Observation:", next_obs)
    print("Reward:", reward)
    print("Terminated:", terminated)
    print("Truncated:", truncated)
    print("Info:", info)

    # Run a simple simulation loop (example)
    print("\nRunning a simple simulation...")
    obs, info = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = env.action_space.sample() # Take random actions
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        # print(f"Step {env.current_time_step}: Obs={obs}, Reward={reward}, Done={done}, Info={info}")
        done = terminated or truncated
    print(f"\nSimple simulation finished with total reward: {total_reward}")
