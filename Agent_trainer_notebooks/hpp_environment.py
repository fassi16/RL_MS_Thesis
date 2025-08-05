import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import os

# Define paths for data files - ENSURE THESE PATHS AND FILENAMES ARE CORRECT
# Update the path to point to the data directory within the cloned repository
DATA_DIR_PATH = '/content/RL_MS_Thesis/data/data_testing/scenario_datasetsv/'

SOLAR_DATA_PATH = os.path.join(DATA_DIR_PATH, 'solar_profile.csv')  # Updated filename and path structure
WIND_DATA_PATH = os.path.join(DATA_DIR_PATH, 'wind_profile.csv')    # Updated filename and path structure
GRID_DATA_PATH = os.path.join(DATA_DIR_PATH, 'YOUR_DEMAND_FILE_NAME.csv')    # Update YOUR_DEMAND_FILE_NAME.csv with your actual filename and path structure


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
            # Attempt to read CSV files
            # Use the correct paths defined at the top
            solar_df = pd.read_csv(SOLAR_DATA_PATH)
            wind_df = pd.read_csv(WIND_DATA_PATH)
            grid_df = pd.read_csv(GRID_DATA_PATH)

            # Process dataframes - assuming 'time' column for index and relevant power columns
            def process_data(df, power_col_name):
                if 'time' in df.columns:
                    df['time'] = pd.to_datetime(df['time'])
                    df.set_index('time', inplace=True)
                    df.sort_index(inplace=True)
                else:
                    print(f"Warning: 'time' column not found in {power_col_name} data file.")

                # Check if the expected power column exists
                if power_col_name not in df.columns:
                     print(f"Error: '{power_col_name}' column not found in data file.")
                     # Return an empty series with the correct index if time was processed
                     return pd.Series(dtype=float, index=df.index if 'time' in df.columns else None)
                return df[power_col_name]


            # Process each data file, ensuring correct column names if necessary
            # Assuming column names are 'solar_power_MW', 'wind_power_MW', 'grid_demand_MW'
            self.solar_power_mw = process_data(solar_df.copy(), 'solar_power_MW') # Use .copy() to avoid modifying original df
            self.wind_power_mw = process_data(wind_df.copy(), 'wind_power_MW')
            self.grid_demand_full_mw = process_data(grid_df.copy(), 'grid_demand_MW')


            # Combine renewable power, filling missing values with 0 after loading and processing
            # Use the indices of the processed dataframes
            all_indices = pd.Index([])
            if not self.solar_power_mw.empty:
                all_indices = all_indices.union(self.solar_power_mw.index)
            if not self.wind_power_mw.empty:
                all_indices = all_indices.union(self.wind_power_mw.index)
            if not self.grid_demand_full_mw.empty:
                 all_indices = all_indices.union(self.grid_demand_full_mw.index)

            # Reindex and combine, filling missing values
            self.combined_renewable_power_mw = self.solar_power_mw.reindex(all_indices).fillna(0).add(
                                               self.wind_power_mw.reindex(all_indices).fillna(0), fill_value=0)

            self.grid_demand_full_mw = self.grid_demand_full_mw.reindex(all_indices).fillna(0)


        except FileNotFoundError as e:
            print(f"Error loading data file: {e}")
            # Raise the error to stop environment creation if files are missing
            raise FileNotFoundError(f"Required data file not found: {e}. Please ensure {SOLAR_DATA_PATH}, {WIND_DATA_PATH}, and {GRID_DATA_PATH} exist at these paths.")
        except Exception as e:
             print(f"An error occurred during data processing: {e}")
             # Re-raise the exception for better debugging
             raise e


        # Ensure data is not empty after loading and aligning
        # Check if the reindexed series are empty or have mismatched lengths
        if self.combined_renewable_power_mw.empty or self.grid_demand_full_mw.empty or len(self.combined_renewable_power_mw) != len(self.grid_demand_full_mw) or len(self.combined_renewable_power_mw) == 0:
            raise ValueError("Data loading or alignment failed. Renewable or grid demand data is empty or has mismatched lengths after processing. Check data files and their contents.")


        self.data_len = len(self.combined_renewable_power_mw)
        self.time_step_duration_hours = time_step_duration_hours
        self.battery_capacity_kwh = battery_capacity_kwh
        self.battery_max_charge_mw = battery_max_charge_mw
        self.battery_max_discharge_mw = battery_max_discharge_mw
        self.battery_efficiency = battery_efficiency
        self.grid_buy_price_per_kwh = grid_buy_price_per_kwh
        self.grid_sell_price_per_kwh = grid_sell_price_per_kwh
        self.penalty_unmet_demand = penalty_unmet_demand

        # Define action space: [battery_power_mw, grid_power_mw]
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
        # The upper bound for current_time_step should be data_len - 1
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
        # Ensure we don't go out of bounds if step is called after termination
        if self.current_time_step >= self.data_len:
             # Return a terminal state observation and zero reward if episode is done
             return self._get_obs(), 0, True, False, {} # Or handle as truncated


        renewable_output_mw = self.combined_renewable_power_mw.iloc[self.current_time_step]
        grid_demand_mw = self.grid_demand_full_mw.iloc[self.current_time_step]

        # --- Power Flow Logic ---

        # 1. Handle battery charge/discharge attempt
        battery_charge_attempt_mw = max(0, battery_action_mw)
        battery_discharge_attempt_mw = abs(min(0, battery_action_mw))

        actual_battery_charge_mw = 0
        actual_battery_discharge_mw = 0
        battery_delta_kwh = 0

        desired_battery_power_mw = action[0] # Positive to charge, negative to discharge

        if desired_battery_power_mw > 0: # Attempt to charge
            # Power needed for charging
            power_needed_for_charge_mw = desired_battery_power_mw
            # Max power can draw for charging
            max_charge_power_mw = min(power_needed_for_charge_mw, self.battery_max_charge_mw)
            # Max energy can store in battery
            max_charge_energy_kwh = (self.battery_capacity_kwh - self.battery_soc_kwh) # No efficiency applied here yet, apply on SOC update
            max_charge_power_based_on_capacity_mw = max_charge_energy_kwh / self.time_step_duration_hours

            actual_battery_charge_mw = min(max_charge_power_mw, max_charge_power_based_on_capacity_mw)
            battery_delta_kwh = actual_battery_charge_mw * self.time_step_duration_hours * self.battery_efficiency # Apply efficiency on charge

        elif desired_battery_power_mw < 0: # Attempt to discharge
             # Power to supply from battery
             power_to_supply_from_battery_mw = abs(desired_battery_power_mw)
             # Max power can discharge
             max_discharge_power_mw = min(power_to_supply_from_battery_mw, self.battery_max_discharge_mw)
             # Max energy can supply from battery (account for efficiency on discharge)
             max_discharge_energy_kwh = self.battery_soc_kwh * self.battery_efficiency
             max_discharge_power_based_on_capacity_mw = max_discharge_energy_kwh / self.time_step_duration_hours

             actual_battery_discharge_mw = min(max_discharge_power_mw, max_discharge_power_based_on_capacity_mw)
             battery_delta_kwh = -actual_battery_discharge_mw * self.time_step_duration_hours / self.battery_efficiency # Apply efficiency on discharge

        self.battery_soc_kwh += battery_delta_kwh
        self.battery_soc_kwh = np.clip(self.battery_soc_kwh, 0, self.battery_capacity_kwh) # Ensure SOC is within bounds


        # 2. Calculate net power available/needed from the system (excluding grid interaction for now)
        # Net power = Renewable Output + Actual Battery Discharge - Actual Battery Charge
        # Positive net_system_output_mw means surplus power available
        # Negative net_system_output_mw means deficit power needed
        net_system_output_mw = renewable_output_mw + actual_battery_discharge_mw - actual_battery_charge_mw

        # 3. Handle grid interaction and meet demand
        # Let's interpret action[1] as desired NET grid flow (positive = sell, negative = buy).

        desired_net_grid_flow_mw = action[1]

        power_sold_mw = 0
        power_bought_mw = 0
        unmet_demand_mw = 0
        net_grid_flow_mw = 0 # Actual net flow

        # The total power balance must be:
        # net_system_output_mw + Power from Grid - Power to Grid = Grid Demand
        # Power from Grid - Power to Grid = Grid Demand - net_system_output_mw
        # Actual net grid flow = Power from Grid - Power to Grid

        # If desired net flow is positive (selling):
        if desired_net_grid_flow_mw > 0:
            # Power available to sell = max(0, net_system_output_mw - grid_demand_mw)
            # Actual power sold is limited by desired amount and available power
            power_sold_mw = min(desired_net_grid_flow_mw, max(0.0, net_system_output_mw - grid_demand_mw)) # Ensure available power is non-negative
            power_bought_mw = 0 # Cannot sell and buy simultaneously (net)
            net_grid_flow_mw = power_sold_mw # Net flow is positive (to grid)
            # Calculate unmet demand after selling
            power_after_selling = net_system_output_mw - power_sold_mw
            unmet_demand_mw = max(0.0, grid_demand_mw - power_after_selling) # Unmet if remaining power is less than demand

        # If desired net flow is negative (buying):
        elif desired_net_grid_flow_mw < 0:
             # Power needed from grid = abs(desired_net_grid_flow_mw)
             power_to_buy_mw = abs(desired_net_grid_flow_mw)

             # Actual power bought - agent tries to buy this much
             power_bought_mw = power_to_buy_mw
             power_sold_mw = 0 # Cannot sell and buy simultaneously (net)
             net_grid_flow_mw = -power_bought_mw # Net flow is negative (from grid)

             # Check if buying was sufficient to meet demand
             power_after_grid_buy = net_system_output_mw + power_bought_mw
             unmet_demand_mw = max(0.0, grid_demand_mw - power_after_grid_buy) # Unmet if total available is less than demand


        # If desired net flow is zero:
        else: # desired_net_grid_flow_mw == 0
            power_sold_mw = 0
            power_bought_mw = 0
            net_grid_flow_mw = 0
            # Unmet demand is the deficit if system output isn't enough to meet demand
            unmet_demand_mw = max(0.0, grid_demand_mw - net_system_output_mw)


        # Convert power flows (MW) to energy (kWh) for the time step
        power_bought_kwh = power_bought_mw * self.time_step_duration_hours
        power_sold_kwh = power_sold_mw * self.time_step_duration_hours
        unmet_demand_kwh = unmet_demand_mw * self.time_step_duration_hours
        renewable_output_kwh = renewable_output_mw * self.time_step_duration_hours # Energy generated in this step


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
        # reward -= abs(battery_action_mw) * 0.001 # Example penalty, adjust weight

        # Optional: Reward for keeping battery within a desired range (e.g., avoiding empty/full)
        # soc_penalty = 0
        # if self.battery_soc_kwh < self.battery_capacity_kwh * 0.1: # Example: penalty if below 10%
        #      soc_penalty += (self.battery_capacity_kwh * 0.1 - self.battery_soc_kwh) * 0.05
        # if self.battery_soc_kwh > self.battery_capacity_kwh * 0.9: # Example: penalty if above 90%
        #      soc_penalty += (self.battery_soc_kwh - self.battery_capacity_kwh * 0.9) * 0.05
        # reward -= soc_penalty


        # --- Update State ---
        self.current_time_step += 1

        # --- Check if episode is done ---
        # The episode is done if we have processed all time steps
        terminated = self.current_time_step >= self.data_len
        truncated = False # Or define other conditions for truncation if needed

        # --- Create Info Dictionary ---
        # Log information about the state *before* the next step
        info = {
            "current_time_step": self.current_time_step -1, # Log the time step that just completed
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

        # Ensure observation is valid even if episode terminates
        if terminated:
            next_obs = self._get_obs(is_terminal=True) # Get a terminal observation if applicable


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
             "renewable_output_mw": self.combined_renewable_power_mw.iloc[self.current_time_step] if self.data_len > 0 else 0,
             "grid_demand_mw": self.grid_demand_full_mw.iloc[self.current_time_step] if self.data_len > 0 else 0,
             "battery_soc_kwh": self.battery_soc_kwh,
             # Include other initial info if needed, e.g., prices
         }

        return obs, info

    def _get_obs(self, is_terminal=False):
        # Return the current observation
        if self.current_time_step < self.data_len and not is_terminal:
             renewable_output_mw = self.combined_renewable_power_mw.iloc[self.current_time_step]
             grid_demand_mw = self.grid_demand_full_mw.iloc[self.current_time_step]
        else:
             # If episode is done or getting terminal observation, return a state
             # that doesn't depend on the data index being valid.
             # A common practice is to return the last valid state or zeros,
             # but for Box observation space, returning the last valid data point
             # is usually expected by RL libraries when done=True.
             if self.data_len > 0:
                  # Return values from the last time step if accessing beyond data
                  renewable_output_mw = self.combined_renewable_power_mw.iloc[min(self.current_time_step, self.data_len -1)]
                  grid_demand_mw = self.grid_demand_full_mw.iloc[min(self.current_time_step, self.data_len -1)]
             else:
                  # Handle case with no data loaded
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
    # Note: This part will only run if the script is executed directly, not when imported
    try:
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
        step_count = 0
        # Limit simulation steps to prevent infinite loop
        max_steps = env.data_len + 5 if env.data_len > 0 else 100 # Run a few steps beyond data end if possible, or fixed steps if no data
        while not done and step_count < max_steps:
            action = env.action_space.sample() # Take random actions
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            # print(f"Step {env.current_time_step}: Obs={obs}, Reward={reward}, Done={done}, Info={info}")
            done = terminated or truncated
        print(f"\nSimple simulation finished after {step_count} steps with total reward: {total_reward}")

    except FileNotFoundError as e:
        print(f"Environment test failed due to missing file: {e}")
    except ValueError as e:
        print(f"Environment test failed due to data error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during environment test: {e}")
