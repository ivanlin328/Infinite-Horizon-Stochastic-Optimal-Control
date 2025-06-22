from dataclasses import dataclass
import numpy as np
import ray
from tqdm import tqdm
import itertools
from scipy.stats import multivariate_normal

from value_function import GridValueFunction, ValueFunction
import utils

@dataclass
class GpiConfig:
    traj: callable
    obstacles: np.ndarray
    ex_space: np.ndarray
    ey_space: np.ndarray
    eth_space: np.ndarray
    v_space: np.ndarray
    w_space: np.ndarray
    Q: np.ndarray
    q: float
    R: np.ndarray
    gamma: float
    num_evals: int
    collision_margin: float
    ref_traj: np.ndarray
    output_dir: str
    V: ValueFunction = None

# Initialize Ray for parallel processing
if not ray.is_initialized():
    ray.init(ignore_reinit_error=True)


class GPI:
    def __init__(self, config: GpiConfig):
        self.config = config
        self.period = len(config.ref_traj)
        self.nx, self.ny, self.nth = len(config.ex_space), len(config.ey_space), len(config.eth_space)
        self.nv, self.nw = len(config.v_space), len(config.w_space)
        self.states = list(itertools.product(range(self.nx), range(self.ny), range(self.nth)))
        self.actions = list(itertools.product(range(self.nv), range(self.nw)))
        self.V = GridValueFunction(self.period, config.ex_space, config.ey_space, config.eth_space)
        self.policy = np.zeros((self.period, self.nx, self.ny, self.nth, 2), dtype=int)
        
        print("Pre-computing transition probabilities and stage costs...")
        self.transitions, self.costs = self.precompute_transitions_and_costs()
        print("Pre-computation finished.")

    def __call__(self, t: int, cur_state: np.ndarray, cur_ref_state: np.ndarray) -> np.ndarray:
        error_state = cur_state - cur_ref_state
        error_state[2] = np.arctan2(np.sin(error_state[2]), np.cos(error_state[2]))
        t_idx = t % self.period
        state_idx = self.state_metric_to_index(error_state)
        action_idx = self.policy[t_idx, state_idx[0], state_idx[1], state_idx[2]]
        v_metric, w_metric = self.control_index_to_metric(np.array([action_idx[0]]), np.array([action_idx[1]]))
        return np.array([v_metric[0], w_metric[0]])

    def state_metric_to_index(self, metric_state: np.ndarray) -> tuple:
        ix = np.argmin(np.abs(self.config.ex_space - metric_state[0]))
        iy = np.argmin(np.abs(self.config.ey_space - metric_state[1]))
        angle_diffs = np.arctan2(np.sin(metric_state[2] - self.config.eth_space), np.cos(metric_state[2] - self.config.eth_space))
        ith = np.argmin(np.abs(angle_diffs))
        return (ix, iy, ith)
    
    def control_index_to_metric(self, v_idx: np.ndarray, w_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.config.v_space[v_idx], self.config.w_space[w_idx]

    @utils.timer
    def precompute_transitions_and_costs(self):
        num_actions = len(self.actions)
        num_neighbors = 8
        
        results = ray.get([
            precompute_worker_batch.remote(self.config, t, self.states, self.actions, num_neighbors)
            for t in tqdm(range(self.period), desc="Dispatching pre-computation jobs")
        ])
        
        transitions = np.zeros((self.period, self.nx, self.ny, self.nth, num_actions, num_neighbors, 3), dtype=int)
        transition_probs = np.zeros((self.period, self.nx, self.ny, self.nth, num_actions, num_neighbors))
        costs = np.full((self.period, self.nx, self.ny, self.nth, num_actions), np.inf)

        for t, batch_results in enumerate(results):
            for s_idx, (res_transitions, res_probs, res_costs) in batch_results.items():
                ix, iy, ith = s_idx
                transitions[t, ix, iy, ith] = res_transitions
                transition_probs[t, ix, iy, ith] = res_probs
                costs[t, ix, iy, ith] = res_costs

        return (transitions, transition_probs), costs

    def compute_policy(self, max_iters: int) -> None:
        for i in range(max_iters):
            print(f"\n--- GPI Iteration {i+1}/{max_iters} ---")
            self.policy_improvement()
            for j in range(self.config.num_evals):
                print(f"--- Policy Evaluation {j+1}/{self.config.num_evals} ---")
                self.policy_evaluation()

    # =================================================================================
    # MODIFIED TO USE RAY.PUT FOR LARGE, SHARED ARRAYS
    # =================================================================================
    @utils.timer
    def policy_evaluation(self):
        # PUT references to the large, read-only arrays into the object store ONCE
        V_old_ref = ray.put(self.V.v.copy())
        transitions_ref = ray.put(self.transitions)
        costs_ref = ray.put(self.costs)
        
        batched_new_values = ray.get([
            # Pass references instead of the full arrays
            evaluation_worker_batch.remote(self.config, t, self.states, self.policy, transitions_ref, costs_ref, V_old_ref)
            for t in tqdm(range(self.period), desc="Policy Evaluating")
        ])

        for t, new_values_dict in enumerate(batched_new_values):
            for s_idx, value in new_values_dict.items():
                self.V.v[t, s_idx[0], s_idx[1], s_idx[2]] = value

    @utils.timer
    def policy_improvement(self):
        # PUT references to the large, read-only arrays into the object store ONCE
        V_curr_ref = ray.put(self.V.v)
        transitions_ref = ray.put(self.transitions)
        costs_ref = ray.put(self.costs)
        
        batched_new_actions = ray.get([
            # Pass references instead of the full arrays
            improvement_worker_batch.remote(self.config, t, self.states, self.actions, transitions_ref, costs_ref, V_curr_ref)
            for t in tqdm(range(self.period), desc="Policy Improving")
        ])

        for t, new_actions_dict in enumerate(batched_new_actions):
            for s_idx, action_idx in new_actions_dict.items():
                self.policy[t, s_idx[0], s_idx[1], s_idx[2]] = self.actions[action_idx]


# --- Ray Worker Functions (defined outside the class) ---

def get_next_error_state_mean(config, t, error_state, control):
    dt = utils.time_step
    v, w = control
    p_err, th_err = error_state[:2], error_state[2]
    
    ref_state = np.array(config.ref_traj[t])
    ref_state_next = np.array(config.ref_traj[(t + 1) % len(config.ref_traj)])
    
    alpha_t = ref_state[2]
    theta_t = th_err + alpha_t
    
    sinc_arg = w * dt / 2
    if abs(sinc_arg) < 1e-9:
        sinc_val = 1.0
    else:
        sinc_val = np.sin(sinc_arg) / sinc_arg

    p_world_inc = np.array([
        dt * sinc_val * np.cos(theta_t + w * dt / 2) * v,
        dt * sinc_val * np.sin(theta_t + w * dt / 2) * v
    ])
    th_world_inc = dt * w
    
    p_err_next = p_err + p_world_inc - (ref_state_next[:2] - ref_state[:2])
    th_err_next = th_err + th_world_inc - (ref_state_next[2] - ref_state[2])
    th_err_next = np.arctan2(np.sin(th_err_next), np.cos(th_err_next))

    return np.array([p_err_next[0], p_err_next[1], th_err_next])

def calculate_obstacle_cost(world_pos, obstacles, robot_radius, warning_scale=2.0, penalty_k=10.0):
    
    cost = 0.0
    for obs in obstacles:
        obs_pos, obs_rad = obs[:2], obs[2]
        dist = np.linalg.norm(world_pos - obs_pos)
        
        
        warning_dist = obs_rad + robot_radius
        
        if dist < warning_dist * warning_scale:
            
            if dist < warning_dist:
                
                cost += penalty_k * (warning_dist / (dist + 1e-6)) 
            else:
                
                cost += penalty_k / (dist - warning_dist)**2

    
    total_penalty = 0.0
    for obs in obstacles:
        obs_pos, obs_rad = obs[:2], obs[2]
        dist = np.linalg.norm(world_pos - obs_pos) - obs_rad - robot_radius
        if dist < 1.0: 
            total_penalty += penalty_k * np.exp(-5.0 * dist)

    return total_penalty


@ray.remote
def precompute_worker_batch(config: GpiConfig, t: int, states: list, actions: list, num_neighbors: int):
    results = {}
    nx, ny, nth = len(config.ex_space), len(config.ey_space), len(config.eth_space)
    num_actions = len(actions)
    noise_dist = multivariate_normal(mean=np.zeros(3), cov=np.diag(utils.sigma**2))

    for s_idx in states:
        metric_state = np.array([config.ex_space[s_idx[0]], config.ey_space[s_idx[1]], config.eth_space[s_idx[2]]])
        
        transitions = np.zeros((num_actions, num_neighbors, 3), dtype=int)
        probs = np.zeros((num_actions, num_neighbors))
        # MODIFICATION: We will now calculate costs, not leave them as np.inf
        costs = np.zeros(num_actions)

        # --- MODIFICATION START ---
       
        world_pos = metric_state[:2] + config.ref_traj[t][:2]
        base_obs_cost = calculate_obstacle_cost(world_pos, config.obstacles, config.collision_margin)
        # --- MODIFICATION END ---
        
        for a_idx, action_indices in enumerate(actions):
            v_metric = config.v_space[action_indices[0]]
            w_metric = config.w_space[action_indices[1]]
            metric_action = np.array([v_metric, w_metric])

           
            base_cost = (metric_state[:2].T @ config.Q @ metric_state[:2] + 
                         config.q * (1 - np.cos(metric_state[2]))**2 + 
                         metric_action.T @ config.R @ metric_action)
            
            e_next_mean = get_next_error_state_mean(config, t, metric_state, metric_action)

            # --- MODIFICATION START ---
            
            next_world_pos = e_next_mean[:2] + config.ref_traj[(t + 1) % len(config.ref_traj)][:2]
            next_obs_cost = calculate_obstacle_cost(next_world_pos, config.obstacles, config.collision_margin)

           
            costs[a_idx] = base_cost + base_obs_cost + next_obs_cost
            # --- MODIFICATION END ---
            
            # --- REMOVED THE HARD COLLISION CHECK ---
            

            ex_idx_lo = np.clip(np.searchsorted(config.ex_space, e_next_mean[0], side='right') - 1, 0, nx - 2)
            ey_idx_lo = np.clip(np.searchsorted(config.ey_space, e_next_mean[1], side='right') - 1, 0, ny - 2)
            eth_idx_lo = np.clip(np.searchsorted(config.eth_space, e_next_mean[2], side='right') - 1, 0, nth - 2)

           
            neighbor_indices = []
            neighbor_probs = []
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        next_e_idx = (ex_idx_lo + i, ey_idx_lo + j, eth_idx_lo + k)
                        metric_neighbor = np.array([config.ex_space[next_e_idx[0]], config.ey_space[next_e_idx[1]], config.eth_space[next_e_idx[2]]])
                        w = metric_neighbor - e_next_mean
                        w[2] = np.arctan2(np.sin(w[2]), np.cos(w[2]))
                        prob = noise_dist.pdf(w)
                        neighbor_probs.append(prob)
                        neighbor_indices.append(next_e_idx)
            
            total_prob = sum(neighbor_probs)
            if total_prob > 1e-9:
                normalized_probs = [p / total_prob for p in neighbor_probs]
                for i in range(8):
                    transitions[a_idx, i, :] = neighbor_indices[i]
                    probs[a_idx, i] = normalized_probs[i]
            else:
                ix_c = np.argmin(np.abs(config.ex_space - e_next_mean[0]))
                iy_c = np.argmin(np.abs(config.ey_space - e_next_mean[1]))
                angle_diffs_c = np.arctan2(np.sin(e_next_mean[2] - config.eth_space), np.cos(e_next_mean[2] - config.eth_space))
                ith_c = np.argmin(np.abs(angle_diffs_c))
                transitions[a_idx, 0, :] = (ix_c, iy_c, ith_c)
                probs[a_idx, 0] = 1.0

        results[s_idx] = (transitions, probs, costs)
    return results

@ray.remote
def evaluation_worker_batch(config: GpiConfig, t: int, states: list, policy: np.ndarray, transitions: tuple, costs: np.ndarray, V_old: np.ndarray):
    results = {}
    next_t = (t + 1) % len(config.ref_traj)
    trans_mat, trans_probs = transitions

    for s_idx in states:
        ix, iy, ith = s_idx
        action_idx = tuple(policy[t, ix, iy, ith])
        action_num = action_idx[0] * len(config.w_space) + action_idx[1]
        
        stage_cost = costs[t, ix, iy, ith, action_num]
        if np.isinf(stage_cost):
            results[s_idx] = np.inf
            continue

        neighbor_indices = trans_mat[t, ix, iy, ith, action_num]
        neighbor_probs = trans_probs[t, ix, iy, ith, action_num]
        
        expected_future_value = 0
        for i in range(len(neighbor_indices)):
            nx_idx, ny_idx, nth_idx = neighbor_indices[i].astype(int)
            prob = neighbor_probs[i]
            if prob > 0:
                expected_future_value += prob * V_old[next_t, nx_idx, ny_idx, nth_idx]
        
        results[s_idx] = stage_cost + config.gamma * expected_future_value
    return results

@ray.remote
def improvement_worker_batch(config: GpiConfig, t: int, states: list, actions: list, transitions: tuple, costs: np.ndarray, V: np.ndarray):
    results = {}
    num_actions = len(actions)
    next_t = (t + 1) % len(config.ref_traj)
    trans_mat, trans_probs = transitions

    for s_idx in states:
        ix, iy, ith = s_idx
        q_values = np.full(num_actions, np.inf)

        for a_idx in range(num_actions):
            stage_cost = costs[t, ix, iy, ith, a_idx]
            if np.isinf(stage_cost):
                continue
                
            neighbor_indices = trans_mat[t, ix, iy, ith, a_idx]
            neighbor_probs = trans_probs[t, ix, iy, ith, a_idx]
            
            expected_future_value = 0
            for i in range(len(neighbor_indices)):
                nx_idx, ny_idx, nth_idx = neighbor_indices[i].astype(int)
                prob = neighbor_probs[i]
                if prob > 0:
                    expected_future_value += prob * V[next_t, nx_idx, ny_idx, nth_idx]
                
            q_values[a_idx] = stage_cost + config.gamma * expected_future_value
        
        results[s_idx] = np.argmin(q_values)
    return results