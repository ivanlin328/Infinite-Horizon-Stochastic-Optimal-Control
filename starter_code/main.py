from time import time
import numpy as np
import utils
from mujoco_car import MujocoCarSim
from cec import CEC
from gpi import GPI, GpiConfig
from value_function import GridValueFunction
from tqdm import tqdm


use_mujoco = False
use_gpi   =False 
use_cec = True

class ErrorDynamics:
    def __init__(self, delta, sigma, horizon):
        self.delta   = delta
        self.sigma   = sigma      
        self.horizon = horizon    

    def __len__(self):
        
        return self.horizon
       

def main():
    # Obstacles in the environment
    obstacles = np.array([[-2, -2, 0.5], [1, 2, 0.5]])

    # Params
    traj = utils.lissajous
    ref_traj = []
    error_trans = 0.0
    error_rot = 0.0
    car_states = []
    times = []
    delta   = utils.time_step                 
    horizon = 15                      
    γ       = 0.98
                                
    Q       = np.diag([10.0, 10.0])             
    q       = 5.0                             
    R       = np.diag([0.05, 0.05])             
    r_robot = 0.3     
   
    N_steps  = int(utils.sim_time / utils.time_step)
    ref_traj_full = np.zeros((N_steps + horizon, 3))

    for k in range(N_steps):
        xk, yk = traj(k)[:2]          
        xk1, yk1 = traj(k + 1)[:2]   
        alpha_k = np.arctan2(yk1 - yk, xk1 - xk)  
        ref_traj_full[k] = (xk, yk, alpha_k)

   
    ref_traj_full[N_steps - 1, 2] = ref_traj_full[N_steps - 2, 2]

    
    for k in range(N_steps, N_steps + horizon):
        ref_traj_full[k] = ref_traj_full[N_steps - 1]
    
    if use_cec:  
        cec = CEC(
        delta   = delta,
        horizon = horizon,
        γ       = γ,
        Q       = Q,
        q       = q,
        R       = R,
        obstacles = obstacles,
        r_robot   = r_robot,
        ref_traj  = ref_traj_full     
    )
    if use_gpi:
        # discretization grids
        ex_space  = np.linspace(-3, 3, 29)
        ey_space  = np.linspace(-3, 3, 29)
        eth_space = np.linspace(-np.pi, np.pi,20)
        v_space   = np.linspace(utils.v_min, utils.v_max, 5)
        w_space   = np.linspace(utils.w_min, utils.w_max ,5)
        num_evals = 2
        
        period = 100
        ref_period = ref_traj_full[:period]
        gpi_cfg = GpiConfig(
            traj             = utils.car_next_state,
            obstacles        = obstacles,
            ex_space         = ex_space,
            ey_space         = ey_space,
            eth_space        = eth_space,
            v_space          = v_space,
            w_space          = w_space,
            Q                = Q,
            q                = q,
            R                = R,
            gamma            = γ,
            num_evals        = num_evals,
            collision_margin = r_robot,
            ref_traj         = ref_period,
            output_dir       = None
        )
        gpi = GPI(gpi_cfg)
        gpi.compute_policy(max_iters=10)
       
      
    # Start main loop
    main_loop = time()  # return time in sec

    # Initialize state
    cur_state = np.array([utils.x_init, utils.y_init, utils.theta_init])
    cur_iter = 0

    # Initialize Mujoco simulation environment
    mujoco_sim = None
    if use_mujoco:
        mujoco_sim = MujocoCarSim()
        
        
    
    # Main loop
    total_steps = int(utils.sim_time / utils.time_step)
    for cur_iter in tqdm(range(total_steps), desc="Simulating"):
        t1 = time()
        # Get reference state
        cur_time = cur_iter * utils.time_step
        cur_ref = traj(cur_iter)
        # Save current state and reference state for visualization
        ref_traj.append(cur_ref)
        car_states.append(cur_state)

        ################################################################
        # Generate control input
        # TODO: Replace this simple controller with your own controller
        if use_gpi:
             control = gpi(cur_iter, cur_state, cur_ref)
        else:
             control = cec(cur_iter, cur_state, cur_ref)

        ################################################################

        # Apply control input
        if use_mujoco:
            next_state = mujoco_sim.car_next_state(control)
        else:
            next_state = utils.car_next_state(utils.time_step, cur_state, control, noise=True)

        # Update current state
        cur_state = next_state
        # Loop time
        t2 = utils.time()
        print(cur_iter)
        print(t2 - t1)
        times.append(t2 - t1)
        cur_err = cur_state - cur_ref
        cur_err[2] = np.arctan2(np.sin(cur_err[2]), np.cos(cur_err[2]))
        error_trans = error_trans + np.linalg.norm(cur_err[:2])
        error_rot = error_rot + np.abs(cur_err[2])
        print(cur_err, error_trans, error_rot)
        print("======================")
        cur_iter = cur_iter + 1

    main_loop_time = time()
    print("\n\n")
    print("Total time: ", main_loop_time - main_loop)
    print("Average iteration time: ", np.array(times).mean() * 1000, "ms")
    print("Final error_trains: ", error_trans)
    print("Final error_rot: ", error_rot)

    # Proper shunt down mujoco
    if use_mujoco:
        mujoco_sim.viewer_handle.close()

    # Visualization
    ref_traj = np.array(ref_traj)
    car_states = np.array(car_states)
    times = np.array(times)
    utils.visualize(car_states, ref_traj, obstacles, times, utils.time_step, save=True)


if __name__ == "__main__":
    main()

