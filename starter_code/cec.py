import casadi
import numpy as np


class CEC:
    def __init__(
        self,
        delta: float,      
        horizon: int,
        γ: float,     
        Q: np.ndarray, 
        q:float,
        R: np.ndarray,
        obstacles: np.ndarray,
        r_robot : float,
        ref_traj: np.ndarray,  
        ) -> None:
        
        """
        Δ:          sampling time
        horizon:    planning horizon T
        γ:          discount factor
        Q:          2×2 symmetric matrix defining the stage cost for deviating from the reference position trajectory
        q:          scalar defining the stage cost for deviating from thr reference orientation trajectory
        R:          2×2 symmetric matrix defining the stage cost for using excessive control effort
        obstacles:  NumPy array of shape (n_obs, 3), each row = [x_center, y_center, r_obs]
        r_robot:    robot radius (scalar)
        ref_traj:   NumPy array of shape (N_steps, 3), each row = [r_x, r_y, α]
        """
        
        # 1. Store parameters
        self.delta = delta
        self.horizon = horizon
        self.γ = γ
        self.Q = casadi.DM(Q)           # 2x2
        self.q_weight = q               # scalar
        self.R = casadi.DM(R)           # 2×2
        self.obstacles = obstacles      # shape = (n_obs, 3)
        self.r_robot = r_robot          # scalar
        self.ref_traj = ref_traj        # shape = (N_steps, 3)
    
    def dynamic_without_noise(self,e_t,u_t,t_idx):
        """
        Compute e_{t+1} = f(e_t, u_t) under deterministic kinematics.
        e_t:   3×1 CasADi vector [tilde_px, tilde_py, tilde_theta]
        u_t:   2×1 CasADi vector [v_k, omega_k]
        t_index:  integer index into ref_traj

        Returns a 3×1 CasADi vector e_{t+1}.
        """
        #decompose e_t
        tilde_px = e_t[0]     # ~pt,x = pt,x - rt,x
        tilde_py = e_t[1]     # ~pt,y = pt,y - rt,y
        tilde_th = e_t[2]     # ~θt = θt - αt
        
        # Based on current error + reference, first reverse-calculate p_t, θ_t
        # Since tilde_p = p_t - r_t  => p_t = tilde_p + r_t
        # Therefore: reconstruct the actual position/angle from the error signal
        r_tx = self.ref_traj[t_idx, 0]
        r_ty = self.ref_traj[t_idx, 1]
        alpha_t = self.ref_traj[t_idx, 2]
        
        p_tx = tilde_px + r_tx
        p_ty = tilde_py + r_ty
        theta_t = tilde_th + alpha_t
        
        #decompose u_t
        v_k = u_t[0]
        omega_k = u_t[1]
        
        # compute  p_{t+1} (deterministic, w=0)
        omegaD = omega_k * self.delta /2.0 
        sinc = casadi.sin(omegaD) / (omegaD + 1e-8)
        delta_px = v_k *self.delta * sinc * casadi.cos(theta_t + omegaD)
        delta_py = v_k * self.delta * sinc * casadi.sin(theta_t + omegaD)
        p_t1x = p_tx + delta_px
        p_t1y = p_ty + delta_py
        
        #compute theta{t+1}
        theta_t1 = theta_t + omega_k * self.delta
        
        #compute tilde_px_next, tlide_py_next
        nxt = t_idx + 1
        r_t1x = self.ref_traj[nxt, 0]
        r_t1y = self.ref_traj[nxt, 1]
        alpha_t1 = self.ref_traj[nxt, 2]
        tilde_px_next = p_t1x - r_t1x
        tilde_py_next = p_t1y - r_t1y
        
        #compute tlide_theta_next
        raw_th_diff = theta_t1 - alpha_t1
        tilde_th_next = casadi.atan2(casadi.sin(raw_th_diff), casadi.cos(raw_th_diff))  #make sure angles are in (−π,π)
        
        return casadi.vertcat(tilde_px_next, tilde_py_next, tilde_th_next)  
    
   
    def __call__(self, t: int, cur_state: np.ndarray, cur_ref_state: np.ndarray) -> np.ndarray:
        """
        Given the time step, current state, and reference state, return the control input.
        Args:
            t (int): time step
            cur_state (np.ndarray): current state
            cur_ref_state (np.ndarray): reference state
        Returns:
            np.ndarray: control input
        """
        # TODO: define optimization variables
        # --- 1. compute et ---
        px = casadi.DM(cur_state[0])
        py = casadi.DM(cur_state[1])
        theta = casadi.DM(cur_state[2])

        r_tx = casadi.DM(cur_ref_state[0])
        r_ty = casadi.DM(cur_ref_state[1])
        alpha = casadi.DM(cur_ref_state[2])

        tilde_px_0 = px - r_tx
        tilde_py_0 = py - r_ty
        raw_th0 = theta - alpha
        tilde_th_0 = casadi.atan2(casadi.sin(raw_th0), casadi.cos(raw_th0))
        e0 = casadi.vertcat(tilde_px_0, tilde_py_0, tilde_th_0)
        
        # 2) Create optimization variables U (2T×1) and E (3T×1)
        T = self.horizon
        U = casadi.MX.sym('U', 2 * T)    # [v₀, ω₀, v₁, ω₁, …, v_{T-1}, ω_{T-1}]
        E = casadi.MX.sym('E', 3 * T)    # [e_{t+1} (3), e_{t+2} (3), …, e_{t+T} (3)]
        X = casadi.vertcat(U, E)         # shape = (5*T, 1)
        
        # TODO: define optimization constraints and optimization objective
        # dynamics constraints
        constr = []
        u0 = U[0:2]
        e1_var = E[0:3]  
        e1_pred = self.dynamic_without_noise(e0, u0, t)
        constr.append(e1_var - e1_pred)
        
        for k in range(1, T):
            ek_var = E[3 * (k - 1) : 3 * (k - 1) + 3]
            uk = U[2 * k : 2 * k + 2]
            ek1_var = E[3 * k : 3 * k + 3]
            ek1_pred = self.dynamic_without_noise(ek_var, uk, t + k )
            constr.append(ek1_var - ek1_pred)
            
        # obstacle-avoidance constraints
        n_obs = self.obstacles.shape[0]
        for k in range(0, T + 1):
            tau = t + k
            if k == 0:
                e_k = e0         
            else:
                e_k = E[3 * (k - 1) : 3 * (k - 1) + 3]
       
            r_tau_x = self.ref_traj[tau, 0]
            r_tau_y = self.ref_traj[tau, 1]

            ptilde = e_k[0:2]                                  # 2×1 

            
            p_tau = casadi.vertcat(r_tau_x, r_tau_y) + ptilde  # 2×1

            # For each obstacle i:
            for i in range(n_obs):
                center_i = self.obstacles[i, 0:2]              # shape=(2,) x_i, y_i
                r_obs_i = self.obstacles[i, 2]                 # scalar     r_obs_i
                ci = casadi.DM(center_i)
                dist_minus = casadi.norm_2(p_tau - ci) - (r_obs_i + self.r_robot)
                constr.append(dist_minus)
                
        # Stack all constraints into one CasADi vector g  
        g = casadi.vertcat(*constr)    
            
        # Objective
        obj=0
        for k in range(0, T):
            # e_{t+k}
            if k == 0:
                e_k = e0
            else:
                e_k = E[3 * (k - 1) : 3 * (k - 1) + 3]
            ptilde_k = e_k[0:2]         # shape = (2,1)
            u_k = U[2 * k : 2 * k + 2]  # shape = (2,1)  
            part1 = casadi.mtimes([ptilde_k.T, self.Q, ptilde_k])
            # part2 = (q_weight)*(1 - cos(e_k[2]))^2
            part2 = (self.q_weight) * (1 - casadi.cos(e_k[2]))**2
            # part3 = uₖᵀ R uₖ
            part3 = casadi.mtimes([u_k.T, self.R, u_k])
            obj += (self.γ**k) * (part1 + part2 + part3)
            
        #terminal cost q(T)   
        e_T = E[3*(T-1) : 3*(T-1) + 3] 
        pT = e_T[0:2]
        thetaT = e_T[2]
        terminal_part1 = casadi.mtimes([pT.T, self.Q, pT])
        terminal_part2 = self.q_weight * (1 - casadi.cos(thetaT))**2
        obj = obj + terminal_part1 + terminal_part2
        
        # TODO: define optimization solver
         # nlp = { 'x': decision_vars, 'f': objective, 'constr': constraints }
        nlp = {'x':X , 'f': obj, 'g': g}
        solver = casadi.nlpsol("S", "ipopt", nlp)
        
        #  Build bounds for X = [U; E] and for g
        inf = casadi.inf
        lbx = np.zeros((5*T, 1))       # shape[5T,1]
        ubx = np.zeros((5*T, 1))       # shape[5T,1]
        
        #  Bounds on U (first 2T entries of X):
        #  v_k ∈ [0.1, 1.0],  omega_k ∈ [−1, +1]
        for i in range(5*T):
            lbx[i,0] = -inf
            ubx[i,0] = +inf
        for k in range(T):
            # v_k = U[2*k], range = [0.1, 1.0]
            lbx[2*k, 0] = 0.1
            ubx[2*k, 0] = 1.0
            # ω_k = U[2*k + 1], range = [−1.0, +1.0]
            lbx[2*k + 1, 0] = -1.0
            ubx[2*k + 1, 0] = +1.0
            
        # Bounds on E (next 3T entries): leave as [−inf, +inf], because no explicit limit.

        #  Bounds on g (constraints):
        #  First 3T constraints are dynamics equalities → lbg = ubg = 0
        #  Next (T+1)*n_obs constraints are obstacle avoidances → lbg = 0, ubg = +inf
        #
        n_constr = g.size()[0]
        lbg = np.zeros((n_constr, 1))
        ubg = np.zeros((n_constr, 1))
        
        # By default set all to [0, +inf]
        for i in range(n_constr):
            lbg[i, 0] = 0
            ubg[i, 0] = +inf
            
        # Override first 3T constraints to be equality: lbg=ubg=0
        for i in range(3*T):
            lbg[i, 0] = 0
            ubg[i, 0] = 0 
        # Provide an initial guess x0   
        x0 = np.zeros((5*T, 1))
        for k in range(T):
            x0[2*k, 0]     = 0.5   #  v_k initial
            x0[2*k+1, 0]   = 0.0   # omega_k initial
            
        # For E part: default 0
        sol = solver(
            x0=x0,   # TODO: initial guess
            lbx=lbx, # TODO: lower bound on optimization variables
            ubx=ubx, # TODO: upper bound on optimization variables
            lbg=lbg, # TODO: lower bound on optimization constraints
            ubg=ubg, # TODO: upper bound on optimization constraints
        )
        x = sol["x"]  # get the solution

        # TODO: extract the control input from the solution
        u = x[0:2]
        u = np.array(u).reshape(2,)
        return u
