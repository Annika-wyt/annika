import numpy as np
from scipy.integrate import solve_ivp
from copy import deepcopy
from scipy.spatial.transform import Rotation as ScipyRot
import matplotlib.pyplot as plt
import time as systemtime
import pandas as pd
class riccati_observer():
    def __init__(self, **kwargs):
        ######################################################
        ##################### Parameters #####################

        # for key, value in kwargs.items():
            # setattr(self, key, value)

        self.which_eq = kwargs.get('which_eq', 0)
        self.stepsize = kwargs.get('stepsize', 0.1)
        self.tol = kwargs.get('tol', 1e-2 * 3) 

        # self.with_image_hz_sim = kwargs.get('with_image_hz_sim', False)
        # self.randomize_image_input = kwargs.get('randomize_image_input', False)
        # self.noise = kwargs.get('noise', False)
        # self.which_omega = kwargs.get('which_omega','full') # "z" or "full"
        # self.time = kwargs.get('time',(0, 10))
        # self.image_hz = kwargs.get('image_hz', 20)
        # self.image_time = np.arange(self.time[0], self.time[1]+1/self.image_hz, 1/self.image_hz)
        # self.use_adaptive = kwargs.get('use_adaptive', True)
        # tol = 1e-2 * 5
        # tol = 0.05

        self.soly = None
        self.solt = None
        ##################### Parameters #####################
        ######################################################

        ######################################################
        ################### initialization ###################
        # landmarks
        self.z = kwargs.get('z', np.array([])) # in svea frame
        self.k = kwargs.get('k', 1)
        self.v = kwargs.get('v', [0.1,1])
        self.l = len(self.z)
        self.q = kwargs.get('q', 10)
        self.q = self.q * self.l
        self.V = np.diag(np.hstack([np.diag(self.v[i]*np.eye(3)) for i in range(len(self.v))]))
        self.Q = np.diag(np.hstack([np.diag(self.q*np.eye(3)) for i in range(self.l)]))
        self.p_ricatti = kwargs.get('p_ricatti', [1,100])
        self.P_ricatti = np.diag(np.hstack([np.diag(self.p_ricatti[i]*np.eye(3)) for i in range(len(self.p_ricatti))]))

        self.Lambda_bar_0 = kwargs.get('Lambda_bar_0', np.array([1, 0, 0, 0]).T)  # quaternion: w, x, y, z
        self.Rot_hat = kwargs.get('Rot_hat', self.rodrigues_formula(self.Lambda_bar_0))
        self.p_hat = kwargs.get('p_hat', np.array([[0, 0, 0]], dtype=np.float64).T)
        self.p_bar_hat = self.add_bar(self.Rot_hat, self.p_hat)

        self.linearVelocity = None
        self.angularVelocity = None
        self.current_time = 0
        self.running_rk45 = False

        self.dt = self.stepsize
        ################### initialization ###################
        ######################################################

        self.print_init()

    def print_init(self):
        Q_str = '   \n'.join(['                             ' + '  '.join(map(str, row)) for row in self.Q])
        V_str = '   \n'.join(['                             ' + '  '.join(map(str, row)) for row in self.V])
        P_ricatti_str = '   \n'.join(['                             ' + '  '.join(map(str, row)) for row in self.P_ricatti])

        print(f"""
        Parameters
        stepsize               | {self.stepsize}
        tol                    | {self.tol}
        which_eq               | {self.which_eq}
        k                      | {self.k}
        Q                      |
    {Q_str}
        V                      |
    {V_str}
        P ricatti              |
    {P_ricatti_str}
        """)

    def update_z(self, landmark):
        if not self.running_rk45:
            self.z_appear = landmark

    def update_linear_velocity(self, linear_velocity):
        if not self.running_rk45:
            self.linearVelocity = linear_velocity

    def update_angular_velocity(self, angular_velocity):
        if not self.running_rk45:
            self.angularVelocity = angular_velocity

    def function_S(self, input):
        '''
        Create a 3x3 skew-symmetric matrix, S(x)y = x x y
        Input: 3x1 array
        Output: 3x3 array
        '''
        # input should be array
        # output array
        flattened_input = input.flatten()
        output = [[0,           -flattened_input[2],    flattened_input[1]],
                [flattened_input[2],  0,              -flattened_input[0]],
                [-flattened_input[1], flattened_input[0],     0]]
        return np.array(output)

    def rodrigues_formula(self, quaternion):
        '''
        Quaternion -> R_tilde_bar
        Input: [w,x,y,z]
        Output R_tile_bar (rotation matrix)
        From page6
        '''
        return np.eye(3) + 2*np.matmul(self.function_S(quaternion[1:]), (quaternion[0]*np.eye(3) + self.function_S(quaternion[1:])))

    def function_A(self, omega):
        '''
        Create the A maxtrix 
        Input = 3x1 array
        Output = 6x6 matrix
        '''
        A11 = self.function_S(-omega)
        A12 = np.zeros((3,3))
        A21 = np.zeros((3,3))
        A22 = self.function_S(-omega)
        return np.vstack((np.hstack((A11, A12)), np.hstack((A21, A22))))

    def function_Pi(self, input):
        '''
        Pi_x := I_3 - xx^T
        Input: array
        Output P_x
        '''
        return np.eye(3) - np.matmul(input, np.transpose(input))

    def function_d(self, input_rot, input_p, input_z, with_noise):
        '''
        Calculate direction d_i(t) := R^T(t)(p(t) - z_i)/|p(t)-z_i|
        Input:
            Rotation matrix R: 3x3 array
            pose p: 3x1 array
            landmark z : 3x1 array
            with_noise : boolean
        Output: 
            direction vector 3x1 array
        '''
        norm = (input_p - input_z)/np.linalg.norm(input_p - input_z)
        dir = np.matmul(np.transpose(input_rot), norm)
        
        if with_noise:
            '''
            calculate noisy d = sign(d_{i,3}) / demon (d_{i,1}/d_{i,3} + n_{n,1}, d_{i,2}/d_{i,3} + n_{i,2}, 1).T
            demon = sqrt((d_{i,1}/d_{i,3} + n_{n,1})^2 + (d_{i,2}/d_{i,3} + n_{i,2})^2 + 1)
            '''
            dir = dir.flatten()
            n_1 = np.random.uniform(-0.005, 0.005, 1)[0]
            n_2 = np.random.uniform(-0.005, 0.005, 1)[0]
            d1_d3 = dir[0]/dir[2] + n_1
            d2_d3 = dir[1]/dir[2] + n_2
            demon = np.sqrt(d1_d3**2 + d2_d3**2 + 1)
            dir = (np.sign(dir[2])/ demon) * np.array([[d1_d3, d2_d3, 1]]).T
        return dir

    def function_C(self, input_R_hat):
        '''
        Create the C maxtrix 
        Input = ...
        Output = num_landmark*3x6 matrix
        '''
        for landmark_idx in range(self.l):
            # S(R_hat.T x z)
            d = np.transpose(self.z[landmark_idx])/ np.linalg.norm(self.z[landmark_idx])
            first = self.function_Pi(d)
            # first = self.function_Pi(self.function_d(input_R, input_p, np.transpose(self.z[landmark_idx])))
            second = self.function_S(np.matmul(np.transpose(input_R_hat), np.transpose(self.z[landmark_idx]))) #TODO
            final = -np.matmul(first, second)
            C_landmark = np.hstack((final, first))
            if landmark_idx == 0:
                output_C = C_landmark
            else:
                output_C = np.vstack((output_C, C_landmark))
        return output_C

    def add_bar(self, input_rot, input_p):
        '''
        Change frame (F -> B)
        '''
        return np.matmul(np.transpose(input_rot), input_p)

    def remove_bar(self, input_rot, input_p_bar):
        '''
        Change frame (B -> F)
        '''
        return np.matmul(np.linalg.inv(np.transpose(input_rot)), input_p_bar)

    def visual_plot(self, figsize = (20, 4), bound_y=True):
        '''
        Plot the result of estimation
        '''
        name = str(self.which_eq) + "_" + str(self.which_omega)
        scipy_solution = pd.read_csv('/home/annika/ITRL/kth_thesis/simulated_result/' + name + '.txt', header=None)
        scipy_solution = np.transpose(scipy_solution.to_numpy().reshape((48,-1)))
        sol_t = scipy_solution[:,-1]
        sol_R = scipy_solution[:,:4]
        sol_est_R = scipy_solution[:,4:8]
        sol_est_p_bar = scipy_solution[:,8:11]
        sol_P_ricatti = scipy_solution[:,11:-1]

        plot_est_p_bar = []
        plot_act_p_bar = []
        plot_est_p = []
        plot_act_p = []
        for t, solution in zip(self.solt, self.soly):
            if self.quaternion:
                own_sol_R = self.rodrigues_formula(solution[:4])
                own_sol_est_R = self.rodrigues_formula(solution[4:8])
                own_sol_est_p_bar = solution[8:11]
            else:
                own_sol_R = solution[:9]
                own_sol_est_R = solution[9:18]
                own_sol_est_p_bar = solution[18:21]
            p = np.array([2.5+2.5*np.cos(0.4*t), 2.5*np.sin(0.4*t), 10])
            p_bar_temp = np.matmul(np.transpose(np.array(own_sol_R).reshape((3,3))), p)
            p_temp = np.matmul(np.array(own_sol_est_R).reshape((3,3)), own_sol_est_p_bar)

            plot_est_p_bar.append(own_sol_est_p_bar.tolist())
            plot_act_p_bar.append(p_bar_temp.tolist())
            plot_est_p.append(p_temp.tolist())
            plot_act_p.append(p.tolist())

        scipy_plot_est_p = []
        scipy_plot_act_p = []

        for t, rotation, p_bar_hat in zip(sol_t, sol_R, sol_est_p_bar):
            p = np.array([2.5+2.5*np.cos(0.4*t), 2.5*np.sin(0.4*t), 10])
            rotation = self.rodrigues_formula(rotation)
            p_temp = np.matmul(np.transpose(np.array(rotation).reshape((3,3))), p)
            scipy_plot_est_p.append(p_bar_hat.tolist())
            scipy_plot_act_p.append(p_temp.tolist())
    ###############################################################################################################################
    ###############################################################################################################################
        plot_err_lambda_bar = []
        plot_act_lambda_bar = []
        plot_est_lambda_bar = []
        for idx, (t, solution) in enumerate(zip(self.solt, self.soly)):
            if self.quaternion:
                own_sol_R = self.rodrigues_formula(solution[:4])
                own_sol_est_R = self.rodrigues_formula(solution[4:8])
            else:
                own_sol_R = solution[:9]
                own_sol_est_R = solution[9:18]
            err = np.matmul(np.linalg.inv(np.array(own_sol_est_R).reshape((3,3))), np.array(own_sol_R).reshape((3,3)))
            est = ScipyRot.from_matrix(np.array(own_sol_est_R).reshape((3,3)))
            act = ScipyRot.from_matrix(np.array(own_sol_R).reshape((3,3)))
            err = ScipyRot.from_matrix(err)
            err = err.as_quat().tolist()
            est = est.as_quat().tolist()
            act = act.as_quat().tolist()
            plot_err_lambda_bar.append(err)
            plot_est_lambda_bar.append(est)
            plot_act_lambda_bar.append(act)
        
        scipy_plot_err_lambda_bar = []
        scipy_plot_act_lambda_bar = []
        scipy_plot_est_lambda_bar = []
        for idx, (t, est_rotation, rotation) in enumerate(zip(sol_t, sol_est_R, sol_R)):
            est_rotation = self.rodrigues_formula(est_rotation)
            rotation = self.rodrigues_formula(rotation)
            try:
                err = np.matmul(np.linalg.inv(np.array(est_rotation).reshape((3,3))), np.array(rotation).reshape((3,3)))
            except:
                err = np.matmul(np.linalg.pinv(np.array(est_rotation).reshape((3,3))), np.array(rotation).reshape((3,3)))
            est = ScipyRot.from_matrix(np.array(est_rotation).reshape((3,3)))
            act = ScipyRot.from_matrix(np.array(rotation).reshape((3,3)))
            err = ScipyRot.from_matrix(err)
            err = err.as_quat().tolist()
            est = est.as_quat().tolist()
            act = act.as_quat().tolist()
            scipy_plot_err_lambda_bar.append(err)
            scipy_plot_est_lambda_bar.append(est)
            scipy_plot_act_lambda_bar.append(act)
    ###############################################################################################################################
    ###############################################################################################################################
        figure, ax = plt.subplots(4,2, figsize=figsize)
        try:
            ax[0,0].plot(self.solt, np.array(plot_act_p_bar)-np.array(plot_est_p_bar), label=["ox", "oy", "oz"], marker='o', markersize=0.2)
            ax[0,0].plot(sol_t, np.array(scipy_plot_act_p)-scipy_plot_est_p, label=["sx", "sy", "sz"], linestyle='dotted', marker='o', markersize=0.2)
            ax[0,0].legend(loc="upper right")
            ax[0,0].set_xlabel("pose error in B frame")
            ax[0,0].grid()
            ax[0,0].set_xlim(0)
            if bound_y:
                ax[0,0].set_ylim(-0.5, 0.5)
            ax[0,0].minorticks_on()

            ax[0,1].plot(self.solt, np.array(plot_act_p)-np.array(plot_est_p), label=["x", "y", "z"], marker='o', markersize=0.2)
            ax[0,1].legend(loc="upper right")
            ax[0,1].set_xlabel("Pose error in F frame ")
            ax[0,1].grid()
            ax[0,1].set_xlim(0)
            ax[0,1].set_ylim(-0.5, 0.5)
            ax[0,1].minorticks_on()

            ax[1,0].plot(self.solt, np.array(plot_act_p), label=["actx", "acty", "actz"], marker='o', markersize=0.2)
            ax[1,0].legend(loc="upper right")
            ax[1,0].set_xlabel("Actual pose in F frame")
            ax[1,0].grid()
            ax[1,0].set_xlim(0)
            ax[1,0].set_ylim(-4, 11)
            ax[1,0].minorticks_on()

            ax[1,1].plot(self.solt, np.array(plot_est_p), label=["estx", "esty", "estz"], marker='o', markersize=0.2)
            ax[1,1].legend(loc="upper right")
            ax[1,1].set_xlabel("Estimated pose in F frame")
            ax[1,1].grid()
            ax[1,1].set_xlim(0)
            ax[1,1].set_ylim(-4, 11)
            ax[1,1].minorticks_on()

            ax[2,0].plot(self.solt, np.array(plot_err_lambda_bar)[:,0:3], label=["x", "y", "z"], marker='o', markersize=0.2)
            ax[2,0].plot(sol_t, np.array(scipy_plot_err_lambda_bar)[:,0:3], label=["sx", "sy", "sz"], linestyle='dotted', marker='o', markersize=0.2)
            ax[2,0].legend(loc="upper right")
            ax[2,0].set_xlabel("Orientation error in B frame")
            ax[2,0].grid()
            ax[2,0].set_xlim(0)
            ax[2,0].set_ylim(-0.1, 0.1)
            ax[2,0].minorticks_on()

            ax[3,0].scatter(self.solt, self.solnumlandmark, marker='o', s=1)
            ax[3,0].set_xlabel("number of landmark")
            ax[3,0].grid()
            ax[3,0].set_xlim(0)
            ax[3,0].minorticks_on()

            dttemp = []
            for idxtemp, ttemp in enumerate(self.solt):
                if idxtemp < len(self.solt)-1:
                    dttemp.append(self.solt[idxtemp+1] - self.solt[idxtemp])
            ax[3,1].plot(self.solt[:-1], dttemp, color="blue", alpha=0.8, label="dt between each timestep")
            ax[3,1].plot(self.solt, self.caltime, color="orange", alpha=0.8, label="cal time")
            ax[3,1].legend(loc="upper right")
            ax[3,1].set_xlabel("number of landmark")
            ax[3,1].grid()
            ax[3,1].set_xlim(0)
            ax[3,1].minorticks_on()
            param = {'use_adaptive': self.use_adaptive,
                     'quaternion': self.quaternion,
                     'time': self.time,
                     'stepsize': self.stepsize,
                     'tol': self.tol,
                     'noise': self.noise,
                     'which_eq': self.which_eq,
                     'which_omega': self.which_omega,
                     'with_image_hz_sim': self.with_image_hz_sim,
                     'image_hz': self.image_hz,
                     'randomize_image_input': self.randomize_image_input,
                     'number of landmark': self.l,
                     'k': self.k,
                     'q': self.q[0],
                     'v': self.v,
                     'P': self.p_ricatti}

            param_text = "\n".join([f"{k} = {v}" for k, v in param.items()])
            ax[2,1].xaxis.set_visible(False)
            ax[2,1].yaxis.set_visible(False)
            ax[2,1].set_frame_on(False)
            ax[2,1].text(0.5, 0.5, param_text, fontsize=10, va='center', ha='left',
              bbox=dict(boxstyle="round", alpha=0.1))
            plt.tight_layout(pad=2.0)
            plt.show()
        except Exception as e:
            print(e)

        return figure, ax
        ###############################################################################################################################
        ###############################################################################################################################

    def observer_equations(self, input_p_bar_hat, input_R, input_R_hat, input_p, input_P):
        # self.observer_equations(input_p_bar_hat, input_R, input_R_hat, input_p, input_P)
        if self.which_eq == 0:
            # omega
            first_upper = self.angularVelocity #TODO: make sure is not dependent on frame, tho it shouldnt  
            
            # -S(omega)p_bat_hat + v_bar
            #TODO: remove function_S
            first_lower = -np.matmul(self.function_S(self.angularVelocity), input_p_bar_hat) + self.linearVelocity
            first_part = np.vstack((first_upper, first_lower))

            # omega_hat second part upper
            if not self.z.any() == None:
                final = np.transpose(np.array([[0, 0, 0]], dtype=np.float64))
                final2 = np.transpose(np.array([[0, 0, 0]], dtype=np.float64))
                for landmark_idx in range(self.l):
                    #R_hat.T z
                    first = np.matmul(np.transpose(input_R_hat), np.transpose(self.z[landmark_idx]))
                    #Pi_d
                    d = np.transpose(self.z[landmark_idx])/ np.linalg.norm(self.z[landmark_idx])
                    Pi_d = self.function_Pi(d) #TODO
                    #(p_bar_hat - R_hat.T x z)
                    second = input_p_bar_hat - np.matmul(np.transpose(input_R_hat), np.transpose(self.z[landmark_idx]))
                    # q*
                    final += self.q[landmark_idx]*np.matmul(np.transpose(np.cross(np.transpose(first), Pi_d)), second)

                    # omega_hat second part lower
                    #q*Pi_d
                    #(p_bar_hat - R_hat.T x z)
                    # second = input_p_bar_hat - np.matmul(np.transpose(input_R_hat), np.transpose(input_z[landmark_idx]))
                    final2 += self.q[landmark_idx]*np.matmul(Pi_d, second)

                second_part = np.vstack((final, final2))
                #kP[]
                #full second part 
                second_part = self.k*np.matmul(input_P, second_part)

                # Final
                output_omega_hat_p_bar_hat_dot = first_part - second_part
            else:
                output_omega_hat_p_bar_hat_dot = first_part
            
        elif self.which_eq == 1:
            print("NO EQUATION 1")

        elif self.which_eq == 2:
            ### First part ###
            # omega hat
            first_upper = self.angularVelocity

            # -S(w)p_bar_hat + v_bar
            first_lower = -np.matmul(self.function_S(self.angularVelocity), input_p_bar_hat) + self.add_bar(input_R, self.linearVelocity)
            # first part final
            first_part = np.vstack((first_upper, first_lower))

            if not self.z.any() == None:
                ### Second part ###
                # omega hat
                final = np.transpose(np.array([[0, 0, 0]], dtype=np.float64))
                final2 = np.transpose(np.array([[0, 0, 0]], dtype=np.float64))
                for landmark_idx in range(self.l):
                    d_bar_hat = (input_p_bar_hat - np.matmul(np.transpose(input_R_hat), np.transpose(self.z[landmark_idx])))/ np.linalg.norm(input_p_bar_hat - np.matmul(np.transpose(input_R_hat), np.transpose(self.z[landmark_idx])))
                    Pi_d_bar_hat = self.function_Pi(d_bar_hat)
                    # q S(R_hat.T z) Pi_d_bar_hat 
                    first = self.q[landmark_idx]*np.matmul(self.function_S(np.matmul(np.transpose(input_R_hat), np.transpose(self.z[landmark_idx]))), Pi_d_bar_hat)
                    # |p_bar_hat - R_hat.T z| di
                    second = np.linalg.norm(input_p_bar_hat - np.matmul(np.transpose(input_R_hat), np.transpose(self.z[landmark_idx])))*self.function_d(input_R, input_p, np.transpose(self.z[landmark_idx]))
                    final += np.matmul(first, second)

                    # q Pi_d_bar_hat
                    first = self.q[landmark_idx]*Pi_d_bar_hat
                    # |p_bar_hat - R_hat.T z| di
                    #second 
                    final2 += np.matmul(first, second)

                second_part = np.vstack((final, final2))
                second_part = self.k*np.matmul(input_P, second_part)

                output_omega_hat_p_bar_hat_dot = first_part + second_part
            else:
                output_omega_hat_p_bar_hat_dot = first_part

        return output_omega_hat_p_bar_hat_dot

    def dynamics(self, t, y):
        # pose
        input_p = np.transpose(np.array([[2.5+2.5*np.cos(0.4*t), 2.5*np.sin(0.4*t), 10]]))

        ####################################
        ########### Measurements ###########
        # Linear Velocity
        # input_v = self.linearVelocity
        # Angular Velocity
        # input_omega = self.angularVelocity
        ########### Measurements ###########
        ####################################

        ####################################
        ############ Quaternion ############
        qua_flat, qua_hat_flat, p_bar_hat_flat, input_P_flat = np.split(y, [4, 8, 11])
        qua_flat = qua_flat/np.linalg.norm(qua_flat)
        qua_hat_flat = qua_hat_flat/np.linalg.norm(qua_hat_flat)
        input_R = self.rodrigues_formula(qua_flat)
        input_R_hat = self.rodrigues_formula(qua_hat_flat)
        ############ Quaternion ############
        ####################################

        # (self.k, z, self.q, self.Q, self.V, self.l)

        input_p_bar_hat = p_bar_hat_flat.reshape((3,1))
        input_P = input_P_flat.reshape((6,6))

        input_A = self.function_A(self.angularVelocity)
        if not self.z.any() == None:
            input_C = self.function_C(input_R_hat)
        ####################################

        ####################################
        ############# Observer #############
        output_omega_hat_p_bar_hat_dot = self.observer_equations(input_p_bar_hat, input_R, input_R_hat, input_p, input_P)
        ############# Observer #############
        ####################################
        
        if not input_z.any() == None:
            output_P_dot = np.matmul(input_A, input_P) + np.matmul(input_P, np.transpose(input_A)) - np.matmul(input_P, np.matmul(np.transpose(input_C), np.matmul(input_Q, np.matmul(input_C, input_P)))) + input_V
        else:
            output_P_dot = np.matmul(input_A, input_P) + np.matmul(input_P, np.transpose(input_A)) + input_V

        p_bar_hat_dot = output_omega_hat_p_bar_hat_dot[3:]

        ####################################
        ############ Quaternion ############
        omega_hat = output_omega_hat_p_bar_hat_dot[0:3].flatten()
        omega_hat_4x4 = np.array([[0, -omega_hat[0], -omega_hat[1], -omega_hat[2]],
                                [omega_hat[0], 0, omega_hat[2], -omega_hat[1]],
                                [omega_hat[1], -omega_hat[2], 0, omega_hat[0]],
                                [omega_hat[2], omega_hat[1], -omega_hat[0], 0]])

        output_qua_hat_flat = 0.5*np.matmul(omega_hat_4x4, qua_hat_flat)
        input_omega = input_omega.flatten()
        omega_4x4 = np.array([[0, -input_omega[0], -input_omega[1], -input_omega[2]],
                                [input_omega[0], 0, input_omega[2], -input_omega[1]],
                                [input_omega[1], -input_omega[2], 0, input_omega[0]],
                                [input_omega[2], input_omega[1], -input_omega[0], 0]])
        output_qua_flat = 0.5*np.matmul(omega_4x4, qua_flat)
        return np.concatenate((output_qua_flat, output_qua_hat_flat, p_bar_hat_dot.flatten(), output_P_dot.flatten()))
        ########### Quaternion ############
        ####################################

    def rk45_step(self):
        self.running_rk45 = True
        a2, a3, a4, a5, a6 = 1/5, 3/10, 4/5, 8/9, 1
        b21 = 1/5
        b31, b32 = 3/40, 9/40
        b41, b42, b43 = 44/45, -56/15, 32/9
        b51, b52, b53, b54 = 19372/6561, -25360/2187, 64448/6561, -212/729
        b61, b62, b63, b64, b65 = 9017/3168, -355/33, 46732/5247, 49/176, -5103/18656
        c1, c2, c3, c4, c5, c6 = 35/384, 0, 500/1113, 125/192, -2187/6784, 11/84

        c1_4, c3_4, c4_4, c5_4, c6_4 = 5179/57600, 7571/16695, 393/640, -92097/339200, 187/2100
        # Runge-Kutta stages
        k1 = self.dynamics(self.current_time, self.soly)
        k2 = self.dynamics(self.current_time + a2*self.dt, self.soly + self.dt*b21*k1)
        k3 = self.dynamics(self.current_time + a3*self.dt, self.soly + self.dt*(b31*k1 + b32*k2))
        k4 = self.dynamics(self.current_time + a4*self.dt, self.soly + self.dt*(b41*k1 + b42*k2 + b43*k3))
        k5 = self.dynamics(self.current_time + a5*self.dt, self.soly + self.dt*(b51*k1 + b52*k2 + b53*k3 + b54*k4))
        k6 = self.dynamics(self.current_time + a6*self.dt, self.soly + self.dt*(b61*k1 + b62*k2 + b63*k3 + b64*k4 + b65*k5))
        self.running_rk45 = False
        # Update step
        y_next = self.soly + dt*(c1*k1 + c2*k2 + c3*k3 + c4*k4 + c5*k5 + c6*k6)
        y_next_4 = self.soly + dt * (c1_4*k1 + c3_4*k3 + c4_4*k4 + c5_4*k5 + c6_4*k6)

        error = np.abs(y_next - y_next_4)
        error_norm = np.linalg.norm(error)
        safety_factor = 0.9
        min_scale_factor = 0.2
        max_scale_factor = 40.0
        if error_norm <= self.tol:
            success = True
            t = t+dt
            dt = dt * min(max_scale_factor, max(min_scale_factor, safety_factor * (self.tol / error_norm)**0.25))
        else:
            success = False
            dt = dt * max(min_scale_factor, safety_factor * (self.tol / error_norm)**0.25)

        return y_next, t, dt, success

    def step_simulation(self):
        ### Run solver
        ######################################################
        ####################### Solver #######################
        success = False
        while not success:
            self.l = len(self.z)
            z = self.z
            self.soly = np.concatenate((self.Lambda_bar_0.flatten(), self.p_bar_hat.flatten(), self.P_ricatti.flatten()))
            # args = (self.k, z, self.q, self.Q, self.V, self.l)
            # y_next, next_time, new_dt, success = self.rk45_step(self.current_time, self.soly[-1], self.dt, args, self.tol, self.use_adaptive)
            
            y_next, next_time, new_dt, success = self.rk45_step()
            if success:
                self.soly = y_next
                self.solt = self.current_time
                # self.solimage.append([self.current_time, self.show_measurement])
                # self.solnumlandmark.append(self.l)
                # end_time = systemtime.time()
                # self.caltime.append(end_time - start_time)
                # start_time = end_time
                self.current_time = next_time
            self.dt = new_dt
            ####################### Solver #######################
            ######################################################
        return (self.solt[-1], self.dt, self.soly[-1])