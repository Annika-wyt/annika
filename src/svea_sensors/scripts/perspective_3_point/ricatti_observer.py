import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import time as systemtime
class riccati_observer():
    def __init__(self, **kwargs):
        ######################################################
        ##################### Parameters #####################
        self.z_estFrame = np.array([])
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
        self.V = np.diag(np.hstack([np.diag(self.v[i]*np.eye(3)) for i in range(len(self.v))]))
        self.Q = np.diag(np.hstack([np.diag(self.q*np.eye(3)) for i in range(self.l)])) if self.l != 0 else np.array([])
        
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

        self.soly = np.concatenate((self.Lambda_bar_0.flatten(), self.p_bar_hat.flatten(), self.P_ricatti.flatten()))

        ################### initialization ###################
        ######################################################

        # self.print_init()

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
            self.z = landmark
            self.l = len(self.z)
            self.Q = np.diag(np.hstack([np.diag(self.q*np.eye(3)) for i in range(self.l)])) if self.l != 0 else np.array([])

    def update_z_estFrame(self, landmarkGroundtruth):
        if not self.running_rk45:
            self.z_estFrame = landmarkGroundtruth

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
        output = [[0,           -input[2],    input[1]],
                [input[2],  0,              -input[0]],
                [-input[1], input[0],     0]]
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
        return np.eye(3) - np.outer(input, input)

    def function_d(self, input_rot, input_p, input_z):
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
        return dir

    def function_C(self):
        '''
        Create the C maxtrix 
        Input = ...
        Output = num_landmark*3x6 matrix
        '''
        # landmark = np.array([[2.5, 2.5, 1], [5, 0, 1], [0, 0, 1]])
        for landmark_idx in range(self.l):
            d = -np.array(self.z[landmark_idx]/ np.linalg.norm(self.z[landmark_idx]))
            first = self.function_Pi(d)
            # first = self.function_Pi(self.function_d(input_R, input_p, self.z[landmark_idx]))
            
            # S(R_hat.T x z) TODO: different from original
            # second = np.matmul(np.transpose(input_R_hat), np.array(landmark[landmark_idx])) 
            second = np.array(self.z_estFrame[landmark_idx]) #self.function_S(np.matmul(np.transpose(input_R_hat), self.z[landmark_idx])) #TODO
            final = -np.cross(first, second)
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
    
    def observer_equations(self, input_p_bar_hat, input_R_hat, input_P):
        # self.observer_equations(input_p_bar_hat, input_R, input_R_hat, input_p, input_P)
        # landmark = np.array([[2.5, 2.5, 1], [5, 0, 1], [0, 0, 1]])
        if self.which_eq == 0:
            # omega
            first_upper = self.angularVelocity #TODO: huh
            
            # -S(omega)p_bat_hat + v_bar
            first_lower = -np.cross(self.angularVelocity, input_p_bar_hat) + self.linearVelocity
            first_part = np.hstack((first_upper, first_lower))
            # omega_hat second part upper
            if len(self.z) != 0:
                final = np.array([0, 0, 0], dtype=np.float64)
                final2 = np.array([0, 0, 0], dtype=np.float64)

                for landmark_idx in range(self.l):
                    #R_hat.T z #TODO: huh??? different from original
                    first = np.array(self.z_estFrame[landmark_idx])
                    # first = np.matmul(np.transpose(input_R_hat), np.array(landmark[landmark_idx]))
                    #Pi_d
                    d = -np.array(self.z[landmark_idx]/ np.linalg.norm(self.z[landmark_idx]))
                    Pi_d = self.function_Pi(d)
                    #(p_bar_hat - R_hat.T x z)
                    second = -np.array(self.z_estFrame[landmark_idx])
                    # q*
                    final += self.q*np.matmul(np.transpose(np.cross(first, Pi_d)), second)
                    # omega_hat second part lower
                    #q*Pi_d
                    #(p_bar_hat - R_hat.T x z)
                    # second = input_p_bar_hat - np.matmul(np.transpose(input_R_hat), np.transpose(input_z[landmark_idx]))
                    final2 += self.q*np.matmul(Pi_d, second)

                second_part = np.hstack((final, final2))
                #kP[]
                #full second part 
                second_part = self.k*np.matmul(input_P, second_part)

                # Final
                print("second_part", second_part)
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
            first_lower = -np.cross(self.angularVelocity, input_p_bar_hat) + self.linearVelocity
            # first part final
            first_part = np.hstack((first_upper, first_lower))

            if len(self.z) != 0:
                ### Second part ###
                # omega hat
                final = np.transpose(np.array([[0, 0, 0]], dtype=np.float64))
                final2 = np.transpose(np.array([[0, 0, 0]], dtype=np.float64))
                for landmark_idx in range(self.l):
                    d_bar_hat = (input_p_bar_hat - np.matmul(np.transpose(input_R_hat), self.z[landmark_idx]))/ np.linalg.norm(input_p_bar_hat - np.matmul(np.transpose(input_R_hat), np.transpose(self.z[landmark_idx])))
                    Pi_d_bar_hat = self.function_Pi(d_bar_hat)
                    # q S(R_hat.T z) Pi_d_bar_hat 
                    first = self.q*np.matmul(self.function_S(np.matmul(np.transpose(input_R_hat), np.transpose(self.z[landmark_idx]))), Pi_d_bar_hat)
                    # |p_bar_hat - R_hat.T z| di
                    second = np.linalg.norm(input_p_bar_hat - np.matmul(np.transpose(input_R_hat), np.transpose(self.z[landmark_idx])))*self.function_d(input_R, input_p, np.transpose(self.z[landmark_idx]))
                    final += np.matmul(first, second)

                    # q Pi_d_bar_hat
                    first = self.q*Pi_d_bar_hat
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
        qua_hat_flat, p_bar_hat_flat, input_P_flat = np.split(y, [4, 7])
        qua_hat_flat = qua_hat_flat/np.linalg.norm(qua_hat_flat)
        # input_R = self.rodrigues_formula(qua_flat)
        input_R_hat = self.rodrigues_formula(qua_hat_flat)
        ############ Quaternion ############
        ####################################

        # (self.k, z, self.q, self.Q, self.V, self.l)

        input_p_bar_hat = p_bar_hat_flat
        input_P = input_P_flat.reshape((6,6))

        input_A = self.function_A(self.angularVelocity)
        if len(self.z) != 0:
            input_C = self.function_C()
        ####################################

        ####################################
        ############# Observer #############
        output_omega_hat_p_bar_hat_dot = self.observer_equations(input_p_bar_hat, input_R_hat, input_P)
        ############# Observer #############
        ####################################
        
        if len(self.z) != 0:
            output_P_dot = np.matmul(input_A, input_P) + np.matmul(input_P, np.transpose(input_A)) - np.matmul(input_P, np.matmul(np.transpose(input_C), np.matmul(self.Q, np.matmul(input_C, input_P)))) + self.V
            # print("CP",  np.matmul(input_C, input_P))
            # print("QCP", np.matmul(self.Q, np.matmul(input_C, input_P)))
            # print("CQCP", np.matmul(np.transpose(input_C), np.matmul(self.Q, np.matmul(input_C, input_P))))
            # print("PCQCP", np.matmul(input_P, np.matmul(np.transpose(input_C), np.matmul(self.Q, np.matmul(input_C, input_P)))))
        else:
            output_P_dot = np.matmul(input_A, input_P) + np.matmul(input_P, np.transpose(input_A)) + self.V
        p_bar_hat_dot = output_omega_hat_p_bar_hat_dot[3:]

        ####################################
        ############ Quaternion ############
        omega_hat = output_omega_hat_p_bar_hat_dot[0:3]
        omega_hat_4x4 = np.array([[0, -omega_hat[0], -omega_hat[1], -omega_hat[2]],
                                [omega_hat[0], 0, omega_hat[2], -omega_hat[1]],
                                [omega_hat[1], -omega_hat[2], 0, omega_hat[0]],
                                [omega_hat[2], omega_hat[1], -omega_hat[0], 0]])
        output_qua_hat_flat = 0.5*np.matmul(omega_hat_4x4, qua_hat_flat)

        return np.concatenate((output_qua_hat_flat, p_bar_hat_dot, output_P_dot.flatten()))
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
        y_next = self.soly + self.dt*(c1*k1 + c2*k2 + c3*k3 + c4*k4 + c5*k5 + c6*k6)
        y_next_4 = self.soly + self.dt * (c1_4*k1 + c3_4*k3 + c4_4*k4 + c5_4*k5 + c6_4*k6)

        error = np.abs(y_next - y_next_4)
        error_norm = np.linalg.norm(error)
        safety_factor = 0.9
        min_scale_factor = 0.2
        max_scale_factor = 40.0
        if error_norm <= self.tol:
            success = True
            t = self.current_time+self.dt
            dt = self.dt * min(max_scale_factor, max(min_scale_factor, safety_factor * (self.tol / error_norm)**0.25))
        else:
            success = False
            t = -1
            dt = self.dt * max(min_scale_factor, safety_factor * (self.tol / error_norm)**0.25)

        return y_next, t, dt, success

    def step_simulation(self):
        ### Run solver
        ######################################################
        ####################### Solver #######################
        success = False
        while not success:
            # args = (self.k, z, self.q, self.Q, self.V, self.l)
            # y_next, next_time, new_dt, success = self.rk45_step(self.current_time, self.soly[-1], self.dt, args, self.tol, self.use_adaptive)
            y_next, next_time, new_dt, success = self.rk45_step()
            if success:
                self.soly = y_next
                self.solt = next_time
                # self.solimage.append([self.current_time, self.show_measurement])
                # self.solnumlandmark.append(self.l)
                # end_time = systemtime.time()
                # self.caltime.append(end_time - start_time)
                # start_time = end_time
                self.current_time = next_time
                # print("t", self.solt, "dt", self.dt, "next dt", new_dt )
                # print("soly qua", self.soly[:4]) #w, x ,y ,z
                # print("soly pose", self.soly[4:7])
            self.dt = new_dt
            ####################### Solver #######################
            ######################################################
        return (self.solt, self.dt, self.soly)