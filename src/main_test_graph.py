#! /usr/bin/python3

##--------------------------------------------------------------------\
#   glods_python
#   './glods_python/src/main_test_graph.py'
#   Test function/example for using the GLODS optimizer.
#       Format updates are
#       for integration in the AntennaCAT GUI.
#       This version builds from 'main_test_details.py' to include a
#       matplotlib plot of the evaluated search locations and the
#       active (locally nondominated) points
#
#   Author(s): Lauren Linkous, Jonathan Lundquist
#   Last update: June 11, 2026
##--------------------------------------------------------------------\


import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from glods import glods

# OBJECTIVE FUNCTION SELECTION
#import one_dim_x_test.configs_F as func_configs     # single objective, 1D input
import himmelblau.configs_F as func_configs         # single objective, 2D input
#import lundquist_3_var.configs_F as func_configs     # multi objective function


class TestGraph():
    def __init__(self):

        self.ctr = 0

        # Constant variables
        R_TOL = 10 ** -6      # Convergence Tolerance (radius/step-size based,
                              # not target based)
        E_TOL = 10 ** -4      # Convergence Error Tolerance
        MAXIT = 3000          # Maximum allowed objective function calls

        # Objective function dependent variables
        LB = func_configs.LB              # Lower boundaries, [[-5, -5]]
        UB = func_configs.UB              # Upper boundaries, [[5, 5]]
        TARGETS = func_configs.TARGETS    # Target values for output

        # threshold is same dims as TARGETS
        # 0 = use target value as actual target. value should EQUAL target
        # 1 = use as threshold. value should be LESS THAN OR EQUAL to target
        # 2 = use as threshold. value should be GREATER THAN OR EQUAL to target
        # DEFAULT THRESHOLD
        THRESHOLD = np.zeros_like(TARGETS)
        evaluate_threshold = False

        # Objective function dependent variables
        func_F = func_configs.OBJECTIVE_FUNC   # objective function
        constr_F = func_configs.CONSTR_FUNC    # constraint function

        # optimizer specific vars
        BP = 0.5              # Beta Par. step size contraction
        GP = 1                # Gamma Par. step size expansion
        SF = 2                # Search Frequency

        # optimizer setting values
        self.best_eval = 1            # Starting eval value

        parent = self                 # Optional parent class for optimizer
                                        # (Used for passing debug messages or
                                        # other information that will appear
                                        # in GUI panels)

        self.suppress_output = True   # Suppress the console output of the optimizer

        self.allow_update = True      # Allow objective call to update state

        # instantiation of GLODS optimizer
        # parameters intentionally match the multi_glods constructor
        opt_params = {'BP': [BP],
                      'GP': [GP],
                      'SF': [SF],
                      'R_TOL': [R_TOL]}
        opt_df = pd.DataFrame(opt_params)
        self.myGlods = glods(LB, UB, TARGETS, E_TOL, MAXIT,
                             func_F, constr_F,
                             opt_df,
                             parent=parent,
                             evaluate_threshold=evaluate_threshold,
                             obj_threshold=THRESHOLD)


        # Matplotlib setup
        self.targets = TARGETS
        self.fig = plt.figure(figsize=(10, 5))#(figsize=(14, 7))
        # position
        self.ax1 = self.fig.add_subplot(121, projection='3d')
        self.ax1.set_title("Search Locations, Iteration: " + str(self.ctr))
        self.ax1.set_xlabel('x_1')
        self.ax1.set_ylabel('x_2')
        self.ax1.set_zlabel('x_3')
        self.scatter1 = None
        # fitness
        self.ax2 = self.fig.add_subplot(122, projection='3d')
        self.ax2.set_title("Fitness Relation to Target")
        self.ax2.set_xlabel('x_1')
        self.ax2.set_ylabel('x_2')
        self.ax2.set_zlabel('x_3')
        self.scatter2 = None

    def debug_message_printout(self, txt):
        if txt is None:
            return
        # sets the string as it gets it
        curTime = time.strftime("%H:%M:%S", time.localtime())
        msg = "[" + str(curTime) +"] " + str(txt)
        print(msg)


    def update_plot(self, x_coords, y_coords, targets, active_coords=None,
                    showTarget=True, clearAx=True):

        # check if any points. first call might not have anythign set yet.
        if len(x_coords) < 1:
            return


        if clearAx == True:
            self.ax1.clear() #use this to git rid of the 'ant tunnel' trails
            self.ax2.clear()

        # SEARCH LOCATION PLOT
        # all evaluated points are plotted in blue. the ACTIVE points
        # (GLODS' local + global minimizer candidates) are overlaid in red
        if np.shape(x_coords)[1]==1: # 1 dim function
            x_plot_coords = np.array(x_coords[:,0])*0.0
            self.ax1.set_title("Search Locations, Iteration: " + str(self.ctr))
            self.ax1.set_xlabel("$x_1$")
            self.ax1.set_ylabel("filler coords")
            self.ax1.set_zlabel("filler coords")
            self.scatter = self.ax1.scatter(x_coords, x_plot_coords, edgecolors='b')
            if (active_coords is not None) and (len(active_coords) > 0):
                a_plot_coords = np.array(active_coords[:,0])*0.0
                self.scatter = self.ax1.scatter(active_coords, a_plot_coords, color='r')

        elif np.shape(x_coords)[1] == 2: #2-dim func
            self.ax1.set_title("Search Locations, Iteration: " + str(self.ctr))
            self.ax1.set_xlabel("$x_1$")
            self.ax1.set_ylabel("$x_2$")
            self.ax1.set_zlabel("filler coords")
            self.scatter = self.ax1.scatter(x_coords[:,0], x_coords[:,1], edgecolors='b')
            if (active_coords is not None) and (len(active_coords) > 0):
                self.scatter = self.ax1.scatter(active_coords[:,0], active_coords[:,1], color='r')

        elif np.shape(x_coords)[1] == 3: #3-dim func
            self.ax1.set_title("Search Locations, Iteration: " + str(self.ctr))
            self.ax1.set_xlabel("$x_1$")
            self.ax1.set_ylabel("$x_2$")
            self.ax1.set_zlabel("$x_3$")
            self.scatter = self.ax1.scatter(x_coords[:,0], x_coords[:,1], x_coords[:,2], edgecolors='b')
            if (active_coords is not None) and (len(active_coords) > 0):
                self.scatter = self.ax1.scatter(active_coords[:,0], active_coords[:,1], active_coords[:,2], color='r')


        # FITNESS PLOT
        if np.shape(y_coords)[1] == 1: #1-dim obj func
            y_plot_filler = np.array(y_coords[:,0])*0.0
            self.ax2.set_title("Global Best Fitness Relation to Target")
            self.ax2.set_xlabel("$F_{1}(x_1,x_2)$")
            self.ax2.set_ylabel("filler coords")
            self.ax2.set_zlabel("filler coords")
            self.scatter = self.ax2.scatter(y_coords, y_plot_filler,  marker='o', s=40, facecolor="none", edgecolors="k")

        elif np.shape(y_coords)[1] == 2: #2-dim obj func
            self.ax2.set_title("Global Best Fitness Relation to Target")
            self.ax2.set_xlabel("$F_{1}(x_1,x_2)$")
            self.ax2.set_ylabel("$F_{2}(x_1,x_2)$")
            self.ax2.set_zlabel("filler coords")
            self.scatter = self.ax2.scatter(y_coords[:,0], y_coords[:,1], marker='o', s=40, facecolor="none", edgecolors="k")

        elif np.shape(y_coords)[1] == 3: #3-dim obj fun
            self.ax2.set_title("Global Best Fitness Relation to Target")
            self.ax2.set_xlabel("$F_{1}(x_1,x_2)$")
            self.ax2.set_ylabel("$F_{2}(x_1,x_2)$")
            self.ax2.set_zlabel("$F_{3}(x_1,x_2)$")
            self.scatter = self.ax2.scatter(y_coords[:,0], y_coords[:,1], y_coords[:,2], marker='o', s=40, facecolor="none", edgecolors="k")


        if showTarget == True: # plot the target point
            if len(targets) == 1:
                self.scatter = self.ax2.scatter(targets[0], 0, marker='*', edgecolors='r')
            if len(targets) == 2:
                self.scatter = self.ax2.scatter(targets[0], targets[1], marker='*', edgecolors='r')
            elif len(targets) == 3:
                self.scatter = self.ax2.scatter(targets[0], targets[1], targets[2], marker='*', edgecolors='r')


        plt.pause(0.0001)  # Pause to update the plot
        if self.ctr == 0:
            time.sleep(2)

        self.ctr = self.ctr + 1

    def run(self):
        time.sleep(2)

        last_iter = 0
        while not self.myGlods.complete():

            # step through optimizer processing
            # consumes the previous evaluation and advances the point list
            self.myGlods.step(self.suppress_output)

            # call the objective function, control
            # when it is allowed to update and return
            # control to the optimizer
            noErr = self.myGlods.call_objective(self.allow_update)
            if noErr == True:
                iter, eval = self.myGlods.get_convergence_data()
                if (eval < self.best_eval) and (eval != 0):
                    self.best_eval = eval
                if iter > last_iter:
                    last_iter = iter
                    if self.suppress_output:
                        if iter % 100 == 0:
                            print("Iteration")
                            print(iter)
                            print("Best Eval")
                            print(self.best_eval)
            else:
                print("ERROR: in executing objective function call.")

            m_coords = self.myGlods.get_search_locations()  # every point evaluated so far
            a_coords, _ = self.myGlods.get_active_points()  # local + global minimizer candidates
            f_coords = self.myGlods.F_Gb # global best of set
            self.update_plot(m_coords, f_coords, self.targets, active_coords=a_coords,
                             showTarget=True, clearAx=True) #update matplot

        print("Optimized Solution")
        print(self.myGlods.get_optimized_soln())
        print("Optimized Outputs")
        print(self.myGlods.get_optimized_outs())
        print("Active points (local + global minimizer candidates)")
        pts, vals = self.myGlods.get_active_points()
        for p, v in zip(pts, vals):
            print("   x = " + str(np.round(p, 4)) + "   |F| = " + str(np.linalg.norm(v)))

        time.sleep(15) # so that the plot does not dissapear immediately

if __name__ == "__main__":
    gl = TestGraph()
    gl.run()