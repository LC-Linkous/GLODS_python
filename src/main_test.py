#! /usr/bin/python3

##--------------------------------------------------------------------\
#   glods_python
#   './glods_python/src/main_test.py'
#   Test function/example for using the GLODS optimizer. Matches the
#       format of the other optimizers in the AntennaCAT suite.
#
#   Author(s): Lauren Linkous
#   Last update: June 11, 2026
##--------------------------------------------------------------------\

import sys
import pandas as pd
import numpy as np

try:  # for outside func calls, program calls
    sys.path.insert(0, './glods_python/src/')
    from glods import glods
except:  # for local, unit testing
    from glods import glods

# OBJECTIVE FUNCTION SELECTION
#import one_dim_x_test.configs_F as func_configs     # single objective, 1D input
import himmelblau.configs_F as func_configs          # single objective, 2D input
#import lundquist_3_var.configs_F as func_configs    # multi objective function


if __name__ == "__main__":
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
    parent = None
    suppress_output = True
    allow_update = True

    best_eval = 1

    # instantiation of GLODS optimizer
    # parameters intentionally match the multi_glods constructor
    opt_params = {'BP': [BP],
                  'GP': [GP],
                  'SF': [SF],
                  'R_TOL': [R_TOL]}
    opt_df = pd.DataFrame(opt_params)
    myGlods = glods(LB, UB, TARGETS, E_TOL, MAXIT,
                    func_F, constr_F,
                    opt_df,
                    parent=parent,
                    evaluate_threshold=evaluate_threshold,
                    obj_threshold=THRESHOLD)

    last_iter = 0
    while not myGlods.complete():
        # step through optimizer processing
        # consumes the previous evaluation and advances the point list
        myGlods.step(suppress_output)

        # call the objective function, control
        # when it is allowed to update and return
        # control to the optimizer
        noErr = myGlods.call_objective(allow_update)
        if noErr == True:
            iter, eval = myGlods.get_convergence_data()
            if (eval < best_eval) and (eval != 0):
                best_eval = eval
            if iter > last_iter:
                last_iter = iter
                if suppress_output:
                    if iter % 100 == 0:
                        print("************************************************")
                        print("Objective Function Iterations: " + str(iter))
                        print("Best Eval: " + str(best_eval))
        else:
            print("ERROR: in executing objective function call.")

    print("************************************************")
    print("Total Objective Function Iterations: " + str(iter))
    print("Best Eval: " + str(best_eval))
    print("Optimized Solution")
    print(myGlods.get_optimized_soln())
    print("Optimized Outputs")
    print(myGlods.get_optimized_outs())
    print("Active points (local + global minimizer candidates)")
    pts, vals = myGlods.get_active_points()
    for p, v in zip(pts, vals):
        print("   x = " + str(np.round(p, 4)) + "   |F| = " + str(np.linalg.norm(v)))
