# glods_python

Python-based GLODS optimizer compatible with the [AntennaCAT](https://github.com/LC-Linkous/AntennaCalculationAutotuningTool) optimizer suite.  Now featuring AntennaCAT hooks for GUI integration and user input handling.

GLODS (Global and Local Optimization using Direct Search) [1] is the single-objective predecessor of MultiGLODS [2]. Unlike [multi_glods_python](https://github.com/jonathan46000/multi_glods_python), which is a direct translation of the original MATLAB MultiGLODS 0.1, this is a clean-slate implementation built on the AntennaCAT `pso_basic` template with the decoupled `step()`/`call_objective()` structure. The `opt_df` parameters (BP, GP, SF, R_TOL) intentionally match the `multi_glods` constructor, so the two are drop-in siblings. Please see the [GLODS](#glods) and [Implementation Origin](#implementation-origin) sections for more information, and GLODS_ALGORITHM.md in this repository for a full description of the algorithm structure and design decisions.

## Table of Contents
* [GLODS](#glods)
* [Implementation Origin](#implementation-origin)
    * [Relation to multi_glods_python](#relation-to-multi_glods_python)
    * [Streamlined Interface for AntennaCAT Optimizer Modularity](#streamlined-interface-for-antennacat-optimizer-modularity)
    * [Addition of Threshold vs. Target](#addition-of-threshold-vs-target)
* [Requirements](#requirements)
* [Implementation](#implementation)
    * [Initialization](#initialization) 
    * [State Machine-based Structure](#state-machine-based-structure)
    * [Constraint Handling](#constraint-handling)
    * [Boundary Types](#boundary-types)
    * [The Active Point Set](#the-active-point-set)
    * [Objective Function Handling](#objective-function-handling)
      * [Creating a Custom Objective Function](#creating-a-custom-objective-function)
      * [Internal Objective Function Example](#internal-objective-function-example)
    * [Target vs. Threshold Configuration](#target-vs-threshold-configuration)
* [Example Implementations](#example-implementations)
    * [Basic Example](#basic-example)
    * [Detailed Messages](#detailed-messages)
    * [Realtime Graph](#realtime-graph)
* [References](#references)
* [Related Publications and Repositories](#related-publications-and-repositories)
* [Licensing](#licensing)  

## GLODS

The Global and Local Optimization using Direct Search (GLODS) [1] algorithm was created by Dr. Ana Luise Custódio (Nova School of Science and Technology, Lisbon) and J. F. A. Madeira (ISEL and IDMEC-IST, Lisbon). It is a derivative-free optimizer designed to locate ALL the local and global minimizers of a multimodal problem, not just the single best point. It is the single-objective predecessor of their MultiGLODS algorithm in [2].

Some key points of this algorithm are:

* The algorithm alternates between initializing new searches, using a multistart strategy, and exploring promising subregions, resorting to directional direct search. The multistart (search) step is what finds new basins; the directional direct search (poll) step refines the basins found.

* Points sufficiently close to each other are compared and only locally best points remain active; the initialized searches are not all conducted until the end, merging when they start to be close to each other. At the end of the optimization process, the set of all active points identifies the local AND global minimizers found. This is the single-objective specialization of the MultiGLODS rule, with scalar comparison in place of Pareto dominance.

In the following section, the origin of this implementation and some of the design choices are described.

## Implementation Origin

### Relation to multi_glods_python

The [multi_glods_python](https://github.com/jonathan46000/multi_glods_python) translation by [jonathan46000](https://github.com/jonathan46000) preserved the original MATLAB control flow as a re-entrant state machine (`multiglods_ctl.py`). This GLODS implementation is instead a clean-slate class written on the AntennaCAT `pso_basic` template, using the same internal evaluation queue approach as the DIRECT implementation to serialize the batch-natured search/poll iterations. The controller sees the standard one-evaluation-per-pass rhythm, and the GLODS iteration boundary is invisible to it.

Where the original GLODS/MultiGLODS parameters had MATLAB configuration counterparts, this implementation follows the multi_glods_python conventions:

* `suf_decrease`. Hardcoded default coefficient, like the translation.
* The initial point set (`Pini`) is deterministic: n points along the domain diagonal plus the midpoint, matching the translation's choice.
* `poll_complete`. Complete polling is hardcoded (every direction evaluated). NOTE: only the best poll child may activate in the merge; the rest enter the point list as inactive records. This is a deliberate design choice to keep the active set clean (see GLODS_ALGORITHM.md).
* `option_pollcenter`. This implementation uses the best-fitness-first poll center (the more local of the GLODS options), where the multiGLODS translation uses largest-step-first (the more global option). The search step maintains global coverage.
* The search step is an infinite random multistart (like the translation), not the paper's finite initialization list.

The following have default values, but are changeable in the optimizer declaration:

* `tol_stop`. This is included in the wrapper and passed through to the GLODS optimizer as E_TOL.
* `max_fevals`. This is included in the wrapper and passed through to the GLODS optimizer as MAXIT. Iterations are counted as objective function calls.
* `search_freq`. Passed in as SF. A search (multistart) step runs after SF consecutive unsuccessful iterations. The default in this repo's examples is 2.
* `beta_par`. Passed in as BP. Coefficient for step size contraction. The default is 0.5, but this can be changed (with mixed results).
* `gamma_par`. Passed in as GP. Coefficient for step size expansion. The default is 1, but this can be changed (with mixed results).
* `stop_alfa`. Passed in as R_TOL (radius/step-size tolerance) as part of the opt_df. Active points whose step size falls below R_TOL are local-minimizer candidates and are no longer polled; when no pollable point remains, the run is converged by radius.

### Streamlined Interface for AntennaCAT Optimizer Modularity  

Prior to the AntennaCAT v2025.2 rollout for publication, the optimizers included were streamlined so that they would have the same constructor format across classes. 

```python
    # instantiation of GLODS optimizer 
    # constant variables
    opt_params = {'BP': [BP],               # Beta Par. step size contraction
                'GP': [GP],                 # Gamma Par. step size expansion
                'SF': [SF],                 # Search Frequency
                'R_TOL': [R_TOL] }          # Radius/step-size tolerance

    opt_df = pd.DataFrame(opt_params)
    myGlods = glods(LB, UB, TARGETS, E_TOL, MAXIT,
                            func_F, constr_F,
                            opt_df,
                            parent=parent)   

```

All optimizers now have a Pandas dataframe object containing their custom variables for operation. These are unpacked in the initialization of the optimizer class. 


### Addition of Threshold vs. Target

Prior to the AntennaCAT v2025.2 rollout for publication, the ability to use thresholds and targets in order to find an optimized solution has been added. 

Setting `evaluate_threshold` to False, or letting it remain in its default False state, will use the original 'distance to exact target' approach for minimization. Setting `evaluate_threshold` to True and giving it values of 0, 1, or 2 can be used to mix and match exact target and threshold values for each target. 

That is, if the desired solution to a multi objective function is [0,0], then the `evaluate_threshold` can be set to False to use those values as the target. If a desired solution can be described as `less than` or `greater than` a threshold, then `evaluate_threshold` can be set to True and the associated array configured to select 'exact', 'less than' or 'greater than'.



## Requirements

This project requires numpy, pandas, and matplotlib for the full demos. To run the optimizer without visualization, only numpy and pandas are requirements

Use 'pip install -r requirements.txt' to install the following dependencies:

```python
contourpy==1.3.3
cycler==0.12.1
fonttools==4.63.0
kiwisolver==1.5.0
matplotlib==3.10.9
numpy==2.4.6
packaging==26.2
pandas==3.0.3
pillow==12.2.0
pyparsing==3.3.2
python-dateutil==2.9.0.post0
six==1.17.0
tzdata==2026.2

```

Optionally, requirements can be installed manually with:

```python
pip install  matplotlib, numpy, pandas

```
This is an example for if you've had a difficult time with the requirements.txt file. Sometimes libraries are packaged together.

## Implementation

### Initialization 

```python
    # Constant variables
    R_TOL = 10 ** -6    # Convergence Tolerance (radius/step-size based,
                            # not target based)
    E_TOL = 10 ** -4    # Convergence Error Tolerance
    MAXIT = 3000        # Maximum allowed objective function calls

    # Objective function dependent variables
    LB = func_configs.LB              # Lower boundaries, [[-5, -5]]
    UB = func_configs.UB              # Upper boundaries, [[5, 5]]
    IN_VARS = func_configs.IN_VARS    # Number of input variables (x-values)   
    OUT_VARS = func_configs.OUT_VARS  # Number of output variables (y-values)
    TARGETS = func_configs.TARGETS    # Target values for output

    # Objective function dependent variables
    func_F = func_configs.OBJECTIVE_FUNC  # objective function
    constr_F = func_configs.CONSTR_FUNC   # constraint function

    # optimizer specific vars
    BP = 0.5            # Beta Par. step size contraction
    GP = 1              # Gamma Par. step size expansion
    SF = 2              # Search Frequency

    # optimizer setting values
    parent = None                 # Optional parent class for optimizer
                                    # (Used for passing debug messages or
                                    # other information that will appear 
                                    # in GUI panels)

    best_eval = 1

    suppress_output = True   # Suppress the console output of GLODS


    allow_update = True      # Allow objective call to update state 
                            # (Can be set on each iteration to allow 
                            # for when control flow can be returned 
                            # to GLODS)   


    # instantiation of GLODS optimizer 
    # Constant variables
    # parameters intentionally match the multi_glods constructor
    opt_params = {'BP': [BP],               # Beta Par
                'GP': [GP],                 # Gamma Par
                'SF': [SF],                 # Search Frequency
                'R_TOL': [R_TOL]}           # Radius/step-size tolerance

    opt_df = pd.DataFrame(opt_params)
    myGlods = glods(LB, UB, TARGETS, E_TOL, MAXIT,
                            func_F, constr_F,
                            opt_df,
                            parent=parent)   

    # arguments should take the form: 
    # glods([[float, float, ...]], [[float, float, ...]], [[float, ...]], float, int,
    # func, func,
    # dataFrame,
    # class obj,
    # bool, [int, int, ...],
    # int) 
    #  
    # opt_df contains class-specific tuning parameters
    # BP: float. beta_par, step size contraction coefficient (0,1). default 0.5
    # GP: float. gamma_par, step size expansion coefficient >= 1. default 1
    # SF: int.   search frequency. a search (multistart) step runs after
    #            SF consecutive unsuccessful iterations. default 2
    # R_TOL: float. radius/step-size tolerance
    #

```

### State Machine-based Structure

This optimizer uses a state machine structure to control the sampling of new points, the call to the objective function, and the evaluation of current positions. The state machine implementation preserves the initial algorithm while making it possible to integrate other programs, classes, or functions as the objective function.

A controller with a `while loop` to check the completion status of the optimizer drives the process. Completion status is determined by at least 1) a set MAX number of iterations, 2) the convergence to a given target using the L2 norm, and 3) the radius (step-size) tolerance, where no active point has a step size above R_TOL and every surviving search has refined to mesh resolution. Iterations are counted by calls to the objective function. 

GLODS iterations are batch-natured (a search wants n evaluations, a poll wants up to 2n), so the implementation serializes them through an internal evaluation queue. The controller sees the standard one-evaluation-per-pass rhythm: `call_objective` evaluates the current queue sample and computes its fitness immediately, and `step` consumes the previous result, running the merge/update and building the next iteration's queue when the queue drains. Nothing in `step` calls the objective function.

Within this `while loop` are three function calls to control the optimizer class:
* **complete**: the `complete function` checks the status of the optimizer and if it has met the convergence or stop conditions.
* **step**: the `step function` takes a boolean variable (suppress_output) as an input to control detailed printout on the current sampling status. This function moves the optimizer one step forward.  
* **call_objective**: the `call_objective function` takes a boolean variable (allow_update) to control if the objective function is able to be called. In most implementations, this value will always be true. However, there may be cases where the controller or a program running the state machine needs to assert control over this function without stopping the loop.

Additionally, **get_convergence_data** can be used to preview the current status of the optimizer, including the current best evaluation and the iterations, and **get_latest_eval** returns the fitness of the point evaluated on the current pass.

The code below is an example of this process:

```python
    while not myOptimizer.complete():
        # step through optimizer processing
        # consumes the previous evaluation and advances the point list
        myOptimizer.step(suppress_output)
        # call the objective function, control 
        # when it is allowed to update and return 
        # control to optimizer
        myOptimizer.call_objective(allow_update)
        # check the current progress of the optimizer
        # iter: the number of objective function calls
        # eval: current 'best' evaluation of the optimizer
        iter, eval = myOptimizer.get_convergence_data()
        if (eval < best_eval) and (eval != 0):
            best_eval = eval
        
        # optional. if the optimizer is not printing out detailed 
        # reports, preview by checking the iteration and best evaluation

        if suppress_output:
            if iter%100 ==0: #print out every 100th iteration update
                print("Iteration")
                print(iter)
                print("Best Eval")
                print(best_eval)
```

### Constraint Handling
Users must create their own constraint function for their problems, if there are constraints beyond the problem bounds.  This is then passed into the constructor. If the default constraint function is used, it always returns true (which means there are no constraints).

### Boundary Types
Feasibility (bounds and the constraint function) is checked at queue-build time, matching the multiGLODS `feasible()` behavior: infeasible candidates are never evaluated and never enter the point list. Search points that fail the constraint check respawn at a new random location; infeasible poll directions are skipped, and a fully infeasible poll counts as an unsuccessful iteration.

Potentially other boundary types may be implemented, but experimentation is needed. 

### The Active Point Set

At convergence, the active point set is the GLODS deliverable: it approximates ALL the local and global minimizers found, not just the single best. The active points can be retrieved with:

```python
    # single best:
    soln = myGlods.get_optimized_soln()
    # ALL minimizers found:
    points, values = myGlods.get_active_points(refined_only=True)
```

`get_active_points(refined_only=True)` returns just the refined candidates (step size below R_TOL), excluding coarse in-flight search points. The distinction matters most after an E_TOL exit, which can leave unrefined search points active. See GLODS_ALGORITHM.md for the merge rules that maintain the active set.

### Objective Function Handling

The objective function is handled in two parts. 

* First, a defined function, such as one passed in from `func_F.py` (see examples), is evaluated based on current sample locations. This allows for the optimizers to be utilized in the context of 1. benchmark functions from the objective function library, 2. user defined functions, 3. replacing explicitly defined functions with outside calls to programs such as simulations or other scripts that return a matrix of evaluated outputs. 

* Secondly, the actual objective function is evaluated. In the AntennaCAT set of optimizers, the objective function evaluation is either a `TARGET` or `THRESHOLD` evaluation. For a `TARGET` evaluation, which is the default behavior, the optimizer minimizes the absolute value of the difference of the target outputs and the evaluated outputs. A `THRESHOLD` evaluation includes boolean logic to determine if a 'greater than or equal to' or 'less than or equal to' or 'equal to' relation between the target outputs (or thresholds) and the evaluated outputs exist. 

Future versions may include options for function minimization when target values are absent. 

#### Creating a Custom Objective Function

Custom objective functions can be used by creating a directory with the following files:
* configs_F.py
* constr_F.py
* func_F.py

`configs_F.py` contains lower bounds, upper bounds, the number of input variables, the number of output variables, the target values, and a global minimum if known. This file is used primarily for unit testing and evaluation of accuracy. If these values are not known, or are dynamic, then they can be included experimentally in the controller that runs the optimizer's state machine. 

`constr_F.py` contains a function called `constr_F` that takes in an array, `X`, of particle positions to determine if the particle or agent is in a valid or invalid location. 

`func_F.py` contains the objective function, `func_F`, which takes two inputs. The first input, `X`, is the array of particle or agent positions. The second input, `NO_OF_OUTS`, is the integer number of output variables, which is used to set the array size. In included objective functions, the default value is hardcoded to work with the specific objective function.

Below are examples of the format for these files.

`configs_F.py`:
```python
OBJECTIVE_FUNC = func_F
CONSTR_FUNC = constr_F
OBJECTIVE_FUNC_NAME = "one_dim_x_test.func_F" #format: FUNCTION NAME.FUNCTION
CONSTR_FUNC_NAME = "one_dim_x_test.constr_F" #format: FUNCTION NAME.FUNCTION

# problem dependent variables
LB = [[0]]             # Lower boundaries
UB = [[1]]             # Upper boundaries
IN_VARS = 1            # Number of input variables (x-values)
OUT_VARS = 1           # Number of output variables (y-values) 
TARGETS = [0]          # Target values for output
GLOBAL_MIN = []        # Global minima sample, if they exist. 

```

`constr_F.py`, with no constraints:
```python
def constr_F(x):
    F = True
    return F
```

`constr_F.py`, with constraints:
```python
def constr_F(X):
    F = True
    # objective function/problem constraints
    if (X[2] > X[0]/2) or (X[2] < 0.1):
        F = False
    return F
```

`func_F.py`:
```python
import numpy as np
import time

def func_F(X, NO_OF_OUTS=1):
    F = np.zeros((NO_OF_OUTS))
    noErrors = True
    try:
        x = X[0]
        F = np.sin(5 * x**3) + np.cos(5 * x) * (1 - np.tanh(x ** 2))
    except Exception as e:
        print(e)
        noErrors = False

    return [F], noErrors
```

#### Internal Objective Function Example

There are three functions included in the repository:
1) Himmelblau's function, which takes 2 inputs and has 1 output
2) A multi-objective function with 3 inputs and 2 outputs (see lundquist_3_var)
3) A single-objective function with 1 input and 1 output (see one_dim_x_test)

Each function has four files in a directory:
   1) configs_F.py - contains imports for the objective function and constraints, CONSTANT assignments for functions and labeling, boundary ranges, the number of input variables, the number of output values, and the target values for the output
   2) constr_F.py - contains a function with the problem constraints, both for the function and for error handling in the case of under/overflow. 
   3) func_F.py - contains a function with the objective function.
   4) graph.py - contains a script to graph the function for visualization.

Other multi-objective functions can be applied to this project by following the same format (and several have been collected into a compatible library, and will be released in a separate repo)

<p align="center">
        <img src="media/himmelblau_plots.png" alt="Himmelblau’s function" height="250">
</p>
   <p align="center">Plotted Himmelblau’s Function with 3D Plot on the Left, and a 2D Contour on the Right</p>

```math
f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2
```

| Global Minima | Boundary | Constraints |
|----------|----------|----------|
| f(3, 2) = 0                 | $-5 \leq x,y \leq 5$  |   | 
| f(-2.805118, 3.121212) = 0  | $-5 \leq x,y \leq 5$  |   | 
| f(-3.779310, -3.283186) = 0 | $-5 \leq x,y \leq 5$  |   | 
| f(3.584428, -1.848126) = 0  | $-5 \leq x,y \leq 5$   |   | 

<p align="center">
        <img src="media/obj_func_pareto.png" alt="Function Feasible Decision Space and Objective Space with Pareto Front" height="200">
</p>
   <p align="center">Plotted Multi-Objective Function Feasible Decision Space and Objective Space with Pareto Front</p>

```math
\text{minimize}: 
\begin{cases}
f_{1}(\mathbf{x}) = (x_1-0.5)^2 + (x_2-0.1)^2 \\
f_{2}(\mathbf{x}) = (x_3-0.2)^4
\end{cases}
```

| Num. Input Variables| Boundary | Constraints |
|----------|----------|----------|
| 3      | $0.21\leq x_1\leq 1$ <br> $0\leq x_2\leq 1$ <br> $0.1 \leq x_3\leq 0.5$  | $x_3\gt \frac{x_1}{2}$ or $x_3\lt 0.1$| 

<p align="center">
        <img src="media/1D_test_plots.png" alt="Function Feasible Decision Space and Objective Space with Pareto Front" height="200">
</p>
   <p align="center">Plotted Single Input, Single-objective Function Feasible Decision Space and Objective Space with Pareto Front</p>

```math
f(\mathbf{x}) = sin(5 * x^3) + cos(5 * x) * (1 - tanh(x^2))
```
| Num. Input Variables| Boundary | Constraints |
|----------|----------|----------|
| 1      | $0\leq x\leq 1$  | $0\leq x\leq 1$| |

Local minima at $(0.444453, -0.0630916)$

Global minima at $(0.974857, -0.954872)$


### Target vs. Threshold Configuration

An April 2025 feature is the user ability to toggle TARGET and THRESHOLD evaluation for the optimized values. The key variables for this are:

```python
# Boolean. use target or threshold. True = THRESHOLD, False = EXACT TARGET
evaluate_threshold = True  

# array
TARGETS = func_configs.TARGETS    # Target values for output from function configs
# OR:
TARGETS = [0,0,0] #manually set BASED ON PROBLEM DIMENSIONS

# threshold is same dims as TARGETS
# 0 = use target value as actual target. value should EQUAL target
# 1 = use as threshold. value should be LESS THAN OR EQUAL to target
# 2 = use as threshold. value should be GREATER THAN OR EQUAL to target
#DEFAULT THRESHOLD
THRESHOLD = np.zeros_like(TARGETS) 
# OR
THRESHOLD = [0,1,2] # can be any mix of TARGET and THRESHOLD  
```

To implement this, the original `self.Flist` objective function calculation has been replaced with the function `objective_function_evaluation`, which returns a numpy array.

The original calculation:
```python
self.Flist = abs(self.targets - self.Fvals)
```
Where `self.Fvals` is a re-arranged and error checked returned value from the passed in function from `func_F.py` (see examples for the internal objective function or creating a custom objective function). 

When using a THRESHOLD, the `Flist` value corresponding to the target is set to epsilon (the smallest system value) if the evaluated `func_F` value meets the threshold condition for that target item. If the threshold is not met, the absolute value of the difference of the target output and the evaluated output is used. With a THRESHOLD configuration, each value in the numpy array is evaluated individually, so some values can be 'greater than or equal to' the target while others are 'equal' or 'less than or equal to' the target. 



## Example Implementations

### Basic Example
`main_test.py` provides a sample use case of the optimizer with tunable parameters, and prints the active point set (the local and global minimizer candidates) at the end of the run.

### Detailed Messages
`main_test_details.py` provides an example using a parent class, and the self.suppress_output flag to control error messages that are passed back to the parent class to be printed with a timestamp. This implementation sets up the hooks for integration with AntennaCAT in order to provide the user feedback of warnings and errors.

### Realtime Graph

`main_test_graph.py` provides an example using a parent class, and the self.suppress_output flag to control error messages that are passed back to the parent class to be printed with a timestamp. Additionally, a realtime graph shows the evaluated search locations at every step, with the active points (the local and global minimizer candidates) overlaid in red. The left plot shows the search location(s), and the right shows the history of the global best fitness values in relation to the target.

NOTE: if you close the graph as the code is running, the code will continue to run, but the graph will not re-open.

## References

[1] A. L. Custódio and J. F. A. Madeira, “GLODS: Global and Local Optimization using Direct Search,” Journal of Global Optimization, vol. 62, no. 1, pp. 1–28, Aug. 2014, doi: https://doi.org/10.1007/s10898-014-0224-9.

[2] A. L. Custódio and J. F. A. Madeira, “MultiGLODS: global and local multiobjective optimization using direct search,” Journal of Global Optimization, vol. 72, no. 2, pp. 323–345, Feb. 2018, doi: https://doi.org/10.1007/s10898-018-0618-1.

## Related Publications and Repositories
This software works as a stand-alone implementation, and as one of the optimizers integrated into AntennaCAT. Publications featuring the code as part of AntennaCAT will be added as they become public.

When citing the algorithm itself, please refer to the original publication for GLODS by the original authors:

 A. L. Custódio and J. F. A. Madeira, GLODS: Global and Local Optimization 
using Direct Search, Journal of Global Optimization, 62 (2014), 1 - 28

## Licensing

The code in this repository has been released under GPL-2.0