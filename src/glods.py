#! /usr/bin/python3

##--------------------------------------------------------------------\
#   glods_python
#   './glods_python/src/glods.py'
#   GLODS (Global and Local Optimization using Direct Search) class.
#       Single-objective predecessor of MultiGLODS (Custodio & Madeira
#       2014). Multistart directional direct search: alternates between
#       scattering new search points and polling around active points,
#       merging searches that approach each other so that only locally
#       nondominated (here: locally best) points remain active. At
#       convergence, the active point set identifies the local AND
#       global minimizers found.
#
#       Follows the AntennaCAT pso_basic template: decoupled
#       step()/call_objective() with one objective evaluation per
#       controller loop pass. Like the DIRECT implementation, the
#       batch-natured search/poll steps are serialized through an
#       internal evaluation queue; the GLODS iteration boundary is
#       invisible to the controller.
#
#       opt_df parameters intentionally match the multi_glods wrapper
#       (BP, GP, SF, R_TOL) so the two are drop-in siblings.
#
#   Author(s): Lauren Linkous, (template: Jonathan Lundquist)
#   Last update: June 11, 2026
##--------------------------------------------------------------------\

import numpy as np
from numpy.random import Generator, MT19937
import sys
np.seterr(all='raise')


class glods:
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
    # R_TOL: float. radius/step-size tolerance. active points whose step
    #            size falls below R_TOL are local-minimizer candidates and
    #            are no longer polled. when no pollable point remains, the
    #            run is converged by radius.
    #

    def __init__(self, lbound, ubound, targets, E_TOL, maxit,
                 obj_func, constr_func,
                 opt_df,
                 parent=None,
                 evaluate_threshold=False, obj_threshold=None,
                 decimal_limit=4):

        # Optional parent class func call to write out values that trigger constraint issues
        self.parent = parent

        self.number_decimals = int(decimal_limit)  # limit the number of decimals
                                              # used in cases where real life has limitations on resolution.
                                              # NOTE: applied only at evaluation time (the x passed to
                                              # func_F). point-list arithmetic (steps, radii, merge
                                              # distances) is kept at full precision in the unit cube.

        # evaluation method for targets
        # True: Evaluate as true targets
        # False: Evaluate as thresholds based on information in obj_threshold
        if evaluate_threshold == False:
            self.evaluate_threshold = False
            self.obj_threshold = None
        else:
            if not(len(obj_threshold) == len(targets)):
                self.debug_message_printout("WARNING: THRESHOLD option selected. +\
                Dimensions for THRESHOLD do not match TARGET array. Defaulting to TARGET search.")
                self.evaluate_threshold = False
                self.obj_threshold = None
            else:
                self.evaluate_threshold = evaluate_threshold  # bool
                self.obj_threshold = np.array(obj_threshold).reshape(-1, 1)  # np.array

        # unpack the opt_df standardized vals
        self.beta_par = float(opt_df['BP'][0])
        self.gamma_par = float(opt_df['GP'][0])
        self.search_freq = int(opt_df['SF'][0])
        self.R_TOL = float(opt_df['R_TOL'][0])

        # optimizer init:
        heightl = np.shape(lbound)[0]
        widthl = np.shape(lbound)[1]
        heightu = np.shape(ubound)[0]
        widthu = np.shape(ubound)[1]

        lbound = np.array(lbound[0], dtype=float)
        ubound = np.array(ubound[0], dtype=float)

        self.rng = Generator(MT19937())

        if ((heightl > 1) and (widthl > 1)) \
           or ((heightu > 1) and (widthu > 1)) \
           or (heightu != heightl) \
           or (widthl != widthu):

            if self.parent == None:
                pass
            else:
                self.parent.debug_message_printout("Error lbound and ubound must be 1xN-dimensional \
                                                        arrays with the same length")

        else:

            self.lbound = lbound
            self.ubound = ubound
            self.n = int(len(lbound))

            '''
            self.X                  : (m, n) point list in UNIT-CUBE coordinates.
            self.Fnorm              : (m,)   scalar fitness per point (L2 norm of its Flist).
            self.F_point            : (m, out) Flist (distance-from-target) per point.
            self.alfa               : (m,)   per-point step size (unit-cube units).
            self.radius             : (m,)   per-point comparison/merge radius.
            self.active             : (m,)   bool. True = locally nondominated (locally best).
            self.output_size        : An integer value for the output size of obj func.
            self.Gb                 : Global best position (problem coordinates).
            self.F_Gb               : Fitness value corresponding to the global best position.
            self.targets            : Target values for the optimization process.
            self.maxit              : Maximum number of OBJECTIVE CALLS.
            self.E_TOL              : Error tolerance.
            self.obj_func           : Objective function to be optimized.
            self.constr_func        : Constraint function.
            self.iter               : Objective function call count.
            self.queue              : List of pending sample points (unit coords + bookkeeping).
            self.current_sample     : Index of the current sample in the queue.
            self.pending            : Evaluated-but-unmerged results for the current iteration.
            self.unsuc_consec       : Consecutive unsuccessful iteration count (search trigger).
            self.poll_center        : Index of the rect being polled (None for search iterations).
            self.allow_update       : Flag indicating whether to allow updates.
            self.Flist              : Fitness (distance) of the most recent evaluation.
            self.Fvals              : Raw objective outputs of the most recent evaluation.
            '''

            self.output_size = len(targets)
            self.targets = np.array(targets).reshape(-1, 1)
            self.maxit = maxit
            self.E_TOL = E_TOL
            self.obj_func = obj_func
            self.constr_func = constr_func
            self.iter = 0
            self.allow_update = 0
            self.Flist = []
            self.Fvals = []

            # point list
            self.X = np.zeros((0, self.n))
            self.Fnorm = np.zeros((0,))
            self.F_point = np.zeros((0, self.output_size))
            self.alfa = np.zeros((0,))
            self.radius = np.zeros((0,))
            self.active = np.zeros((0,), dtype=bool)

            # step size / radius initialization (unit-cube units)
            self.alfa_ini = 0.5
            self.radius_ini = 0.5
            # sufficient decrease coefficient (hardcoded default, matching
            # the multiGLODS translation's handling of suf_decrease)
            self.suf_decrease = 1e-3

            # global best
            self.Gb = sys.maxsize * np.ones(self.n)
            self.F_Gb = sys.maxsize * np.ones((1, self.output_size))

            # iteration bookkeeping
            self.unsuc_consec = 0
            self.poll_center = None
            self.pending = []

            # initial point set: matches the multi_glods_python translation
            # choice (Pini): n points along the domain diagonal plus the
            # midpoint. deterministic, covers the bounds.
            self.queue = []
            self.current_sample = 0
            if self.n > 1:
                fracs = np.linspace(0.0, 1.0, self.n)
                for fr in fracs:
                    self.try_queue(fr * np.ones(self.n), 'search', None)
            else:
                self.try_queue(np.array([0.0]), 'search', None)
                self.try_queue(np.array([1.0]), 'search', None)
            self.try_queue(0.5 * np.ones(self.n), 'search', None)

            self.debug_message_printout("GLODS successfully initialized")

    def debug_message_printout(self, msg):
        if self.parent == None:
            pass
        else:
            self.parent.debug_message_printout(msg)

    # unit cube <-> problem space
    def denormalize(self, x_unit):
        return np.round(self.lbound + x_unit * (self.ubound - self.lbound),
                        self.number_decimals)

    def try_queue(self, x_unit, kind, parent):
        # feasibility is checked at QUEUE-BUILD time, matching the
        # multiGLODS feasible() behavior: infeasible candidates are never
        # evaluated and never enter the point list. returns True if queued.
        if np.any(x_unit < 0.0) or np.any(x_unit > 1.0):
            return False
        if self.constr_func(self.denormalize(x_unit)) == False:
            return False
        self.queue.append({'x': x_unit, 'kind': kind, 'parent': parent})
        return True

    def objective_function_evaluation(self, Fvals, targets):
        # pass in the Fvals & targets so that it's easier to track bugs
        # identical to the pso_basic implementation.
        epsilon = np.finfo(float).eps

        Flist = np.zeros(len(Fvals))

        if self.evaluate_threshold == True:  # THRESHOLD
            ctr = 0
            for i in targets:
                o_thres = int(self.obj_threshold[ctr].item())
                t = targets[ctr].item()
                fv = Fvals[ctr].item()

                if o_thres == 0:  # TARGET. default
                    Flist[ctr] = abs(t - fv)
                elif o_thres == 1:  # LESS THAN OR EQUAL
                    if fv <= t:
                        Flist[ctr] = epsilon
                    else:
                        Flist[ctr] = abs(t - fv)
                elif o_thres == 2:  # GREATER THAN OR EQUAL
                    if fv >= t:
                        Flist[ctr] = epsilon
                    else:
                        Flist[ctr] = abs(t - fv)
                else:
                    self.debug_message_printout("ERROR: unrecognized threshold value. Evaluating as TARGET")
                    Flist[ctr] = abs(t - fv)
                ctr = ctr + 1
        else:  # TARGET as default
            Flist = abs(targets - Fvals)

        return Flist

    def call_objective(self, allow_update):
        # evaluates the objective function at the current queue sample AND
        # computes the target/threshold fitness. After this returns,
        # get_latest_eval() and converged() reflect the evaluation that
        # just happened, BEFORE the next step() consumes it and advances.
        if self.current_sample >= len(self.queue):
            return True  # nothing pending (should not happen in a normal loop)

        x = self.denormalize(self.queue[self.current_sample]['x'])

        newFVals, noError = self.obj_func(x, self.output_size)
        if noError == True:
            self.Fvals = np.array(newFVals).reshape(-1, 1)
            if allow_update:
                # EVALUATE OBJECTIVE FUNCTION - TARGET OR THRESHOLD
                self.Flist = self.objective_function_evaluation(self.Fvals, self.targets)
                self.iter = self.iter + 1
                self.allow_update = 1
            else:
                self.allow_update = 0
        return noError  # return is for error reporting purposes only

    def step(self, suppress_output):
        if not suppress_output:
            msg = "\n-----------------------------\n" + \
                "STEP #" + str(self.iter) + "\n" + \
                "-----------------------------\n" + \
                "Queue sample:\n" + \
                str(self.current_sample) + " of " + str(len(self.queue)) + "\n" + \
                "Active points\n" + \
                str(int(np.sum(self.active))) + " of " + str(np.shape(self.X)[0]) + "\n" + \
                "-----------------------------"
            self.debug_message_printout(msg)

        if self.allow_update:
            # guard: nothing pending to consume (empty queue edge case)
            if self.current_sample >= len(self.queue):
                return
            # 1. CONSUME the previously evaluated sample (fresh data from
            #    the last call_objective). nothing below re-calls func_F.
            entry = self.queue[self.current_sample]
            self.pending.append((entry,
                                 np.linalg.norm(self.Flist),
                                 np.atleast_1d(np.squeeze(np.asarray(self.Flist)))))
            self.current_sample = self.current_sample + 1

            # 2. ADVANCE. when the queue is exhausted, this GLODS iteration
            #    is complete: merge the evaluated points into the list,
            #    determine success, expand/contract step sizes, and build
            #    the next iteration's queue (search or poll).
            if self.current_sample >= len(self.queue):
                self.process_iteration()
                self.build_next_queue()

            if self.complete() and not suppress_output:
                msg = "\nOPTIMIZATION COMPLETE:\nPoints: \n" + str(self.Gb) + "\n" + \
                    "Iterations: \n" + str(self.iter) + "\n" + \
                    "Flist: \n" + str(self.F_Gb) + "\n" + \
                    "Norm Flist: \n" + str(np.linalg.norm(self.F_Gb)) + "\n" + \
                    "Active points (local minimizer candidates): \n" + \
                    str(int(np.sum(self.active))) + "\n"
                self.debug_message_printout(msg)

    def merge_point(self, x_unit, fnorm, flist, alfa_new, radius_new,
                    force_inactive=False):
        # GLODS merge: the single-objective specialization of the
        # MultiGLODS rule "points sufficiently close to each other are
        # compared and only nondominated points remain active."
        #   - link set L: existing points within max(r_i, r_new) of the
        #     candidate.
        #   - the candidate becomes ACTIVE iff it is strictly better than
        #     every active point in L.
        #   - any active point in L that is strictly worse than the
        #     candidate is deactivated (its search merges into the
        #     candidate's).
        # inactive points stay in the list for record/plotting.
        # returns True if the candidate entered as active (a 'change').
        m = np.shape(self.X)[0]
        # sufficient decrease (forcing function rho = c * alfa^2, with the
        # hardcoded default coefficient, like the multiGLODS translation):
        # a candidate only activates if it improves on the best linked
        # active point by more than rho. prevents chains of marginally
        # better points from polluting the active set.
        rho = self.suf_decrease * (alfa_new ** 2)
        became_active = not force_inactive
        if m > 0 and became_active:
            dist = np.linalg.norm(self.X - x_unit, axis=1)
            linked = dist <= np.maximum(self.radius, radius_new)
            # exact duplicate: ignore the candidate entirely
            if np.any(dist < 1e-12):
                return False
            linked_active = np.nonzero(linked & self.active)[0]
            if len(linked_active) > 0:
                best_linked = np.min(self.Fnorm[linked_active])
                became_active = fnorm <= best_linked - rho
            if became_active:
                for i in linked_active:
                    if fnorm <= self.Fnorm[i] - rho:
                        self.active[i] = False

        self.X = np.vstack([self.X, x_unit])
        self.Fnorm = np.append(self.Fnorm, fnorm)
        self.F_point = np.vstack([self.F_point, flist])
        self.alfa = np.append(self.alfa, alfa_new)
        self.radius = np.append(self.radius, radius_new)
        self.active = np.append(self.active, became_active)

        # track global best
        if fnorm < np.linalg.norm(self.F_Gb):
            self.F_Gb = np.array([flist])
            self.Gb = self.denormalize(x_unit)

        return became_active

    def process_iteration(self):
        success = False
        new_active = []

        # poll children: complete polling evaluates every direction, but
        # only the BEST child may activate. siblings are 2*alfa apart with
        # link radius alfa - they can never see each other - so letting
        # several near-tied siblings activate strands stale active points
        # at that scale forever. evaluating all and taking the best keeps
        # the multiGLODS poll_complete=1 behavior while preventing the
        # sibling leak. search points (random, distinct basins) all merge
        # normally.
        poll_pending = sorted([p for p in self.pending if p[0]['kind'] == 'poll'],
                              key=lambda p: p[1])
        search_pending = [p for p in self.pending if p[0]['kind'] == 'search']

        for k, (entry, fnorm, flist) in enumerate(search_pending + poll_pending):
            if entry['kind'] == 'search':
                a_new = self.alfa_ini
                r_new = self.radius_ini
                force_inactive = False
            else:  # poll: children inherit the poll center's step size
                a_new = self.alfa[entry['parent']]
                r_new = self.alfa[entry['parent']]
                force_inactive = (entry is not poll_pending[0][0])
            if self.merge_point(entry['x'], fnorm, flist, a_new, r_new,
                                force_inactive=force_inactive):
                success = True
                new_active.append(np.shape(self.X)[0] - 1)

        if success:
            # expand step sizes of the newly active points; the comparison
            # radius never shrinks below the step size (multiGLODS
            # run_update: radius = max(radius, alfa)).
            self.unsuc_consec = 0
            for i in new_active:
                self.alfa[i] = self.alfa[i] * self.gamma_par
                self.radius[i] = max(self.radius[i], self.alfa[i])
        else:
            self.unsuc_consec = self.unsuc_consec + 1
            if self.poll_center is not None:
                # unsuccessful poll: contract the poll center's step size
                self.alfa[self.poll_center] = self.alfa[self.poll_center] * self.beta_par

        self.pending = []

    def pollable(self):
        # active points whose step size is still above the radius
        # tolerance. points below R_TOL are local-minimizer candidates
        # and are no longer refined.
        return np.nonzero(self.active & (self.alfa >= self.R_TOL))[0]

    def build_next_queue(self):
        self.queue = []
        self.current_sample = 0
        self.poll_center = None

        # safety counter: contraction below R_TOL or full deactivation
        # will end the loop via complete(); this guards degenerate cases.
        attempts = 0
        while len(self.queue) == 0 and attempts < 1000:
            attempts = attempts + 1
            if self.complete():
                return

            cand = self.pollable()
            do_search = (self.unsuc_consec >= self.search_freq) or (len(cand) == 0)

            if do_search:
                # SEARCH STEP (multistart): scatter n random feasible
                # points across the bounds. respawn on constraint
                # violation, like the swarm template.
                self.unsuc_consec = 0
                for _ in range(self.n):
                    tries = 0
                    while tries < 100:
                        tries = tries + 1
                        x_new = self.rng.random(self.n)
                        if self.try_queue(x_new, 'search', None):
                            break
            else:
                # POLL STEP: center = pollable point with the BEST fitness
                # (ties: largest step size). this is the more local of the
                # GLODS poll-center options - it refines the best basin to
                # E_TOL quickly, while the search step maintains global
                # coverage. (the multiGLODS translation uses the more
                # global largest-step-first option instead.)
                order = sorted(cand, key=lambda i: (self.Fnorm[i], -self.alfa[i]))
                pc = order[0]
                self.poll_center = pc
                queued_any = False
                for i in range(self.n):
                    for sign in (1, -1):
                        x_new = self.X[pc].copy()
                        x_new[i] = x_new[i] + sign * self.alfa[pc]
                        queued_any = self.try_queue(x_new, 'poll', pc) or queued_any
                if not queued_any:
                    # every poll direction was infeasible: treat as an
                    # unsuccessful poll and contract, then try again.
                    self.alfa[pc] = self.alfa[pc] * self.beta_par
                    self.unsuc_consec = self.unsuc_consec + 1
                    self.poll_center = None

    # funcs from other optimizers in the AntennaCAT set for stop conditions

    def get_latest_eval(self):
        # L2 norm of the fitness of the MOST RECENTLY evaluated sample
        # (pending, not yet consumed by step()). allow_update == 1 means a
        # fresh evaluation is waiting.
        if self.allow_update and np.shape(self.Flist)[0] > 0:
            return np.linalg.norm(self.Flist)
        return None

    def converged(self):
        convergence = np.linalg.norm(self.F_Gb) < self.E_TOL
        return convergence

    def maxed(self):
        max_iter = self.iter >= self.maxit
        return max_iter

    def radius_converged(self):
        # no active point has a step size above R_TOL: every surviving
        # search has refined down to mesh resolution. the active set now
        # identifies the local (and global) minimizers found.
        if np.shape(self.X)[0] == 0:
            return False
        return len(self.pollable()) == 0

    def complete(self):
        done = self.converged() or self.maxed() or self.radius_converged()
        return done

    def get_convergence_data(self):
        best_eval = np.linalg.norm(self.F_Gb)
        return self.iter, best_eval

    def get_optimized_soln(self):
        return np.vstack(self.Gb)

    def get_optimized_outs(self):
        return np.vstack(np.atleast_1d(np.squeeze(self.F_Gb)))

    def get_active_points(self, refined_only=False):
        # THE GLODS selling point: the active point set approximates the
        # local and global minimizers found, not just the single best.
        # refined_only=True returns only the active points whose step size
        # has contracted below R_TOL - the points GLODS has actually
        # refined to mesh resolution (the true minimizer candidates),
        # excluding coarse active points from in-flight searches.
        # returns (positions, Flist values) in problem coordinates.
        if refined_only:
            idx = np.nonzero(self.active & (self.alfa < self.R_TOL))[0]
        else:
            idx = np.nonzero(self.active)[0]
        pos = self.lbound + self.X[idx] * (self.ubound - self.lbound)
        return pos, self.F_point[idx]

    # for plotting
    def get_search_locations(self):
        # denormalized positions of every point evaluated so far
        if np.shape(self.X)[0] == 0:
            return np.zeros((0, self.n))
        return self.lbound + self.X * (self.ubound - self.lbound)

    def get_fitness_values(self):
        return self.F_point

    def export_glods(self):
        glods_export = {
            'evaluate_threshold': [self.evaluate_threshold],
            'obj_threshold': [self.obj_threshold],
            'targets': [self.targets],
            'lbound': [self.lbound],
            'ubound': [self.ubound],
            'output_size': [self.output_size],
            'maxit': [self.maxit],
            'E_TOL': [self.E_TOL],
            'R_TOL': [self.R_TOL],
            'beta_par': [self.beta_par],
            'gamma_par': [self.gamma_par],
            'search_freq': [self.search_freq],
            'alfa_ini': [self.alfa_ini],
            'radius_ini': [self.radius_ini],
            'iter': [self.iter],
            'allow_update': [self.allow_update],
            'X': [self.X],
            'Fnorm': [self.Fnorm],
            'F_point': [self.F_point],
            'alfa': [self.alfa],
            'radius': [self.radius],
            'active': [self.active],
            'Gb': [self.Gb],
            'F_Gb': [self.F_Gb],
            'Flist': [self.Flist],
            'Fvals': [self.Fvals],
            'queue': [self.queue],
            'current_sample': [self.current_sample],
            'pending': [self.pending],
            'unsuc_consec': [self.unsuc_consec],
            'poll_center': [self.poll_center]}
        return glods_export

    def import_glods(self, glods_export, obj_func):
        self.evaluate_threshold = glods_export['evaluate_threshold'][0]
        self.obj_threshold = glods_export['obj_threshold'][0]
        self.targets = glods_export['targets'][0]
        self.lbound = glods_export['lbound'][0]
        self.ubound = glods_export['ubound'][0]
        self.output_size = glods_export['output_size'][0]
        self.maxit = glods_export['maxit'][0]
        self.E_TOL = glods_export['E_TOL'][0]
        self.R_TOL = glods_export['R_TOL'][0]
        self.beta_par = glods_export['beta_par'][0]
        self.gamma_par = glods_export['gamma_par'][0]
        self.search_freq = glods_export['search_freq'][0]
        self.alfa_ini = glods_export['alfa_ini'][0]
        self.radius_ini = glods_export['radius_ini'][0]
        self.iter = glods_export['iter'][0]
        self.allow_update = glods_export['allow_update'][0]
        self.X = glods_export['X'][0]
        self.Fnorm = glods_export['Fnorm'][0]
        self.F_point = glods_export['F_point'][0]
        self.alfa = glods_export['alfa'][0]
        self.radius = glods_export['radius'][0]
        self.active = glods_export['active'][0]
        self.Gb = glods_export['Gb'][0]
        self.F_Gb = glods_export['F_Gb'][0]
        self.Flist = glods_export['Flist'][0]
        self.Fvals = glods_export['Fvals'][0]
        self.queue = glods_export['queue'][0]
        self.current_sample = glods_export['current_sample'][0]
        self.pending = glods_export['pending'][0]
        self.unsuc_consec = glods_export['unsuc_consec'][0]
        self.poll_center = glods_export['poll_center'][0]
        self.n = int(len(self.lbound))
        self.obj_func = obj_func
