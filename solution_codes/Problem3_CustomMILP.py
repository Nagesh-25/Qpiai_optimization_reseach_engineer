"""
Problem3_CustomMILP_fixed.py

Simplified and corrected branch-and-bound (educational) for small MILPs:
 - Uses scipy.linprog for LP relaxations (HiGHS).
 - Removes the misleading "Gomory cut" implementation.
 - Implements an explicit branch-on-variable strategy: pick most fractional var and branch by adding bound.
 - Uses an iterative stack to avoid recursion depth issues.
 - Good for small teaching problems; not a production MIP solver.

Usage:
 - Define c, A_ub, b_ub, A_eq, b_eq, var_types (list of 'C' or 'I'), and call BranchAndBoundSolver.solve()
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from copy import deepcopy
import math

try:
    from scipy.optimize import linprog
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

@dataclass
class MILPProblem:
    c: np.ndarray                # objective coefficients (minimize c^T x)
    A_ub: Optional[np.ndarray]   # inequality lhs (A_ub x <= b_ub)
    b_ub: Optional[np.ndarray]
    A_eq: Optional[np.ndarray]
    b_eq: Optional[np.ndarray]
    bounds: Optional[List[Tuple[float, float]]]  # bounds per var
    var_types: List[str]         # 'C' continuous, 'I' integer

@dataclass
class BnBNode:
    ub_idx_constraints: Optional[List[Tuple[np.ndarray, float]]]  # added upper-bound style cuts (a, rhs) representing a^T x <= rhs
    lb_idx_constraints: Optional[List[Tuple[int, float]]]         # variable lower-bound settings: list of (var_idx, lower_bound)
    ub_bounds: Optional[List[Tuple[int, float]]]                  # variable upper-bound settings: (var_idx, upper_bound)
    description: str

class BranchAndBoundSolver:
    def __init__(self, problem: MILPProblem, time_limit: float = 60.0, verbose: bool = False):
        self.problem = problem
        self.time_limit = time_limit
        self.verbose = verbose
        self.best_obj = math.inf
        self.best_x = None
        self.node_count = 0

    def _solve_lp(self, node: BnBNode) -> Optional[Tuple[float, np.ndarray]]:
        """
        Solve LP relaxation for problem plus node-specific variable bounds.
        Returns objective value and x if feasible, else None.
        """
        if not SCIPY_AVAILABLE:
            raise RuntimeError("scipy required for LP relaxations.")
        p = self.problem
        c = p.c.copy()
        A_ub = None if p.A_ub is None else p.A_ub.copy()
        b_ub = None if p.b_ub is None else p.b_ub.copy()
        A_eq = None if p.A_eq is None else p.A_eq.copy()
        b_eq = None if p.b_eq is None else p.b_eq.copy()
        bounds = [(0, None) if p.bounds is None else p.bounds[i] for i in range(len(c))]
        # apply node-specific variable bound modifications
        if node.lb_idx_constraints:
            for vi, lb in node.lb_idx_constraints:
                lo, hi = bounds[vi]
                bounds[vi] = (max(lo, lb), hi)
        if node.ub_bounds:
            for vi, ub in node.ub_bounds:
                lo, hi = bounds[vi]
                bounds[vi] = (lo, min(hi, ub) if hi is not None else ub)
        # Solve LP
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        if res.success:
            return float(res.fun), np.array(res.x)
        else:
            return None

    def _is_integral(self, x: np.ndarray, tol: float = 1e-8) -> bool:
        for i, t in enumerate(self.problem.var_types):
            if t == 'I':
                if abs(x[i] - round(x[i])) > tol:
                    return False
        return True

    def solve(self, max_nodes: int = 1000) -> Tuple[Optional[float], Optional[np.ndarray]]:
        """
        Iterative branch and bound:
         - Start with root node (no extra bounds)
         - Pop a node (LIFO stack, depth-first), solve LP relaxation
         - If infeasible or obj >= best_obj, prune
         - If solution integral -> update incumbent
         - Else pick most fractional integer var and branch on floor/ceil
        """
        stack: List[BnBNode] = [BnBNode(None, None, None, "root")]
        while stack and self.node_count < max_nodes:
            node = stack.pop()
            self.node_count += 1
            lp = self._solve_lp(node)
            if lp is None:
                if self.verbose:
                    print(f"Node {self.node_count} infeasible, pruned.")
                continue
            obj, x = lp
            if self.verbose:
                print(f"Node {self.node_count} LP obj={obj:.6f}")
            if obj >= self.best_obj - 1e-12:
                if self.verbose:
                    print("Pruned by bound (obj >= best).")
                continue
            # If integral, update incumbent
            if self._is_integral(x):
                if obj < self.best_obj:
                    self.best_obj = obj
                    self.best_x = x.copy()
                    if self.verbose:
                        print(f"New incumbent obj={obj:.6f}")
                continue
            # Otherwise branch on most fractional integer var
            frac_values = [(i, abs(x[i] - round(x[i]))) for i, t in enumerate(self.problem.var_types) if t == 'I']
            if not frac_values:
                # No integer vars -> treat as continuous optimization
                if obj < self.best_obj:
                    self.best_obj = obj
                    self.best_x = x.copy()
                continue
            # select most fractional
            vi, _ = max(frac_values, key=lambda p: p[1])
            xi = x[vi]
            floor_vi = math.floor(xi)
            ceil_vi = math.ceil(xi)
            # left branch: x_vi <= floor_vi
            left = BnBNode(None, node.lb_idx_constraints[:] if node.lb_idx_constraints else None,
                           node.ub_bounds[:] if node.ub_bounds else None,
                           node.description + f" | x[{vi}] <= {floor_vi}")
            # add upper bound for variable vi
            left_ub = [] if left.ub_bounds is None else left.ub_bounds.copy()
            if left_ub is None:
                left_ub = [(vi, floor_vi)]
            else:
                left_ub = (left.ub_bounds or []) + [(vi, floor_vi)]
            left.ub_bounds = left_ub
            # right branch: x_vi >= ceil_vi
            right = BnBNode(None, node.lb_idx_constraints[:] if node.lb_idx_constraints else None,
                            node.ub_bounds[:] if node.ub_bounds else None,
                            node.description + f" | x[{vi}] >= {ceil_vi}")
            right_lb = [] if right.lb_idx_constraints is None else right.lb_idx_constraints.copy()
            if right_lb is None:
                right_lb = [(vi, ceil_vi)]
            else:
                right_lb = (right.lb_idx_constraints or []) + [(vi, ceil_vi)]
            right.lb_idx_constraints = right_lb
            # push right then left (LIFO -> left explored first)
            stack.append(right)
            stack.append(left)
        return (self.best_obj if self.best_x is not None else None,
                (self.best_x.copy() if self.best_x is not None else None))


# -------------------------
# Small example: binary knapsack as MILP
# minimize negative profit (-profit^T x) s.t. weight constraint sum w_i x_i <= W, x_i in {0,1}
# -------------------------
if __name__ == "__main__":
    # Simple knapsack (maximize profit -> minimize negative profit)
    profits = np.array([10, 7, 6, 18, 3])
    weights = np.array([3, 2, 1, 4, 1])
    W = 7
    n = len(profits)
    c = -profits.astype(float)  # minimize -profit
    A_ub = weights.reshape(1, -1)
    b_ub = np.array([W], dtype=float)
    bounds = [(0.0, 1.0) for _ in range(n)]
    var_types = ['I'] * n
    problem = MILPProblem(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=None, b_eq=None, bounds=bounds, var_types=var_types)
    solver = BranchAndBoundSolver(problem, verbose=True)
    best_obj, best_x = solver.solve(max_nodes=200)
    if best_x is not None:
        print("Best obj:", best_obj, "solution x:", np.round(best_x))
        print("Profit:", -best_obj)
    else:
        print("No integer solution found.")
