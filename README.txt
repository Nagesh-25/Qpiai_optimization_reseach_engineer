
QpiAI_Assessment Code Package
=============================

This package contains three Python skeletons for the Optimization Research Engineer Round 1 assessment.

Folders:
 - Problem1_MaxCut: QUBO mapping + Simulated Annealing + local search. Produces results/problem1_results.csv and plots.
 - Problem2_VRPTW: Synthetic dataset generator + OR-Tools VRPTW solver example + route plotting. Produces results/problem2_dataset.csv and problem2_routes.png when OR-Tools is installed.
 - Problem3_CustomMILP: Simple branch-and-bound MILP prototype using scipy.linprog for LP relaxations. Demonstrates on a small knapsack example.

How to run:
 1. Create and activate a virtual environment.
 2. Install required packages depending on problems you want to run. Example:
    pip install numpy matplotlib networkx scipy ortools
 3. Run individual scripts, e.g.:
    python Problem1_MaxCut.py
    python Problem2_VRPTW.py
    python Problem3_CustomMILP.py

Notes:
 - The scripts include checks and helpful messages if libraries (networkx, ortools, scipy) are missing.
 - They are intended as a starting point. Tune hyperparameters, increase SA steps, and run on real datasets (GSET, TSPLIB/CVRPLIB) for the final submission.
