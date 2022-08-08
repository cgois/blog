---
title: "CVXPY vs. PICOS: performance comparison for linear programs"
date: 2021-01-10T13:54:40-03:00
math: true
draft: false
summary: CVXPY and PICOS are two frequently used optimization program modelling tools for Python. I compare their performance in modelling simple random linear programs.
categories: [programming]
tags: [python, performance, optimization]
---

## Introduction to modelling APIs

Mathematical optimization is the ubiquitous discipline of searching extrema of functions. While well-behaved (e.g., differentiable) functions without too cumbersome constraints on their variables can be optimized via several available methods (e.g., gradient descent), it is usually hard to guarantee the solution will converge to a *global* extremum. Luckily, this can be done, in general, when optimizing convex functions over convex sets. The discipline of *convex optimization* can be further narrowed to several sub-problems, such as semidefinite programming (SDP) and linear programming (LP), which consist of optimizing a linear function over a spectrahedron or over a polytope, respectively. These are two essential and widespread tools in quantum information.

<img src="/convex_program_hierarchy.png" alt="convex hierarchy" width="550"/>

There are several important (and some very efficient) algorithms able to solve general and large instances of SDPs and LPs. Thanks to a considerable amount of both open-source and proprietary software (e.g. ECOS, CPLEX, Gurobi, MOSEK etc.), we can make good use of these methods black-box-wise. This is almost convenient, but not quite: each of these solvers have their own interfaces with which you must interact to tell them what problem you want to solve. Telling the solver about your problem is called *modelling*, and modelling can get quite cumbersome if the solver does not directly support some features of your model, such as complex variables (e.g. Gurobi's interface) and operations (partial trace, partial transpose etc.).

To our luck, several modelling languages were created. These are higher-level interfaces that turn our job of writing optimization problems into child's play. They do that by providing us with all common operations and converting everything to the solver input under the hood. To switch from one solver to the other then becomes a simple matter of giving a different argument to the modelling API.

Another great development in the last decades is that, thanks to NumPy and SciPy in general, Python became a reasonable choice for prototyping (and even carrying out) scientific computations. There are several tools for modelling optimization programs in Python, such as CVXPY, PICOS and Pyomo, among others. As the solvers themselves are stand-alone programs, it doesn't matter whether you choose Python or C as an interface. Or at least it shouldn't.

## Performance comparison

I was writing some reasonably large LPs with CVXPY and solving with Gurobi (a very good choice of solver for LPs with free academic licensing) when I noticed it shouldn't be taking *that* long to finish. I decided to profile my code and, to my surprise, 99% of its running time was spend inside CVXPY's functions, instead of in solving the problem itself. I was further surprised to see that, after switching to PICOS, the same problem got 99% of the time spent in the solver. To further compare these two tools, I wrote the script below. It generates random LPs and calls the same solver through PICOS and CVXPY, storing the running times.

```python
"""
Performance measurement for modelling LPs in PICOS vs CVXPY.

Random instances of LPs with varying numbers of variables and constraints
are generated and modelled in PICOS and CVXPY, then solved with Gurobi.

Time is measured for each instance, and a surface is plotted showing
nof. variables X nof. constraints X time to declare and solve the instance.
"""

import functools
import time

import numpy as np
import picos
import cvxopt
import cvxpy as cp
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt


CVX_SOLVER = cp.GUROBI
PICOS_SOLVER = "gurobi"
SOLVER_NAME = "GUROBI"

MIN_VARS, MAX_VARS, STEPS_VARS = 30, 500, 20
MIN_CONSTS, MAX_CONSTS, STEPS_CONSTS = 50, 8000, 100


def timeit(func):
    """Add running time of func to return vals."""

    @functools.wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        return result, end - start
    return timeit_wrapper


@timeit
def cvx_solve(A, b, c):
    """Minimize c.T * x subject to Ax = b using CVXPY."""

    x = cp.Variable(c.shape[0])
    prob = cp.Problem(cp.Minimize(c.T @ x), [A @ x <= b])
    return prob.solve(solver=CVX_SOLVER, verbose=False)


@timeit
def picos_solve(A, b, c):
    """Minimize c.T * x subject to Ax = b using PICOS."""

    prob = picos.Problem()
    A = picos.Constant('A', A)
    c = picos.Constant('c', c)
    b = picos.Constant('b', b)
    x = picos.RealVariable('x', c.shape[0])
    prob.add_constraint(A * x <= b)
    prob.set_objective('min', c.T * x)
    prob.solve(solver=PICOS_SOLVER, verbosity=-1)
    return prob


def gen_instance(nv, nc):
    """Random nontrivial LP instance with nv variables and nc constraints.
    Ref.: https://www.cvxpy.org/examples/basic/linear_program.html
    """

    sz = np.random.randn(nc)
    lz = np.maximum(-sz, 0)
    sz = np.maximum(sz, 0)
    xz = np.random.randn(nv)
    A = np.random.randn(nc, nv)
    return A, A @ xz + sz, -A.T @ lz


def plot_performance_surface(data, label, ax=None):
    """Plot surface of time taken for each nof. variables and constrainsts."""

    if ax is None:
        fig = plt.figure(figsize=(9, 6))
        ax = plt.axes(projection='3d')
        ax.view_init(elev=15, azim=-45)

        plt.title(f"Performance for LPs in PICOS and CVXPY using {SOLVER_NAME}")
        ax.set_xlabel("Nof. variables")
        ax.set_ylabel("Nof. constraints")
        ax.set_zlabel("Time (s)")

    nvars, nconsts, times = data[:, 0], data[:, 1], data[:, 2]
    surf = ax.plot_trisurf(nvars, nconsts, times, label=label,
                           shade=True, antialiased=True, alpha=.7)
    surf._facecolors2d =surf._facecolor3d   # Hacks for labels to work
    surf._edgecolors2d = surf._edgecolor3d
    ax.legend()
    return ax


if __name__ == '__main__':

    VARS = np.linspace(MIN_VARS, MAX_VARS, STEPS_VARS, dtype=int)
    CONSTS = np.linspace(MIN_CONSTS, MAX_CONSTS, STEPS_CONSTS, dtype=int)
    N_PROBS = len(VARS) * len(CONSTS)

    pic, cvx, count = [], [], 0
    for nv in VARS:
        for nc in CONSTS:
            A, b, c = gen_instance(nv, nc)

            pic_res, tp = picos_solve(A, b, c)
            pic.append([nv, nc, tp])

            cvx_res, tc = cvx_solve(A, b, c)
            cvx.append([nv, nc, tc])

            count = count + 1

            # Just to check progress and validate:
            print(f'{100 * count / N_PROBS:.2f}% done. '
                  f'/ Last time: PICOS = {tp:.2f}s, CVXPY = {tc:.2f}s '
                  f'/ PICOS_VAL = {pic_res.value:.2f}, CVX_VAL = {cvx_res:.2f}\r', end='')

        # Save partial results (will overwrite!).
        np.save('picos_performance.npy', np.array(pic))
        np.save('cvxpy_performance.npy', np.array(cvx))

    ax = plot_performance_surface(np.array(pic), "PICOS")
    plot_performance_surface(np.array(cvx), "CVXPY", ax)
    plt.savefig('performance_surface.png')
```

Before I show you the results, I must make some observations:
- I plot the combined time of modelling and solving. As the same solver was used in both instances, The performance being compared should be of the modelling APIs themselves.
- I do not know if the case would be different for other types of random LPs, and it surely would for other types optimization problems, like SDPs. If you know of some comparison for SDPs please let me know.
- These were computed with **PICOS 2.0** and **CVXPY 1.1** (around July 2020). Both modules are under very active development, and the results may be very different some time onwards. I'll run it again for the newer versions sometime and update the post.

![gurobi old comparison](/cvxpy_vs_picos_gurobi_old.png)

With these results in mind, I switched to PICOS as of July 2020. Another reason for that was that it had some useful operations that CVXPY didn't have at the time (such as partial tracing). I thoroughly recommend both of them, but I will soon have another peek at the new tools in CVXPY to see how it is going.

## (!) Updated comparison for new versions

Looks like tables have turned in new versions! The following plot compares **PICOS 2.1.2** with **CVXPY 1.1.10**:

![gurobi new comparison](/cvxpy_vs_picos_gurobi_new.png)

As I observe, CVXPY was around $1.6$ times faster than PICOS for large random LPs (and even faster for small ones). It still doesn't have partial tracing implemented though, which is very useful when dealing with SDPs in quantum information applications.

## (!) Still more updates... (Jun. 2022)

New versions are around, and some further comparison shows we have a nice race. **PICOS 2.4.1** was now faster than **CVXPY 1.2.1** in $1422$ of $2000$ random instances of these LPs. The largest time difference were an instance where PICOS was $5.2s$ faster, and one where CVXPY was $6.5s$ faster. Also worth mentioning, since version 1.2.0, CVXPY has nice support for partial tracing and partial transposes.

As before, bear in mind that these results are valid for these particular types of sampled instances of LPs (dense constraint matrices etc.), and let me know if you notice any unfairnesses in the comparisons (:

![gurobi new new comparison](/cvxpy_vs_picos_gurobi_new_new.png)
