---
title: "Parallelizing simple loops in Python"
date: 2020-09-26T13:54:40-03:00
math: true
draft: false
summary: Parallelizing loops with no iteration interdependence using Pythons' concurrent.futures module.
categories: [programming]
tags: [python, performance, parallelization]
---

Algorithms vary drastically on how easily parallelizable they are. Newton's method, for one, is an inherently serial method in which every interation relies on the result of the last one, while Monte Carlo methods can be largely parallelized, as they rely on identical computations over several independent inputs. We call algorithms that can easily be separated into parallel parts *embarrasingly parallel*, and with tasks that do not depend on state change (like updating a variable for some next iteration to use), this is usually the case.

In science, lengthy computations can sometimes be hugely optimized through parallelization. In particular, *parameters space study* is a very common task that can easily be parallelized. This type of task can be seen as a black-box algorithm applied to several different inputs, whose outputs will then be collected and analyzed. Let $f$ be our black-box and $x$ its input. Then, $(x, f(x)), \forall x \in \mathcal{X}$, for a discrete but possibly huge set $\mathcal{X}$ of input parameters, is an embarrassingly parallel task, as we can easily distribute the computation of $f(\mathcal{X})$ to $W$ workers that will each compute a given $x$ independently of the others. Given adequate computational resources, doing so obviously leads to a huge speedup in your parameter space study (though not exactly a $W$-fold speedup, as there may be overheads in parallelization), luckily with very little modification to existing code.

Python's standard library includes several packages to streamline parallel execution of code, such as *threading*, *multiprocessing* and *concurrent*. The *concurrent.futures* package provides a very simple, high-level interface to distribute computations to a set number of workers. It doesn't provide as much control over the processes as, say, *multiprocessing*, but it's much simpler and should be our pick as long as it's sufficient for the task at hand. Speaking of which, let's take evaluating the extremely hard to compute $f(x)=x^2$ function as our soon to be parallelized task. First, import the modules we'll need and define the function

```python
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

def f(x): 
  time.sleep(2) # Let's pretend we have a hard task at hand
  return x ** 2
```

You can then get a list of twenty $(x, f(x))$ pairs equally spaced in $[-1,1]$ by saying

```python
x2pairs = [(x, f(x)) for x in np.linspace(-1, 1, 20)]
```

This will, of course, take around forty seconds, as each query to $f$ takes two seconds (you may even [timeit](https://docs.python.org/3.8/library/timeit.html) if you'd like). But we can clearly see this loop has no iteration interdependence at all, as it has no state changes. This means it'll be extremely simple to parallelize it if we rewrite it as a mapping.

[Sometimes things are not so simple, though. It's common to find poorly written loops that *do* have state changes in between iterations, but that *needn't* have (which probably also means it *shouldn't* be updating variables all over the place). In those cases, some sections of your code may need to be rewritten to be amenable to mapping. You'll see why in a bit.]

Python supports some functional programming tools, and the easiest way to spawn parallel processes with *concurrent.futures* is to use a *ProcessPoolExecutor* and its *map* method. Mapping refers to applying a transformation to a list of objects. The following example applies $f$ to each element in the sequence given as the second argument to *map*, and will put $f(\mathcal{X})$ into `yvals`.

```python
xvals = np.linspace(-1, 1, 20)
yvals = map(f, xvals)
```

[Note this will return you an *iterator*, and not straightly the list of $f(x)$'s. To get the list you must iterate over the mapping with `list(yvals)`]

While `map` is still serially applying $f$, it's trivially parallelizable. It all boils down to creating the *executor* with as many workers as you'd like (this should be the at most the number of threads supported by your CPU), and applying its own *map* method to the values we want to compute. You can either set the number of workers manually, or get `cpu_count`automatically:

```python
xvals = np.linspace(-1, 1, 20)
with ProcessPoolExecutor(cpu_count()) as executor: # Create the executor
  yvals = list(executor.map(f, xvals)) # Do the job and put it in a list
```

[The `list` call around `executor.map` is just to cast the result as a list. Otherwise, you'll get a generator. You also do not strictly need the `with` structure to instantiate an executor, but it's cleaner and safer this way.] 

Try it and see how faster it goes. On my computer, with four threads, it was around four times faster than serial execution. For fun, you can plot the results to see it actually works:

```python
import matplotlib.pyplot as plt

plt.scatter(xvals, yvals)
plt.show()
```
![result plot](/tutorial_processpool_plot.png#center)

This is pretty much all you need to do when parallelizing simple tasks. But for even more fun, you can use `tqdm` to include a progress bar to your executor (you may need to `pip install tqdm`):

```python
from tqdm import tqdm

with ProcessPoolExecutor(cpu_count()) as executor: # Same as before
  yvals = list(tqdm(executor.map(f, xvals), total=len(xvals))) # Include the progress bar
```
![tqdm progress bar](/tutorial_processpool_tqdm.png#center)

The progress bar may actually be useful when you have a huge computation to run and want to check its progress along the way. But do not trust the time estimation, as it'll only be precise when the function applications take the same time for any parameter, which is not always the case. The `total` parameter, telling `tqdm` how many elements we'll be mapping onto, is not mandatory, but without it you'll only get an iteration count, without the pretty progress bar.

All this was done to an artificially simple problem. For actual use cases, you'll possibly have to adapt you existing code's structure to be able to use this method of parallelization, but it shouldn't be hard to do so.

There are also other useful patterns when parallelizing with concurrent.futures, such as using `functools.partial` to partially apply arguments to a function before calling `executor.map` on the sequence of parameters. This is very useful if your function takes more than one parameter and you want to set some constant ones before mapping. In case you don't wanna define a whole new function, `itertools.repeat` would also be a good choice in these cases, as it will repeat any argument given to it indefinitely and lazily. Here is an example in which we'll generate a paraboloid like $f(x, y) = a (x^2 + y^2)$, for constant $a$.

```python
from itertools import repeat
from mpl_toolkits.mplot3d import Axes3D

def f(a, x, y):
  return a * (x ** 2 + y ** 2)

# Generate a square grid with 20 points in each axis
xvals, yvals = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))

with ProcessPoolExecutor(cpu_count()) as executor:
  zvals = list(executor.map(f, repeat(-1), xvals, yvals)) # Set a = -1

# Build a surface plot from our data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xvals, yvals, np.asarray(zvals))
plt.show()
```
![paraboloid surface](/tutorial_processpool_surface.png#center)

For more usage examples and details on the features, check [the documentation on *concurrent.futures*](https://docs.python.org/3.8/library/concurrent.futures.html#module-concurrent.futures). And if ever *concurrent.futures* is not enough and you need more control over your parallelization tasks, check [the concurrent execution chapter of the documentation](https://docs.python.org/3.8/library/concurrency.html), which describes other useful modules. Keep in mind, though, that because of [Python's GIL](https://realpython.com/python-gil/), using thread-based parallelization is useless for computationally intensive tasks. Always pick process-based paralelization in these cases.