---
title: "panda2lrs: user-friendly polytope representation in lrs"
date: 2020-10-04T02:46:32-03:00
math: true
draft: false
summary: An user-friendly way to write *PANDA*-like polytope inequalities and run in *lrs*.
categories: [tools]
tags: [inequalities, vertex enumeration, polytopes]
---

## Introduction
Both [*PANDA*](http://comopt.ifi.uni-heidelberg.de/software/PANDA/) and [*lrs*](http://cgm.cs.mcgill.ca/~avis/C/lrs.html) (or, for that matter, also [*PORTA*](http://porta.zib.de/) and [*cdd*](https://github.com/cddlib/cddlib)) are useful tools to solve vertex enumeration and convex hull problems, which are common tasks in computational geometry and quantum foundations. Apart from implementing different algorithms, they also differ in their input specifications, and when dealing with inequalities, *PANDA* is vastly more user-friendly than *lrs*. Take, for instance, this cube: 
```
Names
x y z
Inequalities
x >= 0
y >= 0
z >= 0
x <= 1
y <= 1
z <= 1
```

If we were to convert it to an *lrs* input file, we'd write

```
H-representation
begin
6 4 rational
0 1 0 0
0 0 1 0
0 0 0 1
1 -1 0 0
1 0 -1 0
1 0 0 -1
end
```
as lrs expects inequalities of the form $a_0 + a_1 x_1 + \ldots + a_{n-1} x_{n-1} \geq 0$. When writing larger polytopes by hand, this quickly becomes unreadable and error-prone, so I made a parser to convert a *PANDA*-like H-representation file to an *lrs* one.

## File specifications

Any input to the parser should have a *Names* section with variables names separated by a blank space, an *Inequalities* section, and may optionally have *Equations*. Any constants must be on the RHS of any expression. Hence, the first snippet would be valid, and this one also would:

```
Names
A1 A2 B1 B2
Equations
A1 + A2 = 0
B1 + B2 = 1
Inequalities
A1 - 2A2 >= 0
B1 >= 0
B2 <= 0
```

Calling [panda2lrs.py](https://github.com/cgois/misc/blob/main/panda2lrs.py) on this input would readily output an lrs readable file, with each equation turned into two inequalities, and each inequality manipulated to what *lrs* expects. If you don't care for implementation details and just wanna know how to use it, you can skip the next section and see the usage details at the end.

## *Implementation details
From *lrs*'s user guide, we know that any H-representation input must have the structure

```
H-representation
begin
m n rational

{list of inequalities}

end
```
where each inequality is of the form $a_0 + a_1 x_1 + \ldots + a_{n-1} x_{n-1} \geq 0$. In order to convert, I first read the input file and put each segment in a list. I then store the index where each section (which are *Names*, *Equations* and *Inequalities*) start, and find out what variables are listed under *Names*.

```python
import re

SECTIONS = r"(Names|Equations|Inequalities|\n)"
KEYWORDS = ["Names", "Equations", "Inequalities"]

with open(fname, 'r') as file:
    raw = re.split(SECTIONS, file.read())
# Filter any newline and empty elements left over by splitting.
expressions = list(filter(None, filter((lambda x: x != "\n"), raw)))

# Find starting indexes for each section.
indexes = {}
for kword in KEYWORDS:
    try:
        indexes[kword] = expressions.index(kword)
    except ValueError: # Some sections are not mandatory...
        pass

# Parse all variables' names.
try:
    NAMES = expressions[indexes["Names"] + 1].split()
except KeyError: # Reraise with more information.
    raise KeyError("Missing variables names in input file.")
```

Preprocessing done, I now read the equations one by one, and send them to my expression parser. The parser will then take the equality string and the `NAMES` list, extract the coefficients from the expression as numbers (and put a $0$ for the missing terms), and return a list with them. This list has the same ordering as the `NAMES` specification (see the whole script for the details on the parser function).
```python
# Extract equations' coefficients and turn them into two inequalities.
equations = []
if "Equations" in indexes.keys():
    end = indexes["Inequalities"] if "Inequalities" in indexes.keys() else -1

    for expression in expressions[indexes["Equations"] + 1:end]:
        equations.append(parse_expression(expression, NAMES, "="))
        equations.append(list(map(lambda x: -x, equations[-1])))
```
Note that *lrs* doesn't support equalities, so I turn each input equality into two inequalities in the last line.

The following step is essentially the same, but now parsing the *in*equalities:
```python
# Extract inequalities' coefficients.
inequalities = []
if "Inequalities" in indexes.keys():
    inequalities = [parse_expression(expression, NAMES, "<=|>=")
                    for expression in expressions[indexes["Inequalities"] + 1:]]
```

To wrap everything up, I concatenate these lists and format them in a way that *lrs* will understand by asking
```python
lrs = f"H-representation\nbegin\n{len(equations + inequalities)} {len(NAMES) + 1} rational\n"
lrs += "\n".join([" ".join(map(str, expr)) for expr in equations + inequalities])
```

All this done to the example from the last section will result in the *lrs*-ready output
```
H-representation
begin
7 5 rational
0 1 1 0 0
0 -1 -1 0 0
-1 0 0 1 1
1 0 0 -1 -1
0 1 -2 0 0
0 0 0 1 0
0 0 0 0 -1
end
```

## Usage

You can find `panda2lrs.py` [in here](https://github.com/cgois/misc/blob/main/panda2lrs.py). Given an input with the aforementioned specifications, running this is quite simple: call `python panda2lrs.py input` to print the results to the screen, or `python panda2lrs.py input output` to directly write to a file named `output` **(if a file with the same name already exists, it'll be overwritten!)**. You can then feed `output` to lrs and watch it search your vertices.

I still haven't used it much, so there may be some bugs. Please let me know if you find any issues (:
