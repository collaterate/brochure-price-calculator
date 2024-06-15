# Brochure Price Calculator

[![PyPI license](https://img.shields.io/pypi/l/ansicolortags.svg)](https://pypi.python.org/pypi/ansicolortags/)
[![Python 3.10 | 3.11 | 3.12](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-orange
)](https://www.python.org/downloads/release/python-310/)
[![Smartpress](https://img.shields.io/badge/Smartpress-blue?style=for-the-badge&logo=surrealdb)](https://smartpress.com/)
<!--- ![Smartpress](https://img.shields.io/badge/Smartpress-blue.svg?logo=data:image/svg%2bxml;base64,) --->

This repository contains price calculator functionality for the new, designer-optimized brochure landing page.

The current framework uses a baseline brochure price (determined by market value analysis of our current default brochure configuration), applies press sheet and ink type adjustments, and adds cost-plus-markup operations prices to arrive at a final price. Since prices that are drastically different than those currently listed on SPDC would be disruptive, the model checks its output prices against legacy prices to ensure that they are "in the same ballpark" (and adjusts its outputs if necessary).

Moving forward, this model provides the flexibility to set prices for specific configurations independent of underlying productions costs or markups.


## Prerequisites

The active environment must have installed:
<!--- These are just example requirements. Add, duplicate or remove as required --->
* [Python 3.10](https://www.python.org/) or later.
* [Numpy 1.20](https://numpy.org/) or later.
* [pandas 2.0.0](https://pandas.pydata.org/) or later.
* [rectpack 0.2.2](https://github.com/secnot/rectpack) or later.
* [SciPy 1.13.0](https://scipy.org/) or later.
* The five supplemental CSV and XLSX files in this repository, stored in the same directory as the main PY file.

## Installation

To install Python, follow the instructions at https://www.python.org/downloads/. After installing Python, run the command

```
pip install numpy=1.26.4 pandas=2.2.2 rectpack=0.2.2 scipy=1.13.1
```
from the terminal, command line, or powershell to install the requisite libraries.

## Using <project_name>

The calculator is run from the command line (within the appropriate working directory). For information on command line syntax, run

```
python brochure_price_calculator.py -h
```

## Contact

Please send questions or feedback to <ryan.corkrean@thebernardgroup.com>.
