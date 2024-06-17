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

## Usage

The calculator is run from the command line (within the appropriate working directory). The command line syntax is
```
python brochure_price_calculator.py [-h] [-q] [-w] [-l] [-t] [-p] [-f] [-b] [-o]

options:
  -h, --help            show this help message and exit
  -q, --quantity        quantity ordered (Default: 100)
  -w, --width           finished width of the Brochure (Default: 8.5)
  -l, --height          finished height of the Brochure (Default: 11.0)
  -t, --sheet-type      paper stock type (Default: "Coated Matte - White")
  -p, --sheet-weight    paper stock weight (Default: "100# Text")
  -f, --front-ink       front ink to use (Default: "Full Color")
  -b, --back-ink        back ink to use (Default: None)
  -o, --operations      add-on operations and operations items in "OPERATION"="OPERATION ITEM" format
```

As an example:
```
python brochure_price_calculator.py --quantity 25 --width 8.5 --height 11.0 --sheet-type "Coated Matte - White" --sheet-weight "100# Text" --front-ink "Full Color" --back-ink "Full Color" --operations "Folding"="Single Fold" "Scoring-Only"="Parallel Scores" "Shrink Wrap"="Bundles;5"
```

Syntax and command line functionality can be modified upon request.

## Contact

Please send questions or feedback to <ryan.corkrean@thebernardgroup.com>.
