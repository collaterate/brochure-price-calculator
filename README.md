# Brochure Price Calculator V2

[![PyPI license](https://img.shields.io/pypi/l/ansicolortags.svg)](https://pypi.python.org/pypi/ansicolortags/)
[![Python 3.11 | 3.12 | 3.13](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-orange
)](https://www.python.org/downloads/)
[![Smartpress](https://img.shields.io/badge/Smartpress-blue?style=for-the-badge&logo=surrealdb)](https://smartpress.com/)
<!--- ![Smartpress](https://img.shields.io/badge/Smartpress-blue.svg?logo=data:image/svg%2bxml;base64,) --->

This repository contains price calculator functionality for the new, designer-optimized brochure landing page.

The current framework uses a baseline brochure price (determined by market value analysis of our current default brochure configuration), applies press sheet and ink type adjustments, and adds cost-plus-markup operations prices to arrive at a final price. Since prices that are drastically different than those currently listed on SPDC would be disruptive, the model checks its output prices against legacy prices to ensure that they are "in the same ballpark" (and adjusts its outputs if necessary).

Moving forward, this model provides the flexibility to set prices for specific configurations independent of underlying productions costs or markups.


## Prerequisites

The active environment must have:
<!--- These are just example requirements. Add, duplicate or remove as required --->
* [Python 3.11](https://www.python.org/) or later installed.
* [Numpy 1.20](https://numpy.org/) or later installed.
* [pandas 2.0.0](https://pandas.pydata.org/) or later installed.

The brochure price calculator must be in the same directory as the script, with the file name BrochurePriceCalculator.pkl.

## Installation

To install Python, follow the instructions at https://www.python.org/downloads/. After installing Python, run the command

```
pip install numpy pandas
```
from the terminal, command line, or powershell to install the requisite libraries.

## Usage

The calculator is run from the command line (within the appropriate working directory). The command line syntax is
```
python brochure_price_calculator_v2.py [-h] [-q] [-v] [-x] [-d] [-w] [-l] [-t] [-y] [-c] [-p] [-f] [-g] [-i] [-j] [-o]

Calculate Brochure price

options:
  -h, --help                    show this help message and exit
  -q, --quantity                quantity ordered (Default: 100)
  -v, --versions                versions ordered (Default: 1)
  -x, --quantity-per-version    quantity per version (Default: None)
  -d, --preset-dimensions       preset dimensions ID (Default: None)
  -w, --finished-width          finished width of the Brochure (Default: 8.5)
  -l, --finished-height         finished height of the Brochure (Default: 11.0)
  -t, --press-sheet-type        press sheet type (Default: "Coated Matte - White")
  -y, --press-sheet-weight      press sheet weight (Default: "100# Text")
  -c, --press-sheet-color       press sheet color (Default: "White")
  -p, --press-sheet             press sheet ID (Default: None)
  -f, --side1-ink-type          front ink type to use (Default: "Full Color")
  -g, --side1-ink               front ink ID (Default: None)
  -i, --side2-ink-type          back ink type to use (Default: None)
  -j, --side2-ink               back ink ID (Default: None)
  -o, --operations              add-on operations and operations items in "OPERATION"="OPERATION ITEM,ANSWER" or OPERATION_ID=OPERATION_ITEM_ID,ANSWER format
```

As an example:
```
python brochure_price_calculator_v2.py --quantity 25 --finished-width 8.5 --finished-height 11.0 --press-sheet-type "Coated Matte - White" --press-sheet-weight "100# Text" --press-sheet-color "White" --side1-ink "Full Color" --side2-ink "Full Color" --operations "Folding"="Single Fold" "Scoring-Only"="Parallel Scores" "Shrink Wrap"="Bundles,5"
```

Or, using IDs:
```
python brochure_price_calculator_v2.py --quantity 25 --preset-dimensions 814 --press-sheet 894 --side1-ink 146 --side2-ink 146 --operations 1=34 2=4 14=42,5
```

Syntax and command line functionality can be modified upon request.

## Contact

Please send questions or feedback to <ryan.corkrean@thebernardgroup.com>.
