import argparse
import json
import os
import pickle
import sys
import warnings
from collections import defaultdict
from functools import partial
from math import ceil
from pathlib import Path
from typing import Any, Callable, Self, Sequence, TypeVar

import numpy as np
import pandas as pd
try:
    from dotenv import load_dotenv
    from jinja2 import Template
    from sqlalchemy import create_engine, text, Engine, URL
except ImportError:
    pass

M = TypeVar('M', float, Sequence[float])
Y = TypeVar('Y', float, str, dict[str, str], None)

SOSS_ID = 10

SF_PRINTING_OPERATIONS = ['finished_width', 'finished_height', 'pages',
                          'press_sheet_type', 'press_sheet_weight',
                          'press_sheet_color', 'side1_ink_type',
                          'side2_ink_type', 'cover_press_sheet_type',
                          'cover_press_sheet_weight', 'cover_press_sheet_color',
                          'cover_side1_ink_type', 'cover_side2_ink_type']
LF_PRINTING_OPERATIONS = ['final_width', 'final_height', 'sides',
                          'print_substrate', 'front_laminate', 'back_laminate',
                          'mount_substrate']
OI_LF_PRINTING_OPERATIONS = ['final_width', 'final_height', 'sides',
                             'print_substrate_a', 'print_substrate_b',
                             'front_laminate', 'back_laminate',
                             'mount_substrate']

BASE_DIR = Path(__file__).resolve().parents[0]
CALC_PATH = filepath = BASE_DIR / 'BrochurePriceCalculator.pkl'
ENV_PATH = BASE_DIR / '.env'
QUERY_DIR = BASE_DIR / 'queries'

if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

    CONN_PARAMS = {'drivername': 'redshift+psycopg2',
                   'host': os.getenv('REDSHIFT_HOST'),
                   'port': os.getenv('REDSHIFT_PORT'),
                   'database': os.getenv('REDSHIFT_DATABASE'),
                   'username': os.getenv('REDSHIFT_USERNAME'),
                   'password': os.getenv('REDSHIFT_PASSWORD')}

pd.set_option('future.no_silent_downcasting', True)
warnings.filterwarnings('ignore')


def redshift_engine() -> Engine:
    """Creates a SQLAlchemy engine configured to the Redshift database.

    :return: The SQLAlchemy engine.
    :rtype: Engine
    """
    return create_engine(
        URL.create(**CONN_PARAMS),
        connect_args={'sslmode': 'prefer'}
    )


def get_default_configs(
    engine: Engine,
    /,
    *system_offering_site_share_ids: int | None
) -> dict[str, dict[int, dict[str, Y]]]:
    """Retrieves the default configuration(s) for the provided system offering
    site shares.

    This function queries the database to fetch the default configuration
    details for a set of SOSS IDs. Each configuration includes various default
    settings for print job specifications, such as press sheet type, ink type,
    and default dimensions. Configurations are returned as a dictionary, where
    the keys are possible product formats ('SMALL_FORMAT' or 'LARGE_FORMAT') and
    the values are dicts where the keys are system offering site share IDs and
    the values are the corresponding configurations as dictionaries of
    (configuration parameter, value) key, value pairs.

    :param engine: SQLAlchemy engine for database connectivity.
    :type engine: Engine
    :param system_offering_site_share_ids: The IDs of the system offering site
        share for which the default configuration is requested.
    :type system_offering_site_share_ids: int
    :returns: A dictionary of small format and large format configuration
        sub-dictionaries.
    :rtype: dict[str, dict[int, dict[str, Y]]]

    :example:

    >>> default_configs = get_default_configs(engine, 123)
    >>> print(default_configs)
    {
        'SMALL_FORMAT': {
            123: {
                'quantity': 100,
                'device': 'Small Format B2',
                'finished_width': 8.5,
                'finished_height': 11.0,
                'pages': 1,
                'press_sheet_type': 'Coated Matte - White',
                'press_sheet_weight': '80# Text',
                'press_sheet_color': 'White',
                'side1_ink_type': 'Full Color',
                'side2_ink_type': None,
                'cover_press_sheet_type': None,
                'cover_press_sheet_weight': None,
                'cover_press_sheet_color': None,
                'cover_side1_ink_type': None,
                'cover_side2_ink_type': None,
                'operations': {
                    'Coating - Standard': {
                        'heading': Coating,
                        'item': 'Ultra Gloss UV Coating - 2 sides',
                        'answer': None
                    },
                    'Shrink Wrap': {
                        'heading': 'Shrink-wrapping service',
                        'item': 'Bundles',
                        'answer': 10
                    }
                },
                'turnaround_time': '2 Business Days',
                'proof': 'Soft Proof'
            }
        },
        'LARGE_FORMAT': {}
    }
    """
    soss_condition = (
        ('soss.id IN ('
         + ', '.join(map(str, system_offering_site_share_ids)) + ')')
        if system_offering_site_share_ids
        else 'TRUE'
    )

    query_path = QUERY_DIR / 'get_default_configs.sql'

    with (open(query_path, 'r', encoding='utf-8') as file,
          engine.begin() as connection):
        data = pd.read_sql_query(
            sql=text(
                Template(file.read()).render(
                    soss_condition=soss_condition
                )
            ),
            con=connection,
            index_col='system_offering_site_share_id'
        )

    # Parse operations.
    data['operations'] = data['operations'].apply(json.loads)

    # Partition configurations by product format, drop irrelevant columns, and
    # refactor DataFrames as dictionaries.
    sf = (data[data['pjc_format'] == 'SMALL_FORMAT']
          .drop(columns=['pjc_format'] + LF_PRINTING_OPERATIONS)
          .replace(np.nan, None)
          .astype({col: int for col in ['quantity', 'versions',
                                        'quantity_per_version', 'pages']})
          .to_dict(orient='index'))
    lf = (data[data['pjc_format'] == 'LARGE_FORMAT']
          .drop(columns=['pjc_format'] + SF_PRINTING_OPERATIONS)
          .replace(np.nan, None)
          .astype({col: int for col in ['quantity', 'versions',
                                        'quantity_per_version', 'sides']})
          .to_dict(orient='index'))

    # Return small format and large format dictionaries in JSON-like format.
    return sf | lf


def _calculate_pieces_per_sheet(
    piece_width: float,
    piece_height: float,
    press_sheet_width: float = 29.4375,
    press_sheet_height: float = 20.75,
    left_margin: float = 0.0,
    right_margin: float = 0.0,
    top_margin: float = 0.0,
    bottom_margin: float = 0.0,
    default_piece_margin: float = 0.125,
    default_piece_bleed: float = 0.0,
    max_number_up_on_sheet: int = 0,
    override_number_up_on_sheet: int = 0,
    join_type: str | None = None
) -> int:
    """Calculates the maximum number of pieces that can fit on a press sheet.

    This function determines how many individual pieces can be placed on a given
    press sheet while considering margins, bleed, and optional joining of
    pieces. It takes  top edge and adjusts their dimensions accordingly. If a
    maximum or override number of pieces is specified, the function ensures that
    the final count adheres to these constraints.

    :param piece_width: Width of a single piece before any modifications.
    :type piece_width: float
    :param piece_height: Height of a single piece before any modifications.
    :type piece_height: float
    :param press_sheet_width: Width of the press sheet. Defaults to 29.4375.
    :type press_sheet_width: float, optional
    :param press_sheet_height: Height of the press sheet. Defaults to 20.75.
    :type press_sheet_height: float, optional
    :param left_margin: Left margin of the press sheet. Defaults to 0.0.
    :type left_margin: float, optional
    :param right_margin: Right margin of the press sheet. Defaults to 0.0.
    :type right_margin: float, optional
    :param top_margin: Top margin of the press sheet. Defaults to 0.0.
    :type top_margin: float, optional
    :param bottom_margin: Bottom margin of the press sheet. Defaults to 0.0.
    :type bottom_margin: float, optional
    :param default_piece_margin: Default margin around each piece. Defaults to
        0.125.
    :type default_piece_margin: float, optional
    :param default_piece_bleed: Default bleed applied to each piece. Defaults to
        0.0.
    :type default_piece_bleed: float, optional
    :param max_number_up_on_sheet: Maximum allowed number of pieces on a sheet.
        Defaults to 0.
    :type max_number_up_on_sheet: int, optional
    :param override_number_up_on_sheet: If set, forces the number of pieces on
        the sheet. Defaults to 0.
    :type override_number_up_on_sheet: int, optional
    :param join_type: Specifies whether pieces should be joined along the left
        or top edge. Valid values are "Left" and "Top". Defaults to None.
    :type join_type: str | None, optional
    :returns: The maximum number of pieces that can be placed on the press
        sheet.
    :rtype: int

    :example:

    >>> _calculate_pieces_per_sheet(4.0, 6.0)
    20

    >>> _calculate_pieces_per_sheet(4.0, 6.0, join_type='Left')
    10

    >>> _calculate_pieces_per_sheet(4.0, 6.0, max_number_up_on_sheet=8)
    8
    """
    if join_type == 'Left':
        piece_width *= 2.0
    elif join_type == 'Top':
        piece_height *= 2.0

    actual_edge = max(default_piece_margin, default_piece_bleed)
    piece_width += actual_edge * 2
    piece_height += actual_edge * 2

    try:
        printable_width = press_sheet_width - (left_margin + right_margin)
        printable_height = press_sheet_height - (top_margin + bottom_margin)
    except TypeError:
        print()

    # If saddle stitching equipment is used, the minimum piece size is
    # 8.0" x 8.0".
    if join_type:
        piece_width = max(piece_width, 8.0)
        piece_height = max(piece_height, 8.0)

    # Determine an orientation.
    width_count_normal = printable_width / piece_width
    height_count_normal = printable_height / piece_height

    width_count_flipped = printable_width / piece_height
    height_count_flipped = printable_height / piece_width

    vertical_orientation = (int(width_count_normal) * int(height_count_normal)
                            >= (int(width_count_flipped)
                                * int(height_count_flipped)))

    if ((width_count_normal < 1 or height_count_normal < 1)
            and (width_count_flipped < 1 or height_count_flipped < 1)):
        piece_is_wider_than_tall = piece_width >= piece_height
        sheet_is_wider_than_tall = printable_width >= printable_height
        normal_orientation = (piece_is_wider_than_tall
                              == sheet_is_wider_than_tall)
        max_printable_width = (printable_width
                               if normal_orientation
                               else printable_height)
        max_printable_height = (printable_height
                                if normal_orientation
                                else printable_width)

        joined_piece_size = actual_edge * 2.0

        max_printable_width -= joined_piece_size
        max_printable_height -= joined_piece_size

        if join_type == 'Left':
            max_printable_width /= 2
        elif join_type:
            max_printable_height /= 2

    number_across = (int(width_count_normal)
                     if vertical_orientation
                     else int(width_count_flipped))
    number_down = (int(height_count_normal)
                   if vertical_orientation
                   else int(height_count_flipped))
    number_up = number_across * number_down

    if override_number_up_on_sheet > 0:
        number_up = override_number_up_on_sheet
    elif 0 < max_number_up_on_sheet < number_up:
        number_up = max_number_up_on_sheet

    return number_up


def _calculate_num_press_sheets_and_pieces(
    quantity: int,
    pages: int,
    finished_width: float,
    finished_height: float,
    press_sheet_data: pd.DataFrame,
    device_data: pd.DataFrame,
    default_piece_margin: float = 0.125,
    default_piece_bleed: float = 0.0,
    max_number_up_on_sheet: int = 0,
    override_number_up_on_sheet: int = 0,
    join_type: str | None = None,
    side2_ink: str | None = None
) -> tuple[int, int]:
    """Calculates the number of press sheets required and total pieces produced.

    This function determines the total number of press sheets needed for a print
    job based on the given quantity, page count, and press sheet constraints. It
    also calculates the total number of pieces that will be produced.
    Considerations include margins, bleed, maximum pieces per sheet, and
    whether pages are joined.

    :param quantity: The total quantity of the print job.
    :type quantity: int
    :param pages: The number of pages per unit.
    :type pages: int
    :param finished_width: The final width of a single piece.
    :type finished_width: float
    :param finished_height: The final height of a single piece.
    :type finished_height: float
    :param press_sheet_data: DataFrame containing press sheet dimensions.
    :type press_sheet_data: pd.DataFrame
    :param device_data: A DataFrame containing device margin information.
    :type device_data: pd.DataFrame
    :param default_piece_margin: Margin applied to each piece. Defaults to
        0.125.
    :type default_piece_margin: float, optional
    :param default_piece_bleed: Bleed applied to each piece. Defaults to 0.0.
    :type default_piece_bleed: float, optional
    :param max_number_up_on_sheet: Maximum number of pieces per press sheet.
        Defaults to 0.
    :type max_number_up_on_sheet: int, optional
    :param override_number_up_on_sheet: If set, overrides the number of pieces
        per sheet. Defaults to 0.
    :type override_number_up_on_sheet: int, optional
    :param join_type: Specifies if pages should be joined. Valid values are
        "Left" and "Top". Defaults to None.
    :type join_type: str | None, optional
    :param side2_ink: Specifies ink for the second side, if applicable.
        Defaults to None.
    :type side2_ink: str | None, optional
    :returns: A tuple containing the total number of press sheets required and
        the total number of pieces produced.
    :rtype: tuple[int, int]
    """
    press_runs = 1
    press_sheets_on_last_run = 0

    unfinished_width, unfinished_height = press_sheet_data[
        ['unfinished_width', 'unfinished_height']
    ].drop_duplicates().squeeze()

    left_margin, right_margin, top_margin, bottom_margin = device_data[
        ['press_sheet_left_margin', 'press_sheet_right_margin',
         'press_sheet_top_margin', 'press_sheet_bottom_margin']
    ].drop_duplicates().squeeze()

    pieces_per_sheet = _calculate_pieces_per_sheet(
        piece_width=finished_width,
        piece_height=finished_height,
        press_sheet_width=unfinished_width,
        press_sheet_height=unfinished_height,
        left_margin=left_margin,
        right_margin=right_margin,
        top_margin=top_margin,
        bottom_margin=bottom_margin,
        default_piece_margin=default_piece_margin,
        default_piece_bleed=default_piece_bleed,
        max_number_up_on_sheet=max_number_up_on_sheet,
        override_number_up_on_sheet=override_number_up_on_sheet,
        join_type=join_type
    )

    if pages == 1:
        pieces_per_quantity = 1
        num_pieces = quantity
        press_sheets_per_run = quantity // pieces_per_sheet
        if quantity % pieces_per_sheet > 0:
            press_sheets_per_run += 1
        press_sheets_per_quantity = press_sheets_per_run
        press_runs_per_ink = 1
        press_runs_requiring_make_ready = press_runs
        if side2_ink:
            press_runs += 1
        # Total press sheets is independent of press runs.
        num_press_sheets = press_sheets_per_run
    else:
        # If no fold is performed, a page is one piece.
        pieces_per_quantity = pages
        # If a fold is performed, a page is a quarter of a piece.
        if join_type:
            pieces_per_quantity = pages // 4
        num_pieces = quantity * pieces_per_quantity
        press_sheets_per_quantity = 1

        if pieces_per_quantity == 1:
            press_sheets_per_run = quantity // pieces_per_sheet
            if quantity % pieces_per_sheet != 0:
                press_sheets_per_run += 1
        elif 1 < pieces_per_quantity < pieces_per_sheet:
            piece_sets_per_sheet = pieces_per_sheet // pieces_per_quantity
            if quantity < piece_sets_per_sheet:
                press_sheets_per_run = 1
            else:
                press_sheets_per_run = quantity // piece_sets_per_sheet
        else:
            press_sheets_per_run = quantity
            press_sheets_per_quantity = pieces_per_quantity // pieces_per_sheet
            if pieces_per_quantity % pieces_per_sheet != 0:
                press_sheets_per_quantity += 1
                remaining_pieces = pieces_per_quantity % pieces_per_sheet
                places_available = pieces_per_sheet - remaining_pieces
                if (places_available >= remaining_pieces
                        and places_available // remaining_pieces >= 1):
                    press_sheets_on_last_run = (press_sheets_per_run
                                                // (pieces_per_sheet
                                                    // remaining_pieces))
                else:
                    press_sheets_on_last_run = press_sheets_per_run

            press_runs = press_sheets_per_quantity

        press_runs_per_ink = press_runs_requiring_make_ready = press_runs
        if side2_ink:
            press_runs *= 2

        if press_sheets_on_last_run != 0:
            press_sheets_per_main_quantity = max(
                press_sheets_per_quantity - 1, 1
            )
            press_sheets_per_main_runs = (press_sheets_per_run
                                          * press_sheets_per_main_quantity)
            num_press_sheets = (press_sheets_per_main_runs
                                + press_sheets_on_last_run)
        else:
            num_press_sheets = press_sheets_per_run * press_sheets_per_quantity

    return (num_press_sheets
            + device_data['make_ready_count'].drop_duplicates().item(),
            num_pieces)


def _calculate_square_feet(
    final_width: float,
    final_height: float,
    job_print_substrate_data: dict[str, Any]
) -> tuple[float, float, float]:
    """Calculates square footage, total square footage, and linear footage.

    This function computes the square footage of a printed piece based on its
    final dimensions and substrate specifications. It accounts for bleeds,
    margins, and wastage percentage to determine the total square footage. The
    linear footage is also calculated for pricing or material estimation.

    :param final_width: The final width of the printed piece.
    :type final_width: float
    :param final_height: The final height of the printed piece.
    :type final_height: float
    :param job_print_substrate_data: A dictionary containing substrate data,
        including default bleeds, margins, and wastage percentage.
    :type job_print_substrate_data: dict[str, Any]
    :returns: A tuple containing the square footage, total square footage with
        wastage, and linear footage.
    :rtype: tuple[float, float, float]
    """
    flat_width = (final_width
                  + job_print_substrate_data['default_installer_bleed_left']
                  + job_print_substrate_data['default_installer_bleed_right'])
    flat_height = (final_height
                   + job_print_substrate_data['default_installer_bleed_top']
                   + job_print_substrate_data['default_installer_bleed_bottom'])

    printed_width = (flat_width
                     + job_print_substrate_data['default_piece_bleed_left']
                     + job_print_substrate_data['default_piece_bleed_right'])
    printed_height = (flat_height
                      + job_print_substrate_data['default_piece_bleed_top']
                      + job_print_substrate_data['default_piece_bleed_bottom'])

    outer_width = (printed_width
                   + job_print_substrate_data['default_piece_margin_left']
                   + job_print_substrate_data['default_piece_margin_right'])
    outer_height = (printed_height
                    + job_print_substrate_data['default_piece_margin_top']
                    + job_print_substrate_data['default_piece_margin_bottom'])
    square_feet = outer_width * outer_height / 144.0
    total_square_feet = ((1.0
                          + job_print_substrate_data['area_wastage_percent'])
                         * square_feet)
    linear_feet = 2 * (final_width + final_height)

    return square_feet, total_square_feet, linear_feet


def _interpolate_cost_table(cost_table: pd.DataFrame, cost_basis) -> float:
    """Interpolates cost from a pricing table based on a given cost basis.

    This function calculates the cost based on a provided cost basis using a
    stepwise interpolation approach. If the cost basis falls within an endpoint
    range, the function determines whether to apply a flat price or a per-unit
    rate.

    :param cost_table: DataFrame containing pricing breakpoints, with columns
        'endpoint', 'point_price', 'point_flatprice', and 'price'.
    :type cost_table: pd.DataFrame
    :param cost_basis: The basis for cost calculation (e.g., quantity, area).
    :type cost_basis: float
    :returns: The interpolated cost based on the provided cost basis.
    :rtype: float
    """
    price = 0.0
    start_point = 0

    for _, row in cost_table.iterrows():
        if pd.notna(row['endpoint']):
            point_price = (0.0
                           if pd.isna(row['point_price'])
                           else row['point_price'])

            if cost_basis <= row['endpoint']:
                if row['point_flatprice']:
                    price += point_price
                else:
                    price += point_price * (cost_basis - start_point)
                return price
            else:
                if row['point_flatprice']:
                    price += point_price
                else:
                    price += point_price * (row['endpoint'] - start_point)

                start_point = row['endpoint']

    cost_table_point_price = (0.0
                              if cost_table['price'].isna().all()
                              else cost_table['price'].drop_duplicates().item())
    if cost_table['flatprice'].drop_duplicates().item():
        price += cost_table_point_price
    else:
        price += cost_table_point_price * (cost_basis - start_point)

    return price


def _get_markup_percent(
    markups: pd.Series,
    quantity_or_area: float
) -> float:
    """Determines the markup percentage based on quantity or area.

    This function interpolates a markup percentage from a series of markup
    breakpoints. If the given quantity or area falls between two points, a
    linear interpolation is performed. If it falls outside the range, the
    closest applicable markup is used.

    :param markups: A pandas Series where index values represent breakpoints
        and corresponding values represent markup percentages.
    :type markups: pd.Series
    :param quantity_or_area: The quantity or area for which markup needs to be
        determined.
    :type quantity_or_area: float
    :returns: The interpolated markup percentage.
    :rtype: float
    """
    start_point = 0.0
    start_markup_percent = markups.iloc[0].item()

    for end_point, end_markup_percent in markups.iloc[1:].items():
        if start_point <= quantity_or_area < end_point:
            if start_point == end_point:
                return end_markup_percent
            else:
                line_slope = ((end_markup_percent - start_markup_percent)
                              / (end_point - start_point))
                return (line_slope * (quantity_or_area - start_point)
                        + start_markup_percent)

        start_point = end_point
        start_markup_percent = end_markup_percent

    return start_markup_percent


class InvalidProductFormatException(ValueError):
    """Custom exception for handling invalid product format errors.

    This exception is raised when an invalid product format is provided
    to a function or class that expects a specific format.

    :param message: Optional custom error message to display when the
        exception is raised. Defaults to 'Invalid product format provided.'
    :type message: str

    :example:

    >>> raise InvalidProductFormatException('Unsupported product format.')
    Traceback (most recent call last):
        ...
    InvalidProductFormatException: Unsupported product format.
    """
    def __init__(self, message='Unsupported product format.'):
        super().__init__(message)


class PJCNotFoundException(ValueError):
    """Custom exception for handling invalid product format errors.

    This exception is raised when a system offering does not have an existing
    PJC or the PJC is unable to be found.

    :param message: Optional custom error message to display when the
        exception is raised. Defaults to 'PJC not found.'
    :type message: str

    :example:

    >>> raise PJCNotFoundException('PJC not found.')
    Traceback (most recent call last):
        ...
    PJCNotFoundException: PJC not found.
    """
    def __init__(self, message='PJC not found.'):
        super().__init__(message)


class CostCalculator:
    """A calculator for determining the total cost of a print job.

    This class provides a Python implementation of Collaterate's cost calculator
    logic. Costs are based on specifications like device, press sheets, ink,
    operations, etc., and the associated Collaterate production costs.

    Public Methods:
        - load_calculator: Loads production cost data and configures calculator
             settings.
        - calculate_operation_item_cost: Calculates the cost for a particular
              operation for a particular printing configuration.
        - calculate_cost: Calculates the cost for a print job.

    Instance Variables:
        - system_offering_site_share: The product (share) name.
        - system_offering_site_share_id: The product ID number.
        - system_offering: The parent product (system offering) name.
        - system_offering_id: The parent product ID number.
        - product_format: The format of the product (e.g., 'SMALL_FORMAT').
        - markup_type: The variable used to determine the markup value for a
            job; one of 'QUANTITY', 'SQUARE_FEET', 'UNIT_PRICE', or
            'THROUGHPUT'.
        - multi_pages: Indicates if the PJC allows for multiple pages. (This
            only pertains for small format products.)
        - default_config: The default configuration for the system offering.
        - load_status: The current state of the calculator instance. If PJC data
              has been retrieved and production cost data has been successfully
              loaded, then 'LOADED'; if PJC data has been successfully retrieved
              and production cost data has not been loaded, then 'READY TO
              LOAD'; or if PJC data has not been retrieved then 'NOT READY TO
              LOAD'.
    """
    def __init__(
        self,
        product: str | int,
        engine: Engine | None = None,
        verbose: bool = False
    ):
        """Initializes the CostCalculator with database connection and product
        information.

        Sets up the calculator by loading specified product data and preparing
        it for future cost calculations. Determines and stores relevant
        identifiers for the system offering site_share, system offering, print
        job classification (PJC), product format, markup_type, and default
        configuration.

        :param product: The name or ID of the product for which costs will be
            calculated.
        :type product: str | int
        :param engine: SQLAlchemy engine for database connectivity. If this
            argument is not provided, a new engine instance will be created.
            Defaults to None.
        :type engine: Engine | None
        :param verbose: If True, prints progress during initialization. Defaults
            to False.
        :type verbose: bool
        """
        self._verbose = verbose
        if engine is None and self._verbose:
            print('No Redshift connection provided; creating new engine.')
        self._engine = engine if engine is not None else redshift_engine()
        self._product_data = self._load_product_data(product)
        if isinstance(product, str):
            self.system_offering_site_share = product
            self.system_offering_site_share_id = self._product_data.get(
                'system_offering_site_share_id'
            )
        elif isinstance(product, int):
            self.system_offering_site_share = self._product_data.get(
                'system_offering_site_share'
            )
            self.system_offering_site_share_id = product
        self.system_offering = self._product_data.get(
            'system_offering'
        )
        self.system_offering_id = self._product_data.get(
            'system_offering_id'
        )
        self._pjc_id = self._product_data.get('pjc_id')
        self.product_format = self._product_data.get('product_format')
        self._pjc_format = self._product_data.get('pjc_format')
        self._join_type = self._product_data.get('join_type')
        self._default_piece_margin = self._product_data.get(
            'default_piece_margin'
        )
        self._default_piece_bleed = self._product_data.get(
            'default_piece_bleed'
        )
        self._max_number_up_on_sheet = self._product_data.get(
            'max_number_up_on_sheet'
        )
        self._qty_attrition_percent = self._product_data.get(
            'qty_attrition_percent'
        )
        self._markup_type = self._product_data.get('markup_type')
        self._site_markup_percent = self._product_data.get(
            'site_markup_percent'
        )
        self.default_config = get_default_configs(
            self._engine,
            self.system_offering_site_share_id
        )[self.system_offering_site_share_id]
        self.load_status = ('READY TO LOAD'
                            if self._pjc_id
                            else 'NOT READY TO LOAD')
        # Initialize calculator inputs as None. These instance attributes are
        # defined by calling the `load_calculator` method.
        self._production_cost_data = None
        self.preset_dimensions = None

    def __str__(self):
        """Returns a user-friendly string representation of the CostCalculator
        instance.

        :returns: A string indicating the system offering name and product
            format.
        :rtype: str
        """
        return (f'{self.__class__.__name__} for {self.system_offering} '
                f'({self.product_format})')

    def __repr__(self):
        """Returns a more technical string representation of the CostCalculator
        instance.

        Provides an at-a-glance view of the calculator's loading state:
        - 'LOADED' if production cost data is available.
        - 'READY TO LOAD' if the PJC ID is set but cost data is not loaded.
              (Call the `load_calculator` instance method to load the
              calculator.)
        - 'NOT READY TO LOAD' if initialization failed to identify the product's
              PJC. Test database connectivity and verify the provided system
              offering (id) is correct.


        :returns: A string representation of the calculator's state.
        :rtype: str
        """
        return (f'{self.__class__.__name__}({self.system_offering_id}, '
                f'{self.load_status})')

    def __getstate__(self) -> dict[str, Any]:
        """Prepare instance state for pickling.

        This method removes any non-serializable attributes from the instance
        dictionary to make the object compatible with pickling.

        :returns: A dictionary representing the instance state, excluding the
            `_engine` attribute.
        :rtype: dict[str, Any]
        """
        state = self.__dict__.copy()
        state.pop('_engine', None)

        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore instance state during unpickling.

        This method updates the instance dictionary with the state data loaded
        from a pickle file and reinitializes non-serializable attributes as
        needed.

        :param state: A dictionary containing the pickled instance state.
        :type state: dict[str, Any]
        """
        self.__dict__.update(state)
        self._engine = None

    def load_calculator(
        self,
        as_admin: bool = False
    ) -> Self:
        """Retrieves PJC specifications and the associated costs, making the
        class instance ready to calculate costs.

        IDs for allowed devices, operation items, press sheets, inks, print
        substrates, mount substrates, and laminates are loaded as an
        intermediate variable `pjc_component_ids`, along with a Boolean
        `multi_pages` instance attribute that denotes whether the product has
         more than one page. (Examples of single-page products include business
        cards and brochures, while saddle-stitch booklets and calendars have
        more than one page.)

        These IDs are used by the `_load_production_cost_data` method to read in
        the corresponding cost table data, which are stored in the dict instance
        variable `_production_cost_data`.

        :param as_admin: If true, include administrator materials/operations
            options that may be hidden to some users. Defaults to `False`.
        :type as_admin: bool, optional
        :returns: The initialized calculator instance with loaded cost data.
        :rtype: CostCalculator

        :raises PJCNotFoundException: If a PJC is unable to be associated to the
             system offering when the CostCalculator instance is instantiated.
        """
        if self.load_status == 'NOT READY TO LOAD':
            message = ('Unable to retrieve PJC information. Check database '
                       'connectivity and verify {so} is the correct system '
                       'offering {id}.')
            message = (message.format(
                so=self.system_offering_site_share,
                id='\b'
            )
                       if self.system_offering
                       else message.format(
                so=self.system_offering_site_share_id,
                id='id'
            ))
            raise PJCNotFoundException(message)
        try:
            self._production_cost_data = self._load_production_cost_data(
                as_admin=as_admin
            )
            self.preset_dimensions = self._load_preset_dimensions()
            self.load_status = 'LOADED'
        except (KeyError, ValueError) as e:
            if self._verbose:
                print('Error loading calculator:')
            print(e)

        return self

    def _load_product_data(
        self,
        system_offering_site_share: int | str
    ) -> dict[str, int | str | None]:
        """Retrieves the print job classification ID and product format.

        This function queries a database to retrieve the print job
        classification ID (`pjc_id`) and product format for a given system
        offering. It checks the `system_offerings` table first. If the `pjc_id`
        is -1, it performs a secondary query to obtain `pjc_id`, format, and
        markup type based on `system_offering` or `system_offering_plural` names
        in the print job classification tables.

        :param system_offering_site_share: The product identifier, provided as
            an integer or string, representing `id`, `name`, or `name_plural`.
        :type system_offering_site_share: int | str
        :returns: A dict containing the system offering name, its ID, its PJC
            ID, its product format, and its markup type. If the system offering
            is not located, a dict where all values are None is returned.
        :rtype: dict[str, int| str | None]

        .. note::
            The initial query will return -1 if the
            `system_offering_site_share` cannot be matched. If the fallback
            query is unable to find a match, a dict of all None values will be
            returned.
        """
        # First, query coll_src.system_offering_site_shares to identify the
        # share, which may or may not 1) exist or 2) have a PJC assigned.
        if self._verbose:
            print('Retrieving system offering information ...')
        query_path = QUERY_DIR / 'load_product_data.sql'

        with self._engine.begin() as connection:
            with open(query_path, 'r', encoding='utf-8') as file:
                data = pd.read_sql_query(
                    sql=text(Template(file.read()).render(
                        system_offering_site_share=system_offering_site_share
                    )),
                    con=connection
                ).squeeze()

            keys = data.index

            # If the offering isn't found, it might still be possible to get PJC
            # info from the PJC relations if the name of the offering is
            # provided.
            if data.empty:
                if isinstance(system_offering_site_share, int):
                    if self._verbose:
                        print('Unable to retrieve product or PJC information.')
                    return {key: None for key in keys}
                else:
                    if self._verbose:
                        print('Unable to retrieve product information.')
                    data = pd.Series(
                        data=[-1, system_offering_site_share,
                              system_offering_site_share],
                        index=['pjc_id', 'system_offering_site_share',
                               'system_offering_site_share_plural']
                    )

            elif data['pjc_id'] == -1:
                if self._verbose:
                    print('Attempting direct retrieval of PJC information ...')

                query_path = QUERY_DIR / 'load_product_data_alt.sql'

                with open(query_path, 'r', encoding='utf-8') as file:
                    data = pd.read_sql_query(
                        sql=text(Template(file.read()).render(
                            system_offering_site_share=data[
                                'system_offering_site_share'
                            ],
                            system_offering_site_share_plural=data[
                                'system_offering_site_share_plural'
                            ]
                        )),
                        con=connection
                    ).squeeze()

                # If the PJC still isn't found, give up.
                if data.empty:
                    if self._verbose:
                        print('Unable to retrieve PJC information.')
                    return {key: None for key in keys}

                else:
                    if self._verbose:
                        print('PJC information retrieved; product information '
                              'may be missing.')

            else:
                if self._verbose:
                    print('Product and PJC information retrieved.')

        # Return a dict specifying the system offering, its ID, its PJC ID,
        # and its format.
        return data.drop(
            ['system_offering_site_share_plural', 'system_offering_plural'],
            errors='ignore'
        ).to_dict()

    def _load_pjc_component_ids(self) -> dict[str, list[int]]:
        """Retrieves component IDs for a print job classification.

        This function queries the database to retrieve component IDs for a
        specified print job classification (`pjc_id`) based on the product
        format. For `SMALL_FORMAT`, it retrieves IDs related to devices, press
        sheets, inks, and operation items. For `LARGE_FORMAT`, it retrieves IDs
        for devices, print substrates, mount substrates, laminates, and
        operation items.

        :returns: A dictionary mapping component categories (e.g., `device_ids`,
            `press_sheet_ids`) to lists of component IDs.
        :rtype: dict[str, list[int]]
        """
        query_path = QUERY_DIR / (
            'load_' + ('sf'
                       if self._pjc_format == 'SMALL_FORMAT'
                       else 'lf') + '_pjc_component_data.sql'
        )

        with (open(query_path, 'r', encoding='utf-8') as file,
              self._engine.begin() as connection):
            data = pd.read_sql_query(
                sql=text(Template(file.read()).render(pjc_id=self._pjc_id)),
                con=connection
            ).T.squeeze().to_dict()

        data = {key: sorted([int(x) for x in value.split(',')])
                if value else []
                for key, value in data.items()}

        return data

    def _load_production_cost_data(
        self,
        as_admin: bool = False
    ) -> dict[str, pd.DataFrame]:
        """Fetch all required cost data from the database and store in a
        dictionary.

        :param as_admin: If true, include administrator materials/operations
            options that may be hidden to some users. Defaults to `False`.
        :type as_admin: bool, optional
        :returns: A dictionary with cost data for devices, press sheets, inks,
            and operations.
        :rtype: dict
        """
        if self._verbose:
            print('Fetching production data ...')
        if self._pjc_format == 'SMALL_FORMAT':
            production_cost_data = {
                'devices': self._load_device_cost_data(),
                'press_sheets': self._load_press_sheet_cost_data(),
                'cover_press_sheets': self._load_cover_press_sheet_cost_data(),
                'side1_inks': self._load_side1_ink_cost_data(),
                'side2_inks': self._load_side2_ink_cost_data(),
                'cover_side1_inks': self._load_cover_side1_ink_cost_data(),
                'cover_side2_inks': self._load_cover_side2_ink_cost_data(),
                'operations': self._load_operation_item_cost_data()
            }
        elif self._pjc_format == 'LARGE_FORMAT':
            production_cost_data = {
                'devices': self._load_device_cost_data(),
                'cutting': self._load_cutting_cost_data(),
                'print_substrates': self._load_print_substrate_cost_data(),
                'mount_substrates': self._load_mount_substrate_cost_data(),
                'front_laminates': self._load_front_laminate_cost_data(),
                'back_laminates': self._load_back_laminate_cost_data(),
                'operations': self._load_operation_item_cost_data()
            }
        else:
            production_cost_data = {}

        if not as_admin:
            production_cost_data = {
                key: (value[~value['admin_only']]
                      if 'admin_only' in value.columns and not value.empty
                      else value)
                for key, value in production_cost_data.items()
            }

        if self._verbose:
            print('Production cost data loaded.')

        return production_cost_data

    def _load_preset_dimensions(self) -> pd.DataFrame:
        """Retrieves preset dimensions options for the offering PJC.

        This function queries the database to retrieve preset dimension options,
        which determine whether the customer will incur an additional "Custom
        Dimensions" operation cost.

        :returns: A list of `(width, height)` tuples or a list of `width` preset
            dimensions.
        :rtype: list[tuple[float, float]] or list[float]
        """
        query_path = QUERY_DIR / (
            'load_' + ('sf'
                       if self._pjc_format == 'SMALL_FORMAT'
                       else 'lf') + '_preset_dimensions_data.sql'
        )

        with (open(query_path, 'r', encoding='utf-8') as file,
              self._engine.begin() as connection):
            data = pd.read_sql_query(
                sql=text(Template(file.read()).render(pjc_id=self._pjc_id)),
                con=connection,
                index_col='preset_dimensions_id'
            )

        return data

    def _load_device_cost_data(self) -> pd.DataFrame:
        """Retrieves device cost data for devices associated with the system
        offering site share.

        This function queries the database to retrieve cost data for a list of
        devices. For `SMALL_FORMAT`, it gathers costs related to setup and run
        pricing. For `LARGE_FORMAT`, it additionally includes ink-related costs.

        :returns: A DataFrame indexed by device ID, containing cost information
            including version price, run price, and setup price. For large
            format, additional ink cost columns are provided.
        :rtype: pd.DataFrame
        """
        query_path = QUERY_DIR / (
            'load_' + ('sf'
                       if self._pjc_format == 'SMALL_FORMAT'
                       else 'lf') + '_device_cost_data.sql'
        )

        with (open(query_path, 'r', encoding='utf-8') as file,
              self._engine.begin() as connection):
            data = pd.read_sql_query(
                sql=text(Template(file.read()).render(pjc_id=self._pjc_id)),
                con=connection,
                index_col='device_id'
            ).replace([None], np.nan)

        return data

    def _load_press_sheet_cost_data(self) -> pd.DataFrame:
        """Retrieves cost data for press sheets.

        This function queries the database to fetch cost information for a list
        of press sheets, including attributes such as press sheet type, weight,
        dimensions, and unit pricing.

        :returns: A DataFrame indexed by press sheet ID, containing details such
            as press sheet type, weight, dimensions, and pricing information.
        :rtype: pd.DataFrame
        """
        query_path = QUERY_DIR / 'load_press_sheet_cost_data.sql'

        with (open(query_path, 'r', encoding='utf-8') as file,
              self._engine.begin() as connection):
            data = pd.read_sql_query(
                sql=text(Template(file.read()).render(pjc_id=self._pjc_id)),
                con=connection,
                index_col='press_sheet_id'
            ).replace([None], np.nan)

        return data

    def _load_cover_press_sheet_cost_data(self) -> pd.DataFrame:
        """Retrieves cost data for cover press sheets.

        This function queries the database to fetch cost information for a list
        of cover press sheets, including attributes such as press sheet type,
        weight, dimensions, and pricing.

        :returns: A DataFrame indexed by press sheet ID, containing details such
            as press sheet type, weight, dimensions, and pricing information.
        :rtype: pd.DataFrame
        """
        query_path = QUERY_DIR / 'load_cover_press_sheet_cost_data.sql'

        with (open(query_path, 'r', encoding='utf-8') as file,
              self._engine.begin() as connection):
            data = pd.read_sql_query(
                sql=text(Template(file.read()).render(pjc_id=self._pjc_id)),
                con=connection,
                index_col='press_sheet_id'
            ).replace([None], np.nan)

        return data

    def _load_side1_ink_cost_data(self) -> pd.DataFrame:
        """Retrieves cost data for side 1 inks.

        :returns: A DataFrame indexed by ink ID, containing details such as ink
            type, unit price, and flat pricing status.
        :rtype: pd.DataFrame
        """
        query_path = QUERY_DIR / 'load_side1_ink_cost_data.sql'

        with (open(query_path, 'r', encoding='utf-8') as file,
              self._engine.begin() as connection):
            data = pd.read_sql_query(
                sql=text(Template(file.read()).render(pjc_id=self._pjc_id)),
                con=connection,
                index_col='ink_id'
            ).replace([None], np.nan)

        return data

    def _load_side2_ink_cost_data(self) -> pd.DataFrame:
        """Retrieves cost data for side 2 inks.

        :returns: A DataFrame indexed by ink ID, containing details such as ink
            type, unit price, and flat pricing status.
        :rtype: pd.DataFrame
        """
        query_path = QUERY_DIR / 'load_side2_ink_cost_data.sql'

        with (open(query_path, 'r', encoding='utf-8') as file,
              self._engine.begin() as connection):
            data = pd.read_sql_query(
                sql=text(Template(file.read()).render(pjc_id=self._pjc_id)),
                con=connection,
                index_col='ink_id'
            ).replace([None], np.nan)

        return data

    def _load_cover_side1_ink_cost_data(self) -> pd.DataFrame:
        """Retrieves cost data for cover side 1 inks.

        :returns: A DataFrame indexed by ink ID, containing details such as ink
            type, unit price, and flat pricing status.
        :rtype: pd.DataFrame
        """
        query_path = QUERY_DIR / 'load_cover_side1_ink_cost_data.sql'

        with (open(query_path, 'r', encoding='utf-8') as file,
              self._engine.begin() as connection):
            data = pd.read_sql_query(
                sql=text(Template(file.read()).render(pjc_id=self._pjc_id)),
                con=connection,
                index_col='ink_id'
            ).replace([None], np.nan)

        return data

    def _load_cover_side2_ink_cost_data(self) -> pd.DataFrame:
        """Retrieves cost data for cover side 2 inks.

        :returns: A DataFrame indexed by ink ID, containing details such as ink
            type, unit price, and flat pricing status.
        :rtype: pd.DataFrame
        """
        query_path = QUERY_DIR / 'load_cover_side2_ink_cost_data.sql'

        with (open(query_path, 'r', encoding='utf-8') as file,
              self._engine.begin() as connection):
            data = pd.read_sql_query(
                sql=text(Template(file.read()).render(pjc_id=self._pjc_id)),
                con=connection,
                index_col='ink_id'
            ).replace([None], np.nan)

        return data

    def _load_cutting_cost_data(self):
        """Retrieves cutting cost data for large format PJCs.

        :returns: A DataFrame containing cutting cost data.
        :rtype: pd.DataFrame
        """
        query_path = QUERY_DIR / 'load_cutting_cost_data.sql'

        with (open(query_path, 'r', encoding='utf-8') as file,
              self._engine.begin() as connection):
            data = pd.read_sql_query(
                sql=text(Template(file.read()).render(pjc_id=self._pjc_id)),
                con=connection
            ).replace([None], np.nan)

        return data

    def _load_print_substrate_cost_data(self) -> pd.DataFrame:
        """Retrieves cost data for specified print substrates.

        This function queries the database to fetch cost information for a list
        of print substrates, including attributes such as name, dimensions, and
        square foot cost.

        :returns: A DataFrame indexed by print substrate ID, containing details
            such as name, width, height, and square foot cost.
        :rtype: pd.DataFrame
        """
        query_path = QUERY_DIR / 'load_print_substrate_cost_data.sql'

        with (open(query_path, 'r', encoding='utf-8') as file,
              self._engine.begin() as connection):
            data = pd.read_sql_query(
                sql=text(Template(file.read()).render(pjc_id=self._pjc_id)),
                con=connection,
                index_col='print_substrate_id'
            ).replace([None], np.nan)

        return data

    def _load_mount_substrate_cost_data(self) -> pd.DataFrame:
        """Retrieves cost data for specified mount substrates.

        This function queries the database to fetch cost information for a list
        of mount substrates, including attributes such as name, dimensions, and
        square foot cost.

        :returns: A DataFrame indexed by mount substrate ID, containing details
            such as name, width, height, and square foot cost.
        :rtype: pd.DataFrame

        .. note::
            This function expects mount substrate data to be stored in the
            `mount_substrates` table with columns for dimensions and square foot
            cost.
        """
        query_path = QUERY_DIR / 'load_mount_substrate_cost_data.sql'

        with (open(query_path, 'r', encoding='utf-8') as file,
              self._engine.begin() as connection):
            data = pd.read_sql_query(
                sql=text(Template(file.read()).render(pjc_id=self._pjc_id)),
                con=connection,
                index_col='mount_substrate_id'
            ).replace([None], np.nan)

        return data

    def _load_front_laminate_cost_data(self) -> pd.DataFrame:
        """Retrieves cost data for front laminates.

        This function queries the database to fetch cost information for a list
        of laminates, including attributes such as name and square foot cost.

        :returns: A DataFrame indexed by laminate ID, containing details such as
            name and square foot cost.
        :rtype: pd.DataFrame
        """
        query_path = QUERY_DIR / 'load_front_laminate_cost_data.sql'

        with (open(query_path, 'r', encoding='utf-8') as file,
              self._engine.begin() as connection):
            data = pd.read_sql_query(
                sql=text(Template(file.read()).render(pjc_id=self._pjc_id)),
                con=connection,
                index_col='laminate_id'
            ).replace([None], np.nan)

        return data

    def _load_back_laminate_cost_data(self) -> pd.DataFrame:
        """Retrieves cost data for back laminates.

        This function queries the database to fetch cost information for a list
        of laminates, including attributes such as name and square foot cost.

        :returns: A DataFrame indexed by laminate ID, containing details such as
            name and square foot cost.
        :rtype: pd.DataFrame
        """
        query_path = QUERY_DIR / 'load_back_laminate_cost_data.sql'

        with (open(query_path, 'r', encoding='utf-8') as file,
              self._engine.begin() as connection):
            data = pd.read_sql_query(
                sql=text(Template(file.read()).render(pjc_id=self._pjc_id)),
                con=connection,
                index_col='laminate_id'
            ).replace([None], np.nan)

        return data

    def _load_operation_item_cost_data(self) -> pd.DataFrame:
        """Retrieves cost data for specified operation items.

        This function queries the database to fetch cost information for a list
        of operation items, including details such as setup and run costs, cost
        basis, and endpoint values. The function handles both small-format and
        large-format operations based on the `product_format` parameter.

        :returns: A DataFrame containing cost details for each operation item
            ID, with calculated accumulated costs.
        :rtype: pd.DataFrame
        """
        query_path = QUERY_DIR / (
            'load_' + ('sf'
                       if self._pjc_format == 'SMALL_FORMAT'
                       else 'lf') + '_operation_item_cost_data.sql'
        )

        with (open(query_path, 'r', encoding='utf-8') as file,
              self._engine.begin() as connection):
            data = pd.read_sql_query(
                sql=text(Template(file.read()).render(pjc_id=self._pjc_id)),
                con=connection,
                index_col='operation_item_id'
            ).replace([None], np.nan)

        return data

    def calculate_operation_item_cost(
        self,
        operation_data: pd.DataFrame,
        quantity: int,
        **kwargs: dict[str, float]
    ) -> float:
        """Calculate the total cost for an operation item.

        This function computes the cost for a given operation item based on the
        operation data and the specified number of press sheets, cover sheets,
        and pieces. The cost calculation varies based on `cost_basis` in
        `operation_data`, with conditions for flat pricing, per-piece costs,
        per-sheet costs, and other cost-basis types.

        :param operation_data: A DataFrame containing operation item data with
        cost basis, run price, setup price, flat pricing, and endpoints.
        :type operation_data: pd.DataFrame
        :param quantity: The quantity ordered.
        :type quantity: int
        :param kwargs: A dict of relevant configuration parameters for the job.
            For small format jobs, this dict should include the number of press
            sheets (`num_press_sheets`), and the number of cover press sheets
            (`num_cover_press_sheets`). For large format jobs, it should include
            square footage (`square_feet`), and the length in inches of each
            side that should be considered in calculating linear feet
            (`left_side`, `right_side`, `top_side`, and `bottom_side`).
        :type kwargs: dict[str, float]
        :returns: The calculated cost for the operation item.
        :rtype: float
        """
        if operation_data.empty:
            return 0.0

        cost_basis = operation_data['cost_basis'].drop_duplicates().item()
        if self._pjc_format == 'SMALL_FORMAT':
            num_press_sheets = kwargs['num_press_sheets']
            num_cover_press_sheets = kwargs['num_cover_press_sheets']
            num_pieces = kwargs['num_pieces']
            num_cover_pieces = kwargs['num_cover_pieces']
            answer = kwargs['answer']

            if operation_data['cover_only'].all():
                num_press_sheets = num_cover_press_sheets
                num_pieces = num_cover_pieces
            else:
                num_press_sheets += num_cover_press_sheets
                num_pieces += num_cover_pieces

            if cost_basis == 'Cost Number of Sheets':
                cost_basis_num = num_press_sheets
            elif cost_basis == 'Cost Number of Pieces':
                cost_basis_num = num_pieces
            elif (cost_basis
                  == 'Cost Number of Pieces after Setting Finished Quantity'):
                cost_basis_num = quantity
            elif (cost_basis
                  == 'Cost Number of Pieces after Dividing by Answer'):
                cost_basis_num = quantity // answer
            else:
                # TODO: Implement logic for other cost basis types.
                pass
        else:
            if cost_basis == 'Quantity':
                cost_basis_num = quantity
            elif cost_basis == 'Square Feet':
                cost_basis_num = kwargs['square_feet']
            elif cost_basis == 'Linear Feet':
                cost_basis_num = ((kwargs['left_side'] + kwargs['right_side']
                                   + kwargs['top_side'] + kwargs['bottom_side'])
                                  / 12.0 * quantity)
            elif cost_basis == 'Direct Add-On':
                return (operation_data['default_add_on_value']
                        .drop_duplicates()
                        .item())
            elif cost_basis == 'Costed Add-On':
                cost_basis_num = (operation_data['default_add_on_value']
                                  .drop_duplicates()
                                  .item())
            elif cost_basis == 'Costed Add-On Per Piece':
                cost_basis_num = (operation_data['default_add_on_value']
                                  .drop_duplicates()
                                  .item()) * quantity
            elif cost_basis == 'Quantity Divisor':
                cost_basis_num = np.divide(
                    quantity,
                    (operation_data['default_add_on_value']
                     .drop_duplicates()
                     .item()),
                    out=np.zeros_like(quantity, dtype=float),
                    where=(operation_data['default_add_on_value']
                           .drop_duplicates()
                           .item()) != 0
                )

        return _interpolate_cost_table(operation_data, cost_basis_num)

    def _calculate_sf_component_costs(
        self,
        **config: dict[str, Y]
    ) -> dict[str, float]:
        """Calculate component costs for a small format (SF) print job.

        This method computes costs for various components of a small format
        print job, including press sheets, inks, devices, and operations, based
        on the provided configuration parameters.

        :param config: Configuration dictionary specifying job parameters:
            - quantity (int): The total number of units.
            - versions (int): The total number of versions.
            - quantity_per_version (int): The total number of units per version.
            - device: The device to use in production. If not specified, the
                  system offering's default device is used.
            - pages (int): The number of pages per unit. Defaults to 1.
            - finished_width (float): The finished width of each piece.
            - finished_height (float): The finished height of each piece.
            - press_sheet_type (str): The press sheet (paper stock) type.
            - press_sheet_weight (str): The press sheet (paper stock) weight.
            - press_sheet_color (str): The press sheet (paper stock) color.
            - side1_ink_type (str): The ink type for side 1 of the press sheet.
            - side2_ink_type (str): The ink type for side 2 of the press sheet,
                  if any.
            - cover_press_sheet_type (str): The cover press sheet (paper stock)
                  type, if any.
            - cover_press_sheet_weight (str): The cover press sheet (paper
                  stock) weight, if any.
            - cover_press_sheet_color (str): The cover press sheet (paper stock)
                color, if any.
            - cover_side1_ink_type: The ink type for side 1 of the cover press,
                  sheet if any.
            - cover_side2_ink_type: The ink type for side 2 of the cover press
                  sheet, if any.
            - operations: Dictionary of operations and corresponding operation
                  items, if any.
            - turnaround_time (str): The number of production days requested by
                the customer.
            - proof (str): The proof type requested by the customer.
        :type config: dict[str, Y]
        :returns: A dictionary with calculated costs for each component.
        :rtype: dict[str, float]

        .. note::
            This method relies on production cost data loaded through
            `load_calculator`. Ensure that the cost data is loaded prior to
            calling this method to avoid errors.
        """
        config = {key: value
                  for key, value in config.items()
                  if value is not None}

        device = config.get('device', self.default_config['device'])
        versions = config.get('versions', 1)
        quantity = config.get(
            'quantity',
            config.get('quantity_per_version')
        ) * versions
        num_pages = config.get('pages', 1)
        if config.get(
            'cover_press_sheet',
            config.get('cover_press_sheet_type')
        ):
            if self._join_type:
                num_cover_pages = 4
            else:
                num_cover_pages = 2
            num_pages -= num_cover_pages

        finished_width = config.get(
            'finished_width',
            (self.preset_dimensions.loc[config.get(
                'preset_dimensions',
                self.preset_dimensions.index[0]
            ), 'finished_width'])
        )
        finished_height = config.get(
            'finished_height',
            (self.preset_dimensions.loc[config.get(
                'preset_dimensions',
                self.preset_dimensions.index[0]
            ), 'finished_height'])
        )

        device_data = self._production_cost_data['devices']
        if isinstance(device, int):
            job_device_data = device_data.loc[[device]]
        else:
            job_device_data = device_data[device_data['device'] == device]

        if isinstance(job_device_data, pd.Series):
            job_device_data = job_device_data.to_frame().T

        press_sheet_data = self._production_cost_data['press_sheets']
        if 'press_sheet' in config:
            job_press_sheet_data = press_sheet_data.loc[[config['press_sheet']]]
        elif isinstance(config['press_sheet_type'], int):
            job_press_sheet_data = (press_sheet_data
                                    .loc[[config['press_sheet_type']]])
        else:
            job_press_sheet_data = press_sheet_data[
                (press_sheet_data['press_sheet_type']
                 == config['press_sheet_type'])
                & (press_sheet_data['press_sheet_weight']
                   == config['press_sheet_weight'])
                & (press_sheet_data['press_sheet_color']
                   == config['press_sheet_color'])
            ]

        if isinstance(job_press_sheet_data, pd.Series):
            job_press_sheet_data = job_press_sheet_data.to_frame().T

        if len(job_press_sheet_data.index.drop_duplicates()) > 1:
            # Accounts for Hardcover Books press sheet logic.
            job_press_sheet_data = job_press_sheet_data[
                job_press_sheet_data[['unfinished_width', 'unfinished_height']]
                .apply(set, axis=1) == {finished_width, finished_height}
            ]

        component_costs = defaultdict(float)

        num_press_sheets, num_pieces = _calculate_num_press_sheets_and_pieces(
            quantity=quantity,
            pages=num_pages,
            finished_width=finished_width,
            finished_height=finished_height,
            press_sheet_data=job_press_sheet_data,
            device_data=job_device_data,
            default_piece_margin=self._default_piece_margin,
            default_piece_bleed=self._default_piece_bleed,
            max_number_up_on_sheet=self._max_number_up_on_sheet,
            join_type=self._join_type,
            side2_ink=config.get('side2_ink_type')
        )

        component_costs['press_sheet_cost'] = _interpolate_cost_table(
            cost_table=job_press_sheet_data,
            cost_basis=num_press_sheets
        )

        # If the item has a cover press sheet, calculate its costs and add to
        # press sheet costs.
        if config.get(
            'cover_press_sheet',
            config.get('cover_press_sheet_type')
        ):
            cover_press_sheet_data = (
                self._production_cost_data['cover_press_sheets']
            )
            if 'cover_press_sheet' in config:
                job_cover_press_sheet_data = (
                    cover_press_sheet_data.loc[[config['cover_press_sheet']]]
                )
            elif isinstance(config['cover_press_sheet_type'], int):
                job_cover_press_sheet_data = cover_press_sheet_data.loc[
                    [config['cover_press_sheet_type']]
                ]
            else:
                job_cover_press_sheet_data = cover_press_sheet_data[
                    (cover_press_sheet_data['press_sheet_type']
                     == config['cover_press_sheet_type'])
                    & (cover_press_sheet_data['press_sheet_weight']
                       == config['cover_press_sheet_weight'])
                    & (cover_press_sheet_data['press_sheet_color']
                       == config['cover_press_sheet_color'])
                ]

            num_cover_press_sheets, num_cover_pieces = (
                _calculate_num_press_sheets_and_pieces(
                    quantity=quantity,
                    pages=num_cover_pages,
                    finished_width=finished_width,
                    finished_height=finished_height,
                    press_sheet_data=job_cover_press_sheet_data,
                    device_data=job_device_data,
                    default_piece_margin=self._default_piece_margin,
                    default_piece_bleed=self._default_piece_bleed,
                    join_type=self._join_type,
                    side2_ink=config.get('cover_side2_ink_type'),
                    max_number_up_on_sheet=self._max_number_up_on_sheet
                )
            )

            component_costs['cover_press_sheet_cost'] = _interpolate_cost_table(
                cost_table=job_cover_press_sheet_data,
                cost_basis=num_cover_press_sheets
            )
        else:
            num_cover_press_sheets = num_cover_pieces = 0

        # Calculate ink costs.
        for ink_side, press_sheet_count in zip(
            ['side1_ink', 'side2_ink', 'cover_side1_ink', 'cover_side2_ink'],
            [num_press_sheets] * 2 + [num_cover_press_sheets] * 2
        ):
            ink_type = config.get(ink_side, config.get(ink_side + '_type'))
            if ink_type is not None:
                ink_data = self._production_cost_data[f'{ink_side}s']
                if isinstance(ink_type, int):
                    job_ink_data = ink_data.loc[[ink_type]]
                else:
                    job_ink_data = ink_data[ink_data['ink_type'] == ink_type]

                component_costs[ink_side + '_cost'] = _interpolate_cost_table(
                    cost_table=job_ink_data,
                    cost_basis=press_sheet_count
                )

        # Calculate version and device costs.
        component_costs['version_cost'] = (job_device_data['version_price']
                                           .drop_duplicates()
                                           .item()) * versions

        component_costs['device_cost'] = _interpolate_cost_table(
            cost_table=job_device_data,
            cost_basis=num_press_sheets + num_cover_press_sheets
        )

        if self._verbose:
            print(f"Version cost: ${component_costs['version_cost']:,.2f}")
            print(f"Device cost: ${component_costs['device_cost']:,.2f}")
            params = list(dict.fromkeys(
                [x.replace('_type', '').replace('_weight', '')
                 for x in SF_PRINTING_OPERATIONS
                 if not any(d in x for d in ['width', 'height', 'pages'])]
            ))

            for param in params:
                desc = ' '.join(param.split('_')).capitalize()
                print(f"{desc} cost: ${component_costs[param + '_cost']:,.2f}")

        # Calculate operation costs.
        job_operations = config.get('operations')

        if job_operations:
            markup_included_operation_costs = defaultdict(float)
            markup_excluded_operation_costs = defaultdict(float)

            operations_data = self._production_cost_data['operations']

            if all(isinstance(op['item'], int)
                   for op in job_operations.values()):
                job_operations_data = operations_data.loc[
                    [op['item'] for op in job_operations.values()]
                ]
            else:
                job_operations_data = operations_data[
                    operations_data[['operation', 'operation_item']]
                    .apply(tuple, axis=1)
                    .isin(list(zip(
                        job_operations.keys(),
                        [c['item'] for c in job_operations.values()]
                    )))
                ]

            job_operations_data = job_operations_data.sort_values(
                by=['excluded_from_markup', 'operation_item', 'endpoint']
            )

            for _, group in job_operations_data.groupby(level=0, sort=False):
                if group['active'].drop_duplicates().item():
                    answer = job_operations.get(
                        group['operation_id'].drop_duplicates().item(),
                        job_operations.get(group['operation']
                                           .drop_duplicates()
                                           .item())
                    )['answer']
                    job_operation_cost = self.calculate_operation_item_cost(
                        operation_data=group,
                        quantity=quantity,
                        num_pieces=num_pieces,
                        num_cover_pieces=num_cover_pieces,
                        num_press_sheets=num_press_sheets,
                        num_cover_press_sheets=num_cover_press_sheets,
                        answer=answer
                    )
                    marked_up = not group['excluded_from_markup'].all()

                    operation_str = (
                        group['operation'].drop_duplicates().item() + ' ('
                        + group['operation_item'].drop_duplicates().item() + ')'
                    )

                    if marked_up:
                        markup_included_operation_costs[
                            operation_str
                        ] = job_operation_cost
                        component_costs['markup_included_operation_costs'] += (
                            job_operation_cost
                        )
                    else:
                        markup_excluded_operation_costs[
                            operation_str
                        ] = job_operation_cost
                        component_costs['markup_excluded_operation_costs'] += (
                            job_operation_cost
                        )

            if self._verbose:
                iim = component_costs['markup_included_operation_costs']
                efm = component_costs['markup_excluded_operation_costs']

                if markup_included_operation_costs:
                    print(f"Marked-up operations costs: ${iim:,.2f}")
                    for key, value in markup_included_operation_costs.items():
                        print(f'\t{key}: ${value:,.2f}')
                if markup_excluded_operation_costs:
                    print(f"Marked-excluded operations costs: ${efm:,.2f}")
                    for key, value in markup_excluded_operation_costs.items():
                        print(f'\t{key}: ${value:,.2f}')
                print(f"Total operations costs: ${iim + efm:,.2f}")

        if (finished_width, finished_height) not in self.preset_dimensions:
            # TODO: add operation cost for custom dimensions
            pass

        for key, value in component_costs.items():
            if hasattr(value, 'item'):
                component_costs[key] = value.item()

        return component_costs

    def _calculate_lf_component_costs(
        self,
        **config: dict[str, Y]
    ) -> dict[str, float]:
        """Calculate component costs for a large format (LF) print job.

        This method computes costs for various components of a large format
        print job, including substrates, laminates, devices, and operations,
        based on the provided configuration parameters.

        :param config: Configuration dictionary specifying job parameters:
            - quantity (int): The total number of units.
            - versions (int): The total number of versions.
            - quantity_per_version (int): The total number of units per version.
            - device: The device to use in production. If not specified, the
                  system offering's default device is used.
            - sides (int): The number of sides per unit. Defaults to 1.
            - final_width (float): The final width of each unit.
            - final_height (float): The final height of each unit.
            - print_substrate (str): The print substrate used.
            - mount_substrate (str): The mount substrate used, if any.
            - front_laminate (str): The front laminate used, if any.
            - back_laminate (str): The back laminate used, if any.
            - operations: Dictionary of operations and corresponding operation
                  items, if any.
            - turnaround_time (str): The number of production days requested by
                the customer.
            - proof (str): The proof type requested by the customer.
        :type config: dict[str, Y]
        :returns: A dictionary with calculated costs for each component.
        :rtype: dict[str, float]

        .. note::
            This method relies on production cost data loaded through
            `load_calculator`. Ensure that the cost data is loaded prior to
            calling this method to avoid errors.
        """
        config = {key: value
                  for key, value in config.items()
                  if value is not None}

        versions = config.get('versions', 1)
        quantity = config.get(
            'quantity',
            config.get('quantity_per_version')
        ) * versions
        final_width = config.get(
            'final_width',
            (self.preset_dimensions
             .loc[config['preset_dimensions'], 'final_width'])
        )
        final_height = config.get(
            'final_height',
            (self.preset_dimensions
             .loc[config['preset_dimensions'], 'final_height'])
        )
        num_sides = config.get('sides', 1)

        attrition_qty = ceil(quantity * self._qty_attrition_percent)
        quantity += attrition_qty

        if 'print_substrate' in config:
            print_substrate = config['print_substrate']
            print_substrate_b = (None
                                 if (self
                                     ._production_cost_data['print_substrates']
                                     .query('print_substrate == '
                                            '@print_substrate')['two_sided']
                                     .drop_duplicates()
                                     .item()) or num_sides == 1
                                 else print_substrate)
        else:
            print_substrate = config['print_substrate_a']
            print_substrate_b = config['print_substrate_b']

        component_costs = defaultdict(float)

        # Calculate print substrate costs.
        print_substrate_data = self._production_cost_data['print_substrates']
        if isinstance(print_substrate, int):
            job_print_substrate_data = (print_substrate_data
                                        .loc[print_substrate]
                                        .to_dict())
        else:
            job_print_substrate_data = print_substrate_data[
                print_substrate_data['print_substrate'] == print_substrate
            ].iloc[0:1].squeeze().to_dict()

        (square_feet,
         total_square_feet,
         linear_feet) = _calculate_square_feet(
            final_width=final_width,
            final_height=final_height,
            job_print_substrate_data=job_print_substrate_data
        )

        cost_basis = total_square_feet * quantity

        component_costs['print_substrate_cost'] += (
            job_print_substrate_data['square_foot_price'] * cost_basis
        )

        if print_substrate_b:
            if isinstance(print_substrate_b, int):
                job_print_substrate_b_data = (print_substrate_data
                                              .loc[print_substrate_b]
                                              .to_dict())
            else:
                job_print_substrate_b_data = print_substrate_data[
                    print_substrate_data['print_substrate'] == print_substrate_b
                ].squeeze().to_dict()

            component_costs['print_substrate_b_cost'] += (
                job_print_substrate_b_data['square_foot_price'] * cost_basis
            )

        # If the item has a mount substrate, calculate its costs.
        if config.get('mount_substrate') is not None:
            mount_substrate_data = self._production_cost_data[
                'mount_substrates'
            ]
            mount_substrate = config['mount_substrate']
            if isinstance(mount_substrate, int):
                job_mount_substrate_data = (mount_substrate_data
                                            .loc[mount_substrate]
                                            .to_dict())
            else:
                job_mount_substrate_data = mount_substrate_data[
                    mount_substrate_data['mount_substrate'] == mount_substrate
                ].squeeze().to_dict()

            component_costs['mount_substrate_cost'] += (
                job_mount_substrate_data['square_foot_price'] * cost_basis
            )

        # Calculate laminate costs.
        for laminate_side in ['front_laminate', 'back_laminate']:
            laminate = config.get(laminate_side)
            if laminate is not None:
                laminate_data = self._production_cost_data[f'{laminate_side}s']
                if isinstance(laminate, int):
                    job_laminate_data = laminate_data.loc[laminate].to_dict()
                else:
                    job_laminate_data = laminate_data[
                        laminate_data['laminate'] == config[laminate_side]
                    ].squeeze().to_dict()

                component_costs[laminate_side] += (
                    job_laminate_data['square_foot_price'] * cost_basis
                )

        # Calculate version, device, and ink costs.
        job_device_data = self._production_cost_data['devices']
        component_costs['version_cost'] = (job_device_data['version_price']
                                           .drop_duplicates()
                                           .item()) * versions

        cost_basis = num_sides * square_feet * quantity

        device_cost_table = (job_device_data[['price', 'flatprice', 'endpoint',
                                              'point_price', 'point_flatprice']]
                             .drop_duplicates(ignore_index=True))

        ink_cost_table = (job_device_data[['ink_price', 'ink_flatprice',
                                           'ink_endpoint', 'ink_point_price',
                                           'ink_point_flatprice']]
                          .rename(columns={col: col.replace('ink_', '')
                                           for col in job_device_data.columns})
                          .drop_duplicates(ignore_index=True))

        component_costs['device_cost'] += _interpolate_cost_table(
            cost_table=device_cost_table,
            cost_basis=cost_basis
        )
        component_costs['device_cost'] += _interpolate_cost_table(
            cost_table=ink_cost_table,
            cost_basis=cost_basis
        )

        # Calculate cutting costs.
        component_costs['cutting_cost'] += _interpolate_cost_table(
            cost_table=self._production_cost_data['cutting'],
            cost_basis=linear_feet * quantity
        )

        if self._verbose:
            print(f"Version cost: ${component_costs['version_cost']:,.2f}")
            print(f"Device cost: ${component_costs['device_cost']:,.2f}")
            print(f"Cutting cost: ${component_costs['cutting_cost']:,.2f}")
            params = list(dict.fromkeys(
                [x.replace('_type', '').replace('_weight', '')
                 for x in (LF_PRINTING_OPERATIONS
                           if 'print_substrate' in config
                           else OI_LF_PRINTING_OPERATIONS)
                 if not any(d in x for d in ['width', 'height', 'pages'])]
            ))

            for param in params:
                desc = ' '.join(param.split('_')).capitalize()
                print(f"{desc} cost: ${component_costs[param + '_cost']:,.2f}")

        # Calculate operation costs.
        job_operations = config.get('operations')

        if job_operations:
            markup_included_operation_costs = defaultdict(float)
            markup_excluded_operation_costs = defaultdict(float)

            operations_data = self._production_cost_data['operations']

            if all(isinstance(op['item'], int)
                   for op in job_operations.values()):
                job_operations_data = operations_data.loc[
                    [op['item'] for op in job_operations.values()]
                ]
            else:
                job_operations_data = operations_data[
                    operations_data[['operation', 'operation_item']]
                    .apply(tuple, axis=1)
                    .isin(list(zip(
                        job_operations.keys(),
                        [c['item'] for c in job_operations.values()]
                    )))
                ]

            job_operations_data = job_operations_data.sort_values(
                by=['excluded_from_markup', 'operation_item', 'endpoint']
            )

            for _, group in job_operations_data.groupby(level=0, sort=False):
                if group['active'].drop_duplicates().item():
                    job_operation_cost = self.calculate_operation_item_cost(
                        operation_data=group,
                        quantity=quantity,
                        square_feet=square_feet * quantity,
                        left_side=final_width * (group['default_left_side']
                                                 .drop_duplicates()
                                                 .item()),
                        right_side=final_width * (group['default_right_side']
                                                  .drop_duplicates()
                                                  .item()),
                        top_side=final_height * (group['default_top_side']
                                                 .drop_duplicates()
                                                 .item()),
                        bottom_side=final_height * (group['default_bottom_side']
                                                    .drop_duplicates()
                                                    .item())
                    )
                    marked_up = not group['excluded_from_markup'].all()

                    operation_str = (
                        group['operation'].drop_duplicates().item() + ' ('
                        + group['operation_item'].drop_duplicates().item() + ')'
                    )

                    if marked_up:
                        markup_included_operation_costs[
                            operation_str
                        ] = job_operation_cost
                        component_costs['markup_included_operation_costs'] += (
                            job_operation_cost
                        )
                    else:
                        markup_excluded_operation_costs[
                            operation_str
                        ] = job_operation_cost
                        component_costs['markup_excluded_operation_costs'] += (
                            job_operation_cost
                        )

            if self._verbose:
                iim = component_costs['markup_included_operation_costs']
                efm = component_costs['markup_excluded_operation_costs']

                if markup_included_operation_costs:
                    print(f"Marked-up operations costs: ${iim:,.2f}")
                    for key, value in markup_included_operation_costs.items():
                        print(f'\t{key}: ${value:,.2f}')
                if markup_excluded_operation_costs:
                    print(f"Marked-excluded operations costs: ${efm:,.2f}")
                    for key, value in markup_excluded_operation_costs.items():
                        print(f'\t{key}: ${value:,.2f}')
                print(f"Total operations costs: ${iim + efm:,.2f}")

        if (final_width, final_height) not in self.preset_dimensions:
            # TODO: add operation cost for custom dimensions
            pass

        for key, value in component_costs.items():
            if hasattr(value, 'item'):
                component_costs[key] = value.item()

        return component_costs

    def calculate_cost(self, **config: dict[str, Y]) -> float:
        """Calculate the total cost for a small format (SF) or large format (LF)
        print job.

        This method calculates the total cost of a print job by summing
        the individual component costs based on the product format and
        configuration provided.

        :param config: Configuration dictionary specifying job parameters. See
            the documentation for the `_calculate_sf_component_costs` and
            `_calculate_lf_component_costs` methods for a complete list of
            parameter options.
        :type config: dict[str, Y]
        :returns: The total cost of the print job.
        :rtype: float
        """
        return (sum(self._calculate_sf_component_costs(**config).values())
                if self._pjc_format == 'SMALL_FORMAT'
                else sum(self._calculate_lf_component_costs(**config).values()))


class PriceCalculator(CostCalculator):
    """A calculator for determining the price for a print job as shown to the
    customer.

    This class provides a Python implementation of Collaterate's price
    calculator logic. Costs are based on specifications like device,
    press sheets, inks, operations, etc., and the associated Collaterate
    production costs. A markup is applied on top of these costs to arrive at a
    final price

    Public Methods:
        - load_calculator: Loads production cost data and configures calculator
             settings.
        - calculate_operation_item_cost: Calculates the cost for a particular
              operation for a particular printing configuration.
        - calculate_cost: Calculates the cost for a print job.
        - calculate_price: Calculates the price shown to the customer.

    Instance Variables:
        - system_offering_site_share: The product (share) name.
        - system_offering_site_share_id: The product ID number.
        - system_offering: The parent product (system offering) name.
        - system_offering_id: The parent product ID number.
        - product_format: The format of the product (e.g., 'SMALL_FORMAT').
        - markup_type: The variable used to determine the markup value for a
            job; one of 'QUANTITY', 'SQUARE_FEET', 'UNIT_PRICE', or
            'THROUGHPUT'.
        - multi_pages: Indicates if the PJC allows for multiple pages. (This
            only pertains for small format products.)
        - default_config: The default configuration for the system offering.
        - load_status: The current state of the calculator instance. If PJC data
              has been retrieved and production cost data has been successfully
              loaded, then 'LOADED'; if PJC data has been successfully retrieved
              and production cost data has not been loaded, then 'READY TO
              LOAD'; or if PJC data has not been retrieved then 'NOT READY TO
              LOAD'.
        - system_markups: The markups and the corresponding quantities currently
            in Collaterate.
        - test_markups: New markups for testing purposes.
    """
    def __init__(
        self,
        system_offering_site_share: str | int,
        engine: Engine | None = None,
        verbose: bool = False
    ):
        """Initializes the PriceCalculator with database connection and product
        information.

        Sets up the calculator by loading product data related to the specified
        system offering and prepares it for future cost and/or price
        calculations. Determines and stores relevant identifiers for the system
        offering site_share, system offering, print job classification (PJC),
        product format, markup_type, and default configuration.

        :param system_offering_site_share: The name or ID of the product for
            which prices will be calculated.
        :type system_offering_site_share: str | int
        :param engine: SQLAlchemy engine for database connectivity. If this
            argument is not provided, a new engine instance will be created.
            Defaults to None.
        :type engine: Engine | None
        :param verbose: If True, prints progress during initialization. Defaults
            to False.
        :type verbose: bool
        """
        super().__init__(system_offering_site_share, engine, verbose)
        self.system_markups = None
        self.test_markups = None

    def load_calculator(
        self,
        as_admin: bool = False
    ) -> Self:
        """Retrieves PJC specifications and the associated costs, as well as
        system markups, making the class instance ready to calculate costs and
        prices.

        IDs for allowed devices, operation items, press sheets, inks, print
        substrates, mount substrates, and laminates are loaded as an
        intermediate variable `pjc_component_ids`, along with a Boolean
        `multi_pages` instance attribute that denotes whether the product has
         more than one page. (Examples of single-page products include business
        cards and brochures, while saddle-stitch booklets and calendars have
        more than one page.)

        These IDs are used by the `_load_production_cost_data` method to read in
        the corresponding cost table data, which are stored in the dict instance
        variable `_production_cost_data`.

        System markups are similarly read in as a pandas Series and stored in
        their own instance variable.

        :param as_admin: If true, include administrator materials/operations
            options that may be hidden to some users. Defaults to `False`.
        :type as_admin: bool, optional
        :returns: The initialized calculator instance with loaded cost data.
        :rtype: PriceCalculator

        :raises PJCNotFoundException: If a PJC is unable to be associated to the
             system offering when the PriceCalculator instance is instantiated.
        """
        if self.load_status == 'NOT READY TO LOAD':
            message = ('Unable to retrieve PJC information. Check database '
                       'connectivity and verify {soss} is the correct system '
                       'offering site share {id}.')
            message = (message.format(
                soss=self.system_offering_site_share,
                id='\b'
            )
                       if self.system_offering_site_share
                       else message.format(
                soss=self.system_offering_site_share_id,
                id='id'
            ))
            raise PJCNotFoundException(message)

        try:
            self._production_cost_data = self._load_production_cost_data(
                as_admin=as_admin
            )
            self.preset_dimensions = self._load_preset_dimensions()
            self.system_markups = self._load_system_markups()
            self.load_status = 'LOADED'
        except (KeyError, ValueError) as e:
            if self._verbose:
                print('Error loading calculator:')
            print(e)

        return self

    def _load_system_markups(self) -> pd.Series:
        """Load and return markup percentages for the system offering site
        share.

        This method retrieves markup percentages from the database for the
        specified system offering site share and organizes them into a series
        indexed by endpoint values. The markups represent scaling factors
        applied to the cost calculation at specific endpoints.

        If the system offering site share does not have an associated set of
        markup records, or if those records are missing, the method attempts to
        substitute system offering site share markups with system offering
        markups.

        :returns: A pandas Series with markup percentages for each endpoint,
            where the index is the endpoint and the value is the markup
            percentage.
        :rtype: pd.Series
        """
        query_path = QUERY_DIR / 'load_system_markups.sql'
        with open(query_path, 'r', encoding='utf-8') as file:
            query = Template(file.read()).render(
                soss_id=self.system_offering_site_share_id
            )

        with self._engine.begin() as connection:
            data = pd.read_sql_query(
                sql=text(query),
                con=connection,
                index_col='endpoint'
            )

        return data.squeeze(1)

    def calculate_price(
        self,
        test_markups: bool = False,
        **config: dict[str, Y]
    ) -> float:
        """Calculate the total price for a small format (SF) or large format
        (LF) print job.

        This method calculates the total price of a print job scaling the
        individual component costs (calculated based on the product format and
        configuration provided) by a markup (when appropriate) and summing
        the results.

        :param test_markups: Whether to use testing markups, as opposed to
            existing system markups. If test markups have not been configured,
            system markups are used instead. Defaults to `False`.
        :param config: Configuration dictionary specifying job parameters. See
            the documentation for the `_calculate_sf_component_costs` and
            `_calculate_lf_component_costs` methods for a complete list of
            parameter options.
        :type config: dict[str, Y]
        :returns: The total price of the print job.
        :rtype: float
        """
        quantity = config['quantity']

        component_costs = (self._calculate_sf_component_costs(**config)
                           if self._pjc_format == 'SMALL_FORMAT'
                           else self._calculate_lf_component_costs(**config))

        iim_costs = sum([value
                         for key, value in component_costs.items()
                         if key != 'markup_excluded_operation_costs'])
        efm_costs = component_costs['markup_excluded_operation_costs']

        markups = (self.test_markups
                   if test_markups and self.test_markups is not None
                   else self.system_markups)

        if self._markup_type == 'UNIT_PRICE':
            price = markups.loc[
                markups.index[markups.index <= quantity].max()
            ].item()
            markup = price / (iim_costs + efm_costs) - 1.0
        else:
            if self._markup_type == 'QUANTITY':
                markup_basis = quantity
            elif self._markup_type == 'SQUARE_FEET':
                print_substrate_data = self._production_cost_data[
                                           'print_substrates'
                                       ]

                markup_basis, _, _ = _calculate_square_feet(
                    final_width=config['final_width'],
                    final_height=config['final_height'],
                    job_print_substrate_data=print_substrate_data[
                        print_substrate_data['print_substrate']
                        == config.get(
                            'print_substrate',
                            config.get('print_substrate_a')
                        )
                    ].iloc[0:1].squeeze().to_dict()
                )
                markup_basis *= quantity

            if len(markups) == 1:
                markup = markups.item()
            else:
                markup = _get_markup_percent(markups, markup_basis)

            price = iim_costs * (1.0 + markup) + efm_costs

        final_price = price * (1.0 + self._site_markup_percent)

        if self._verbose:
            costs = sum(component_costs.values())
            margin = price - costs
            final_margin = final_price - costs
            print(f'Marked-up costs: ${iim_costs:,.2f}')
            print(f'Markup-excluded costs: ${efm_costs:,.2f}')
            print(f'Markup percent: {markup:.1%}')
            print(f'Price: ${price:,.2f}')
            print(f'Margin: ${margin:,.2f}')
            print(f'Margin percent: {margin / price:.1%}')
            print(f'Site markup percent: {self._site_markup_percent:.1%}')
            print(f'Final price: ${final_price:,.2f}')
            print(f'Site margin: ${price * self._site_markup_percent:,.2f}')
            print(f'Final margin percent: {final_margin / final_price:.1%}')

        return final_price


class CustomHelpFormatter(argparse.RawTextHelpFormatter):
    """Subclass of argparse.RawTextHelpFormatter to format command line help
    output that doesn't look terrible.
    """
    def _metavar_formatter(
        self,
        action: argparse.Action,
        default_metavar: str = '\b'
    ) -> Callable[[int], tuple[str]]:
        def format(metavar_result: str, tuple_size: int) -> tuple[str]:
            if isinstance(metavar_result, tuple):
                return metavar_result
            else:
                return (metavar_result,) * tuple_size

        if action.metavar is not None:
            result = action.metavar
        elif action.choices is not None:
            choice_strs = [str(choice) for choice in action.choices]
            result = '{%s}' % ','.join(choice_strs)
        else:
            result = default_metavar

        return partial(format, result)

    def _format_args(
        self,
        action: argparse.Action,
        default_metavar: str = '\b'
    ) -> str:
        get_metavar = self._metavar_formatter(action, default_metavar)
        if action.nargs is None:
            result = '%s' % get_metavar(1)
        elif action.nargs == '?':
            result = '[%s]' % get_metavar(1)
        elif action.nargs == '*':
            metavar = get_metavar(1)
            if len(metavar) == 2:
                result = '%s' % metavar
            else:
                result = '%s' % metavar
        elif action.nargs == '+':
            result = '%s' % get_metavar(2)
        elif action.nargs == '...':
            result = '...'
        elif action.nargs == 'A...':
            result = '%s ...' % get_metavar(1)
        elif action.nargs == '==SUPPRESS==':
            result = ''
        else:
            try:
                formats = ['%s' for _ in range(action.nargs)]
            except TypeError:
                raise ValueError("invalid nargs value") from None
            result = ' '.join(formats) % get_metavar(action.nargs)
        return result

    def _format_action(self, action: argparse.Action) -> str:
        help_position = min(
            self._action_max_length + 2,
            self._max_help_position
        )
        help_width = max(self._width - help_position, 11)
        action_width = help_position - self._current_indent - 2
        action_header = self._format_action_invocation(action)

        if not action.help:
            tup = self._current_indent, '', action_header
            action_header = '%*s%s\n' % tup

        else:  # len(action_header) <= action_width:
            tup = self._current_indent, '', action_width, action_header
            action_header = '%*s%-*s  ' % tup
            indent_first = 0

        parts = [action_header]

        if action.help and action.help.strip():
            help_text = self._expand_help(action)
            if help_text:
                help_lines = self._split_lines(help_text, help_width)
                parts.append('%*s%s\n' % (indent_first, '', help_lines[0]))
                for line in help_lines[1:]:
                    parts.append('%*s%s\n' % (help_position, '', line))

        elif not action_header.endswith('\n'):
            parts.append('\n')

        for subaction in self._iter_indented_subactions(action):
            parts.append(self._format_action(subaction))

        return self._join_parts(parts)


class ParseKwargs(argparse.Action):
    """Conceptual class to parse operations command line input, converting a
    list of concatenated operations/operation items into a dict of operations
    to be passed to the price calculator.
    """
    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: list[str],
        option_string: str | None = None
    ) -> None:
        """The method defining the class's functionality.

        :param parser: The argument parser instance to act upon.
        :type parser: argparse.ArgumentParser
        :param namespace: The namespace instance to read from and write to.
        :type namespace: argparse.Namespace
        :param list[str] values: The command line arguments to parse.
        :param option_string: The option string that was used to invoke this
            action. The option_string argument is optional, and will be absent
            if the action is associated with a positional argument.
        :type option_string: str or None
        """
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split('=')
            if ',' in value:
                value = dict(
                    heading=None,
                    item=value.split(',')[0].strip(),
                    answer=int(value.split(',')[1].strip())
                )
            else:
                value = dict(
                    heading=None,
                    item=value.strip(),
                    answer=None
                )
            try:
                key = int(key)
                value['item'] = int(value['item'])
            except ValueError:
                pass

            getattr(namespace, self.dest)[key] = value


class CustomArgumentParser(argparse.ArgumentParser):
    """Subclass of ArgumentParser to override the default help message."""
    def __init__(self, *args, **kwargs):
        if 'add_help' in kwargs and kwargs['add_help']:
            kwargs['add_help'] = False  # Disable automatic help addition.
        super().__init__(*args, **kwargs)

        # Manually add a modified help argument.
        default_prefix = ('-'
                          if '-' in self.prefix_chars
                          else self.prefix_chars[0])
        self.add_argument(
            default_prefix + 'h', default_prefix * 2 + 'help',
            action='help', default=argparse.SUPPRESS,
            help='\tshow this help message and exit'
        )


if __name__ == '__main__':
    parser = CustomArgumentParser(
        prog='python brochure_price_calculator_v2.py',
        formatter_class=CustomHelpFormatter,
        description='Calculate Brochure price',
        epilog='As an example:\n'
               + 'python brochure_price_calculator_v2.py --quantity 25 '
               + '--finished-width 8.5 --finished-height 11.0 '
               + '--press-sheet-type "Coated Matte - White" '
               + '--press-sheet-weight "100# Text" --press-sheet-color "White" '
               + '--side1-ink "Full Color" --side2-ink "Full Color" '
               + '--operations "Folding"="Single Fold" '
               + '"Scoring-Only"="Parallel Scores" "Shrink Wrap"="Bundles,5"\n'
               + 'Or, using IDs:\n'
               + 'python brochure_price_calculator_v2.py --quantity 25 '
               + '--preset-dimensions 814 --press-sheet 894 --side1-ink 146 '
               + '--side2-ink 146 --operations 1=34 2=4 14=42,5',
        add_help=False
    )
    parser.add_argument(
        '-q', '--quantity',
        type=int,
        metavar='\b',
        help='\t\tquantity ordered (Default: 100)',
        default=100
    )
    parser.add_argument(
        '-v', '--versions',
        type=int,
        metavar='\b',
        help='\t\tversions ordered (Default: 1)',
        default=1
    )
    parser.add_argument(
        '-x', '--quantity-per-version',
        type=int,
        metavar='\b',
        help='\tquantity per version (Default: None)',
        default=None
    )
    parser.add_argument(
        '-d', '--preset-dimensions',
        type=int,
        metavar='\b',
        help='\tpreset dimensions ID (Default: None)',
        default=None
    )
    parser.add_argument(
        '-w', '--finished-width',
        type=float,
        metavar='\b',
        help='\tfinished width of the Brochure (Default: 8.5)',
        default=8.5
    )
    parser.add_argument(
        '-l', '--finished-height',
        type=float,
        metavar='\b',
        help='\tfinished height of the Brochure (Default: 11.0)',
        default=11.0
    )
    parser.add_argument(
        '-t', '--press-sheet-type',
        metavar='\b',
        help='\tpress sheet type (Default: "Coated Matte - White")',
        default='Coated Matte - White'
    )
    parser.add_argument(
        '-y', '--press-sheet-weight',
        metavar='\b',
        help='\tpress sheet weight (Default: "100# Text")',
        default='100# Text'
    )
    parser.add_argument(
        '-c', '--press-sheet-color',
        metavar='\b',
        help='\tpress sheet color (Default: "White")',
        default='100# Text'
    )
    parser.add_argument(
        '-p', '--press-sheet',
        type=int,
        metavar='\b',
        help='\t\tpress sheet ID (Default: None)',
        default=None
    )
    parser.add_argument(
        '-f', '--side1-ink-type',
        metavar='\b',
        help='\tfront ink type to use (Default: "Full Color")',
        default='Full Color'
    )
    parser.add_argument(
        '-g', '--side1-ink',
        type=int,
        metavar='\b',
        help='\t\tfront ink ID (Default: None)',
        default=None
    )
    parser.add_argument(
        '-i', '--side2-ink-type',
        metavar='\b',
        help='\tback ink type to use (Default: None)',
        default=None
    )
    parser.add_argument(
        '-j', '--side2-ink',
        type=int,
        metavar='\b',
        help='\t\tback ink ID (Default: None)',
        default=None
    )
    parser.add_argument(
        '-o', '--operations',
        metavar='\b',
        action=ParseKwargs,
        help='\t\tadd-on operations and operations items in '
             + '"OPERATION"="OPERATION ITEM,ANSWER" or '
             + 'OPERATION_ID=OPERATION_ITEM_ID,ANSWER format',
        nargs='*'
    )

    args = parser.parse_args()

    if len(sys.argv) == 1:
        print('No arguments provided. Creating and saving the PriceCalculator '
              'object...')
        try:
            with open(CALC_PATH, 'wb') as f:
                pickle.dump(PriceCalculator(
                    system_offering_site_share=SOSS_ID,
                    engine=redshift_engine()
                ).load_calculator(), f)

            print(f'Calculator saved successfully to {CALC_PATH}.')
        except Exception as e:
            print('Error saving calculator:', e)
    else:
        try:
            with open(CALC_PATH, 'rb') as f:
                calc = pickle.load(f)

            output = json.dumps(dict(
                value=round(calc.calculate_price(
                    quantity=args.quantity,
                    versions=args.versions,
                    quantity_per_version=args.quantity_per_version,
                    finished_width=args.finished_width,
                    finished_height=args.finished_height,
                    preset_dimensions=args.preset_dimensions,
                    press_sheet_type=args.press_sheet_type,
                    press_sheet_weight=args.press_sheet_weight,
                    press_sheet_color=args.press_sheet_color,
                    press_sheet=args.press_sheet,
                    side1_ink_type=args.side1_ink_type,
                    side1_ink=args.side1_ink,
                    side2_ink_type=args.side2_ink_type,
                    side2_ink=args.side2_ink,
                    operations=args.operations
                ), 2),
                currency='USD'
            ))

            print(output)

        except FileNotFoundError:
            print('File "BrochurePriceCalculator.pkl" not found.')
        except Exception as e:
            print('Invalid configuration parameters:', e)
