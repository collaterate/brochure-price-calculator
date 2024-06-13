import argparse
import re
import warnings
from collections.abc import Callable
from functools import partial

import numpy as np
import pandas as pd
from rectpack import (MaxRectsBl,
                      MaxRectsBssf,
                      MaxRectsBaf,
                      MaxRectsBlsf,
                      newPacker)
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.special import softmax
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sqlalchemy import text
from sqlalchemy.engine import Engine

warnings.filterwarnings('ignore')

SEED = 42
QUANTITIES = [1, 10, 25, 50, 100, 200, 500, 1000, 2500, 5000]
ALLOWED_DIMS = [{5.5, 8.5},
                {8.5, 11.0},
                {9.0, 12.0},
                {8.5, 14.0},
                {11.0, 17.0},
                {11.0, 25.5}]
ALLOWED_AREAS = [width * height for width, height, in ALLOWED_DIMS]
QUANTITY_AREA_RATIOS = pd.DataFrame(
    data=[[40.0, 42.5, 45.0, 47.5],
          [2.75, 3.0, 3.25, 3.75],
          [1.375, 1.5, 1.9, 2.3],
          [1.0, 1.2, 1.5, 2.25],
          [0.75, 1.0, 1.425, 2.175],
          [0.55, 0.825, 1.5, 2.15],
          [0.375, 0.7, 1.25, 2.125],
          [0.25, 0.5, 0.875, 2.0],
          [0.175, 0.4, 0.75, 1.75],
          [0.15, 0.35, 0.625, 1.5]],
    columns=ALLOWED_AREAS[:2] + ALLOWED_AREAS[-2:],
    index=QUANTITIES
)
SHEET_NAMES = ['Coated Gloss - White;80# Text',
               'Coated Gloss - White;100# Text',
               'Coated Gloss - White;80# Cover',
               'Coated Gloss - White;100# Cover',
               'Coated Gloss - White;120# Cover',
               'Coated Matte - White;80# Text',
               'Coated Matte - White;100# Text',
               'Coated Matte - White;80# Cover',
               'Coated Matte - White;100# Cover',
               'Coated Matte - White;120# Cover',
               'Uncoated Smooth - White;70# Text',
               'Uncoated Smooth - White;80# Text',
               'Uncoated Smooth - White;100# Text',
               'Uncoated Smooth - White;80# Cover',
               'Uncoated Smooth - White;100# Cover',
               'Uncoated Smooth - White;130# Cover',
               'Uncoated 100% Recycled - White;100# Text',
               'Uncoated 100% Recycled - White;80# Cover',
               'Uncoated 100% Recycled - White;100# Cover',
               'Uncoated 100% Recycled - White;130# Cover',
               'Coated Semigloss 1 Side (C1S) - White;12 pt.',
               'Coated Semigloss 1 Side (C1S) - White;14 pt.',
               'Coated Semigloss 2 Sides (C2S) - White;12 pt.',
               'Coated Semigloss 2 Sides (C2S) - White;14 pt.',
               'Linen - White;70# Text',
               'Linen - White;110# Cover',
               'Uncoated Smooth - Natural;80# Text',
               'Uncoated Smooth - Natural;100# Cover',
               'Linen - Natural;100# Cover',
               'Felt Weave - Natural;100# Cover',
               'Felt Weave - White;80# Text',
               'Felt Weave - White;100# Cover']
OLD_PRICES = dict(
    zip(
        SHEET_NAMES,
        [s.assign(finished_area=lambda x: (x['Quantity']
                                           .str
                                           .split(' x ', expand=True)
                                           .astype(float)
                                           .prod(axis=1)))
         .drop('Quantity', axis=1)
         .set_index('finished_area')
         for s
         in pd.read_excel('old_brochure_prices.xlsx', sheet_name=None).values()]
    )
)
for old_price_df in OLD_PRICES.values():
    old_price_df.columns.name = 'quantity'

OLD_INK_PRICES = {
    key: (value
          .assign(finished_area=lambda x: (x['Quantity']
                                           .str
                                           .split(' x ', expand=True)
                                           .astype(float)
                                           .prod(axis=1)))
          .drop('Quantity', axis=1)
          .set_index('finished_area'))
    for key, value
    in pd.read_excel('ink_brochure_prices.xlsx', sheet_name=None).items()
}
OLD_INK_PRICES['Full Color;Full Color'] = OLD_PRICES[
    'Coated Matte - White;100# Text'
].copy()
for old_ink_price_df in OLD_PRICES.values():
    old_ink_price_df.columns.name = 'quantity'

OPTIMAL_UNIT_PRICE = 0.4405877130399652
PRESS_SHEETS = pd.read_csv('press_sheets_costs.csv')
OPERATIONS = pd.read_csv('operations_costs.csv')
INK_ADJ = pd.read_excel(
    'ink_price_differences.xlsx',
    sheet_name=None,
    index_col='quantity'
)
INK_ADJ['Black Only;Full Color'] = INK_ADJ['Full Color;Black Only']

def estimate_demand_curve(
    df: pd.DataFrame,
    var: str,
    nbins: int = 10
) -> tuple[float, float]:
    """Calculates the optimal price based on historical sales data and plots the
    underlying estimated demand curve and revenue function.

    :param pd.DataFrame df: The wide DataFrame of sales data.
    :param str var: The variable to group by while constructing the histogram.
    :param int nbins: The number of bins used while constructing the histogram.
    :return: The estimated optimal price and the corresponding correlation
             coefficient.
    :rtype: tuple[float, float]
    """

    def revenue(p: float, reg: LinearRegression) -> float:
        """Uses a linear regression to predict revenue earned at a
        particular price.

        :param float p: The price of interest.
        :param LinearRegression reg: The linear regression object.
        :return: The revenue predicted for this price.
        :rtype: float
        """
        return reg.coef_.item() * p ** 2 + reg.intercept_ * p

    hist, bin_edges = np.histogram(
        [item
         for l in [[p] * q
                   for p, q in df[[var, 'quantity']].to_records(index=False)]
         for item in l],
        bins=nbins
    )
    bin_midpoints = (bin_edges[1:] + bin_edges[:-1]) / 2

    reg = (LinearRegression()
           .fit(bin_midpoints[hist > 0].reshape(-1, 1), hist[hist > 0]))
    r2 = reg.score(bin_midpoints[hist > 0].reshape(-1, 1), hist[hist > 0])
    prices = np.linspace(0.0, -reg.intercept_ / reg.coef_.item(), 100)

    prices = dict(zip(prices, revenue(prices, reg)))
    optimal_price_per_thousand = max(prices, key=prices.get)

    return optimal_price_per_thousand, r2


def get_optimal_price(engine: Engine) -> tuple[float, float, float]:
    rng = np.random.default_rng(SEED)
    with engine.begin() as connection:
        df = pd.read_sql_query(
            text("""WITH iit AS (SELECT i.id AS ink_id,
                                        it.id AS ink_type_id,
                                        it.name AS ink_type_name
                                 FROM coll_src.inks AS i
                                 INNER JOIN coll_src.ink_types AS it
                                 ON i.type_id = it.id),
                         pspstpsw AS (SELECT ps.id AS press_sheet_id,
                                      pst.id AS press_sheet_type_id,
                                      pst.name AS press_sheet_type_name,
                                      psw.id AS press_sheet_weight_id,
                                      psw.name AS press_sheet_weight_name
                               FROM coll_src.press_sheets AS ps
                               INNER JOIN coll_src.press_sheet_types AS pst
                               ON ps.type_id = pst.id
                               INNER JOIN coll_src.press_sheet_weights AS psw
                               ON ps.weight_id = psw.id)

                     SELECT oi.id AS order_item_id,
                            oi.name AS order_item_name,
                            oi.product_id,
                            o.created_on AS order_date,
                            COALESCE(oi.total_adjusted_quantity, oipjc.qty)
                                AS quantity,
                            CAST(oi.price AS FLOAT) AS price,
                            CAST(oipjc.total_price_override_amount AS FLOAT)
                                AS total_price_override_amount,
                            CAST(oipjc.job_cost_price AS FLOAT) AS job_cost,
                            CAST(oipjc.operations_price AS FLOAT)
                                AS operations_cost,
                            CAST(oipjc.proof_price AS FLOAT) AS proof_cost,
                            CAST(oipjc.turnaround_price AS FLOAT)
                                AS turnaround_cost,
                            CAST(oipjc.markup_percent AS FLOAT) + 1.0
                                AS markup_rate,
                            CAST(oi.site_markup_percent AS FLOAT) / 100.0 + 1.0
                                AS site_markup_rate,
                            CAST(oipjc.finished_width AS FLOAT)
                                AS finished_width,
                            CAST(oipjc.finished_height AS FLOAT)
                                AS finished_height,
                            oipjc.pages,
                            pspstpsw.press_sheet_type_id,
                            pspstpsw.press_sheet_type_name,
                            pspstpsw.press_sheet_weight_id,
                            pspstpsw.press_sheet_weight_name,
                            cpspstpsw.press_sheet_type_id
                                AS cover_press_sheet_type_id,
                            cpspstpsw.press_sheet_type_name
                                AS cover_press_sheet_type_name,
                            cpspstpsw.press_sheet_weight_id
                                AS cover_press_sheet_weight_id,
                            cpspstpsw.press_sheet_weight_name
                                AS cover_press_sheet_weight_name,
                            s1iit.ink_type_id AS side1_ink_type_id,
                            s1iit.ink_type_name AS side1_ink_type_name,
                            s2iit.ink_type_id AS side2_ink_type_id,
                            s2iit.ink_type_name AS side2_ink_type_name,
                            cs1iit.ink_type_id AS cover_side1_ink_type_id,
                            cs1iit.ink_type_name AS cover_side1_ink_type_name,
                            cs2iit.ink_type_id AS cover_side2_ink_type_id,
                            cs2iit.ink_type_name AS cover_side2_ink_type_name,
                            oipjco.operation_id,
                            op.name AS operation_name,
                            oipjco.operation_item_id,
                            opi.name AS operation_item_name,
                            COALESCE(CAST(oipjco.price AS FLOAT), 0.0)
                                AS operation_item_cost,
                            op.excluded_from_markup,
                            CAST(ct.price AS FLOAT) AS operation_item_base_cost,
                            ct.flatprice AS flat_price
                     FROM coll_src.order_items AS oi
                     INNER JOIN coll_src.system_offering_site_shares AS soss
                     ON oi.system_offering_site_share_id = soss.id
                         AND soss.id = 10
                     INNER JOIN coll_src.orders AS o
                     ON oi.order_id = o.id AND o.status_id != 4
                         AND o.created_on > '2021-01-01'
                         AND DATEDIFF(SECOND, o.created_on, oi.created_on) < 2
                     INNER JOIN coll_src.order_item_print_job_classifications
                         AS oipjc
                     ON oi.order_item_print_job_classification_id = oipjc.id
                     INNER JOIN pspstpsw
                     ON oipjc.press_sheet_id = pspstpsw.press_sheet_id
                     LEFT JOIN pspstpsw AS cpspstpsw
                     ON oipjc.cover_press_sheet_id = cpspstpsw.press_sheet_id
                     INNER JOIN iit AS s1iit
                     ON oipjc.side1_ink_id = s1iit.ink_id
                     LEFT JOIN iit AS s2iit
                     ON oipjc.side2_ink_id = s2iit.ink_id
                     LEFT JOIN iit AS cs1iit
                     ON oipjc.cover_side1_ink_id = cs1iit.ink_id
                     LEFT JOIN iit AS cs2iit
                     ON oipjc.cover_side2_ink_id = cs2iit.ink_id
                     LEFT JOIN
                         coll_src.order_item_print_job_classification_operations
                         AS oipjco
                     ON oipjc.id = oipjco.order_item_print_job_classification_id
                     LEFT JOIN coll_src.operations AS op
                     ON oipjco.operation_id = op.id
                     LEFT JOIN coll_src.operation_items AS opi
                     ON oipjco.operation_item_id = opi.id
                         AND opi.name NOT IN ('None', 'No')
                     LEFT JOIN coll_src.cost_tables AS ct
                     ON opi.cost_table_id = ct.id
                     WHERE NOT oi.cancelled
                         AND COALESCE(oi.project_id, oi.sales_estimate_id)
                             IS NULL
                         AND oi.order_item_type = 'OI'
                         AND CAST(oi.price AS FLOAT) > 0.0
                         AND (oi.product_id != 'SAMPLES'
                                  OR oi.product_id IS NULL)
                     ORDER BY oi.ordered_on;"""),
            con=connection,
            parse_dates=['order_date']
        ).fillna(np.nan).drop_duplicates(
            subset=['order_item_id', 'operation_id'],
            keep=False
        ).astype(
            {'excluded_from_markup': bool, 'flat_price': bool}
        ).drop(
            ['press_sheet_weight_id', 'press_sheet_weight_name',
             'cover_press_sheet_type_id', 'cover_press_sheet_type_name',
             'cover_press_sheet_weight_id', 'cover_press_sheet_weight_name',
             'cover_side1_ink_type_id', 'cover_side1_ink_type_name',
             'cover_side2_ink_type_id', 'cover_side2_ink_type_name'],
            axis=1
        )

    df.loc[df['operation_item_cost'].isna(), 'excluded_from_markup'] = False
    df = df[df['order_item_id'].isin(
        df.groupby('order_item_id')[['operations_cost', 'operation_item_cost']]
          .agg({'operations_cost': 'mean', 'operation_item_cost': 'sum'})
          .round(2)
          .query('operations_cost == operation_item_cost')
          .index
          .unique()
    )]

    operations_columns_index = df.columns.get_loc('operation_id')
    markup_column_index = [i
                           for i, v in enumerate(df.columns
                                                   .str
                                                   .endswith('markup_rate'))
                           if v][-1]

    costs = pd.pivot_table(
        data=df[['order_item_id',
                 'operation_id',
                 'operation_item_cost']].dropna(how='any')
                                        .astype({'order_item_id': int,
                                                 'operation_id': int}),
        values='operation_item_cost',
        index='order_item_id',
        columns='operation_id'
    )
    costs = costs.rename({col: f'operation_id_{col}_cost'
                          for col in costs.columns}, axis=1)

    excluded = pd.pivot_table(
        data=df[['order_item_id',
                 'operation_id',
                 'excluded_from_markup']].dropna().astype(int),
        values='excluded_from_markup',
        index='order_item_id',
        columns='operation_id',
        aggfunc='sum',
        fill_value=0
    ).astype(bool)
    excluded = excluded.rename({col: f'operation_id_{col}_efm'
                                for col in excluded.columns}, axis=1)

    wide_df = (df.iloc[:, :operations_columns_index]
                 .drop_duplicates()
                 .set_index('order_item_id')
                 .merge(
                      pd.pivot_table(
                          data=df[['order_item_id',
                                   'operation_id',
                                   'operation_item_id']].dropna().astype(int),
                          values='operation_item_id',
                          index='order_item_id',
                          columns='operation_id',
                          aggfunc=pd.Series.mode
                      ),
                      how='left',
                      left_index=True,
                      right_index=True
                  )
                 .merge(
                      pd.pivot_table(
                          data=df[['order_item_id',
                                   'operation_name',
                                   'operation_item_name']].dropna(),
                          values='operation_item_name',
                          index='order_item_id',
                          columns='operation_name',
                          aggfunc=pd.Series.mode
                      ),
                      how='left',
                      left_index=True,
                      right_index=True
                  )
                 .merge(
                      costs,
                      how='left',
                      left_on='order_item_id',
                      right_index=True
                  )
                 .merge(
                      excluded,
                      how='left',
                      left_on='order_item_id',
                      right_index=True
                  ))

    if wide_df.index.name == 'order_item_id':
        operations_columns_index -= 1
        markup_column_index -= 1

    operations = (df[['operation_id',
                      'operation_name']].dropna(how='any')
                                        .drop_duplicates()
                                        .astype({'operation_id': int,
                                                 'operation_name': str})
                                        .sort_values('operation_id')
                                        .to_records(index=False))
    operations = [(i, name, f'operation_id_{i}_cost', f'operation_id_{i}_efm')
                  for i, name in operations]

    wide_df = wide_df[list(wide_df.columns[:operations_columns_index])
                      + [item
                         for operation in operations
                         for item in operation]].rename(
                             {col: (f'operation_id_{col}'
                                    if isinstance(col, int)
                                    else f'{col.replace(' ', '_').lower()}')
                              for col in wide_df.columns}, axis=1
                         )

    costs = wide_df[[col
                     for col in wide_df.columns
                     if col.startswith('operation_id_')
                        and col.endswith('_cost')]]
    markup_costs = costs.mul(
        wide_df['markup_rate'] * wide_df['site_markup_rate'],
        axis=0
    )

    wide_df['price'] = (wide_df['job_cost'] * wide_df['markup_rate']
                        + costs.where(
                             wide_df[[col
                                      for col in wide_df.columns
                                      if col.endswith('_efm')]].to_numpy(),
                             markup_costs
                        ).sum(axis=1))

    standard = wide_df.iloc[:, markup_column_index + 1:].value_counts(
                                                              dropna=False
                                                          )
    standard = {key: value
                for key, value in
                dict(zip(standard.index.names, standard.idxmax())).items()
                if key in ['finished_width', 'finished_height', 'pages']
                or re.search('_id(_[0-9]+)?$', key)}

    standard_df = (wide_df.query(' and '.join([f'{key}.isna()'
                                               if not isinstance(value, str)
                                               and np.isnan(value)
                                               else f'{key} == {repr(value)}'
                                               for key, value
                                               in standard.items()
                                               if not key.startswith(
                                                              'operation_id_'
                                                          )]))
                   .iloc[:, :operations_columns_index])

    standard_df = standard_df[['order_date', 'quantity', 'price']].assign(
        price_per_thousand=lambda x: x['price'] / x['quantity'] * 1000
    )

    desc = pd.Series(
        [item
         for l in [[p] * q
                   for p, q
                   in standard_df[['price_per_thousand',
                                   'quantity']].to_records(index=False)]
         for item in l],
        name='price_per_thousand'
    ).describe().round(2)

    opt_price_df = []

    for nbins in range(5, 16):
        opt_price_df.append(
            [nbins,
             *estimate_demand_curve(
                 df=standard_df[standard_df['price_per_thousand'] <= 2.5
                                * desc['75%'] - 1.5 * desc['25%']],
                 var='price_per_thousand',
                 nbins=nbins
             )]
        )

    opt_price_df = pd.DataFrame(
        data=opt_price_df,
        columns=['nbins', 'price_per_thousand', 'r2']
    ).set_index('nbins')
    mc = rng.choice(
        a=opt_price_df['price_per_thousand'],
        size=1000000,
        p=softmax(opt_price_df['r2'])
    )

    dist = norm.fit(mc)
    x = np.linspace(
        opt_price_df['price_per_thousand'].min() * 0.95,
        opt_price_df['price_per_thousand'].min() * 1.05,
        1000
    )
    y = norm.pdf(x, *dist)
    return tuple([p / 1000
                  for p in [x[y.argmax()],
                            norm.ppf(0.05, *dist),
                            norm.ppf(0.95, *dist)]])


def pieces_per_sheet(
    w: float,
    h: float,
    W: float = 29.4375,
    H: float = 20.75,
    bleed: float = 0.125,
    edge: float = 0.125,
    collaterate: bool = True
) -> int:
    """Determines the number of pieces that will be cut from
    a larger press sheet.

    :param float w: The width of the piece in inches.
    :param float h: The height of the piece in inches.
    :param float W: The width of the press sheet in inches.
    :param float H: The height of the press sheet in inches.
    :param float bleed: The bleed width for the device.
    :param float edge: The piece edge width.
    :param bool collaterate: Whether to use Collaterate's press sheet logic.
    :return: The number of pieces per sheet.
    :rtype: int
    """
    rectangles = [(w + bleed + edge, h + bleed + edge)] * 100

    algo_results = []

    for algo in [MaxRectsBaf,
                 MaxRectsBlsf,
                 MaxRectsBl,
                 MaxRectsBssf][:1 if collaterate else None]:
        for orientation in [rectangles, [(r[1], r[0]) for r in rectangles]]:
            packer = newPacker(pack_algo=algo, rotation=False)
            packer.add_bin(W, H)
            for r in orientation:
                packer.add_rect(*r)
            packer.pack()

            algo_results.append(packer.rect_list())

    best_fit = max(algo_results, key=len)

    return len(best_fit)


def calculate_operation_price(
    opi_data: pd.DataFrame,
    num_press_sheets: int,
    num_pieces: int,
    answer: int | None = None
) -> float:
    """Estimates the cost of and determines an appropriate price increase for a
    particular operation.

    :param pd.DataFrame opi_data: The DataFrame containing cost data specific to
        that operation item (cost basis, unit costs, endpoints, etc.).
    :param int num_press_sheets: The number of press sheets used for the job.
    :param int num_pieces: The number of pieces involved.
    :param int answer: The customer's answer, if an operation requires
        additional to be specified (e.g., number of pieces per bundle in
        shrink-wrapping)
    :return: The price increase due to the operation.
    :rtype: float
    """
    cost_basis = opi_data['cost_basis'].drop_duplicates().item()

    if opi_data.shape[0] == 1:
        if opi_data['flatprice'].item():
            return opi_data['run_cost'].item()
        elif cost_basis == 'Cost Number of Sheets':
            cost = ((num_press_sheets - 1) * opi_data['run_cost']
                    + opi_data['setup_cost']).item()
        elif cost_basis == 'Cost Number of Pieces':
            cost = ((num_pieces - 1) * opi_data['run_cost']
                    + opi_data['setup_cost']).item()
        elif cost_basis == 'Cost Number of Pieces after Dividing by Answer':
            cost = ((np.ceil(num_pieces / answer) - 1) * opi_data['run_cost']
                    + opi_data['setup_cost']).item()
        else:
            cost = 0.0
    else:
        cost_basis_num = (num_press_sheets
                          if cost_basis == 'Cost Number of Sheets'
                          else num_pieces)

        if cost_basis_num >= opi_data['endpoint'].iloc[-1]:
            cost = ((cost_basis_num - opi_data['endpoint'].iloc[-1])
                    * opi_data['run_cost'].drop_duplicates().item()
                    + opi_data['accumulated_cost'].iloc[-1])
        else:
            lb = [e for e in opi_data['endpoint'] if e <= cost_basis_num]
            cost = ((cost_basis_num - lb[-1])
                    * opi_data.iloc[len(lb),
                                    opi_data.columns.get_loc('setup_cost')]
                    + opi_data.iloc[len(lb) - 1,
                                    opi_data.columns
                                    .get_loc('accumulated_cost')])

    return cost * interp1d(
        QUANTITIES,
        [1.71, 1.26, 1.46, 1.97, 2.7, 3.22, 3.45, 2.68, 2.35, 2.04],
        fill_value='extrapolate'
    )(num_pieces)


def calculate_price(
    quantity: int = 25,
    finished_width: float = 8.5,
    finished_height: float = 11.0,
    press_sheet_type: str | None = 'Coated Matte - White',
    press_sheet_weight: str | None = '100# Text',
    side1_ink_type: str | int = 'Full Color',
    side2_ink_type: str | int = 'Full Color',
    **operations: str | int
) -> float | None:
    """Calculates the price of a particular brochure configuration.

    :param int quantity: The number of brochures in the order (item).
    :param float finished_width: The width of the brochure.
    :param float finished_height: The height of the brochure.
    :param str press_sheet_type: The paper stock used.
    :param press_sheet_weight: The weight of the paper stock.
    :param str side1_ink_type: The type of ink used on the front of the
        brochure.
    :param side2_ink_type: The type of ink used on the back of the brochure.
    :type side2_ink_type: str or None
    :param operations: A sequence of additional operations.
    :type operations: dict[str, str]
    :return: The dollar price quoted to the customer.
    :rtype: float
    """
    # Check to make sure the press sheet type/weight combination is valid
    try:
        press_sheet_data = PRESS_SHEETS[
            (PRESS_SHEETS['press_sheet_type_name'] == press_sheet_type)
            & (PRESS_SHEETS['press_sheet_weight_name'] == press_sheet_weight)
        ].squeeze()
    except ValueError as e:
        print(type(e))
        print('Not a valid stock/weight combination')
        return

    # Apply a custom size surcharge, if appropriate
    if {finished_width, finished_height} not in ALLOWED_DIMS:
        operations['Custom Size Surcharge'] = 'Brochure Surcharge'

    # Estimate the number of press sheets used for the job
    num_press_sheets = np.ceil(
        quantity / pieces_per_sheet(
            finished_width,
            finished_height,
            *press_sheet_data[['unfinished_width',
                               'unfinished_height']].squeeze().astype(float)
        )
    )
    finished_area = finished_width * finished_height  # calculate finished area

    # Retrieve the grid of old/current prices for the given press sheet
    old_prices = OLD_PRICES.get(';'.join([press_sheet_type,
                                          press_sheet_weight]))

    # If current price data is not available, adjust the optimal (unit) price
    # based on quantity and finished area using the default ratios
    if old_prices is None:
        press_sheet_correction = RegularGridInterpolator(
            (QUANTITY_AREA_RATIOS.index, QUANTITY_AREA_RATIOS.columns),
            QUANTITY_AREA_RATIOS.to_numpy(),
            bounds_error=False,
            fill_value=None
        )((quantity, finished_area)).item()

        price = OPTIMAL_UNIT_PRICE * quantity * press_sheet_correction

    else:
        # The estimate for the optimal price for 100 units of the default
        # configuration is $44.06, which is ~7% less than the current price

        # Rescale the optimal unit price so that the percent price change for
        # 100 units of 8.5 x 11.0 is the same (-7%)
        optimal_unit_price = (old_prices.loc[93.5, 100]
                              * ((OPTIMAL_UNIT_PRICE * 100
                                  - OLD_PRICES['Coated Matte - White;100# Text']
                                  .loc[93.5, 100])
                                 / OLD_PRICES['Coated Matte - White;100# Text']
                                 .loc[93.5, 100] + 1.0)) / 100

        # Calculate percent price change for all other quantities/areas
        interp = ((old_prices.iloc[[0, 1, 4, 5]].T
                   - old_prices.loc[93.5, 100])
                  / old_prices.loc[93.5, 100]) + 1.0

        # I found that Coated Gloss - White and Uncoated Smooth - White followed
        # similar quantity/area price adjustment trends as the default (Coated
        # Matte - White), so I used the same correction factors for these stocks
        if press_sheet_type in ['Coated Gloss - White',
                                'Coated Matte - White',
                                'Uncoated Smooth - White']:
            press_sheet_correction = RegularGridInterpolator(
                (QUANTITY_AREA_RATIOS.index, QUANTITY_AREA_RATIOS.columns),
                QUANTITY_AREA_RATIOS.to_numpy(),
                bounds_error=False,
                fill_value=None
            )((quantity, finished_area)).item()
            cbound = 0.1
        else:
            press_sheet_correction = RegularGridInterpolator(
                (interp.index, interp.columns),
                interp.to_numpy(),
                bounds_error=False,
                fill_value=None
            )((quantity, finished_area)).item()
            cbound = 0.08

        # I made the difference between the new price and the current price
        # scale inversely with quantity, so that the old price "buffers" become
        # narrower as the quantity/total price increases
        cbound = min(cbound, cbound / np.log10(quantity))

        price = optimal_unit_price * quantity * press_sheet_correction

        old_price = RegularGridInterpolator(
            (old_prices.columns, old_prices.index),
            old_prices.T.to_numpy(),
            bounds_error=False,
            fill_value=None
        )((quantity, finished_area)).item()

        price = np.clip(
            price,
            old_price * (1.0 - cbound),
            old_price * (1.0 + cbound)
        )

    # Apply correction for front and back ink types
    if side1_ink_type != 'Full Color' or side2_ink_type != 'Full Color':
        ink_correction = INK_ADJ[';'.join([side1_ink_type,
                                           str(side2_ink_type)])]
        ink_correction = RegularGridInterpolator(
            (ink_correction.index, ink_correction.columns),
            ink_correction.to_numpy(),
            bounds_error=False,
            fill_value=None
        )((quantity, finished_area)).item()

        price += ink_correction

    # Estimate each operation's cost individually (using the same calculator
    # logic as Collaterate does) and keep a running total of operations costs
    for operation, operation_item in operations.items():
        if operation_item:
            answer = None  # answer for bundle size in shrink wrapping
            if not isinstance(operation_item, str):
                operation_item, answer = operation_item

            opi_data = OPERATIONS.loc[
                (OPERATIONS['operation_name'] == operation)
                & (OPERATIONS['operation_item_name'] == operation_item),
                ['cost_basis', 'endpoint', 'flatprice',
                 'run_cost', 'setup_cost', 'accumulated_cost']
            ]
            try:
                price += calculate_operation_price(
                    opi_data,
                    num_press_sheets,
                    quantity,
                    answer
                )
            except ValueError:
                print('Invalid operation/operation item: '
                      f'{operation}/{operation_item}')

    return round(price, 2)


class CustomHelpFormatter(argparse.RawTextHelpFormatter):
    """Subclass of argparse.RawTextHelpFormatter to format command line help
    output that doesn't look like it was the brainchild of a four-year-old.
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
        # determine the required width and the entry label
        help_position = min(self._action_max_length + 2,
                            self._max_help_position)
        help_width = max(self._width - help_position, 11)
        action_width = help_position - self._current_indent - 2
        action_header = self._format_action_invocation(action)

        # no help; start on same line and add a final newline
        if not action.help:
            tup = self._current_indent, '', action_header
            action_header = '%*s%s\n' % tup

        # short action name; start on the same line and pad two spaces
        else:  # len(action_header) <= action_width:
            tup = self._current_indent, '', action_width, action_header
            action_header = '%*s%-*s  ' % tup
            indent_first = 0

        # long action name; start on the next line
        # else:
        #     tup = self._current_indent, '', action_header
        #     action_header = '%*s%s\n' % tup
        #     indent_first = help_position

        # collect the pieces of the action help
        parts = [action_header]

        # if there was help for the action, add lines of help text
        if action.help and action.help.strip():
            help_text = self._expand_help(action)
            if help_text:
                help_lines = self._split_lines(help_text, help_width)
                parts.append('%*s%s\n' % (indent_first, '', help_lines[0]))
                for line in help_lines[1:]:
                    parts.append('%*s%s\n' % (help_position, '', line))

        # or add a newline if the description doesn't end with one
        elif not action_header.endswith('\n'):
            parts.append('\n')

        # if there are any sub-actions, add their help as well
        for subaction in self._iter_indented_subactions(action):
            parts.append(self._format_action(subaction))

        # return a single string
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
            if ';' in value:
                value = tuple([int(v.strip())
                               if v.strip().isnumeric()
                               else v.strip()
                               for v in value.split(';')])
            getattr(namespace, self.dest)[key] = value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='python brochure_price_calculator.py',
        formatter_class=CustomHelpFormatter,
        description='calculate Brochure price',
        epilog='As an example:' + '\n'
               + 'python brochure_price_calculator.py --quantity 25 '
               + '--width 8.5 --height 11.0 '
               + '--sheet-type "Coated Matte - White" '
               + '--sheet-weight "100# Text" --front-ink "Full Color" '
               + '--back-ink "Full Color" --operations "Folding"="Single Fold" '
               + '"Scoring-Only"="Parallel Scores" "Shrink Wrap"="Bundles;5"'
    )
    parser.add_argument(
        '-q', '--quantity',
        metavar='\b',
        help='\tquantity ordered (Default: 100)',
        type=int,
        default=100
    )
    parser.add_argument(
        '-w', '--width',
        type=float,
        metavar='\b',
        help='\tfinished width of the Brochure (Default: 8.5)',
        default=8.5
    )
    parser.add_argument(
        '-l', '--height',
        type=float,
        metavar='\b',
        help='\tfinished height of the Brochure (Default: 11.0)',
        default=11.0
    )
    parser.add_argument(
        '-t', '--sheet-type',
        metavar='\b',
        help='\tpaper stock type (Default: "Coated Matte - White")',
        default='Coated Matte - White'
    )
    parser.add_argument(
        '-p', '--sheet-weight',
        metavar='\b',
        help='\tpaper stock weight (Default: "100# Text")',
        default='100# Text'
    )
    parser.add_argument(
        '-f', '--front-ink',
        metavar='\b',
        help='\tfront ink to use (Default: "Full Color")',
        default='Full Color'
    )
    parser.add_argument(
        '-b', '--back-ink',
        metavar='\b',
        help='\tback ink to use (Default: None)',
        default=None
    )
    parser.add_argument(
        '-o', '--operations',
        metavar='\b',
        action=ParseKwargs,
        help='\tadd-on operations and operations items in '
             + '"OPERATION"="OPERATION ITEM" format',
        nargs='*'
    )

    args = parser.parse_args()
    print(calculate_price(
        quantity=args.quantity,
        finished_width=args.width,
        finished_height=args.height,
        press_sheet_type=args.sheet_type,
        press_sheet_weight=args.sheet_weight,
        side1_ink_type=args.front_ink,
        side2_ink_type=args.back_ink,
        **args.operations if args.operations else {}
    ))
