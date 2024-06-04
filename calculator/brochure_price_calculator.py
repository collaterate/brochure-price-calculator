import random
import re
import warnings
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime
from itertools import product
from multiprocessing import Pool

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker
from matplotlib.patches import Rectangle
from rectpack import MaxRectsBaf, newPacker
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.special import softmax
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, URL
from tqdm.auto import tqdm

warnings.filterwarnings('ignore')

SEED = 42
CONN_PARAMS = {'drivername': 'redshift+psycopg2',
               'host': 'redshift-tbg-bi-1.thebernardgroup.io',
               'port': 5439,
               'database': 'tbgbi',
               'username': 'rcorkrean',
               'password': 'Clone0701exclamationpoint'}
QUANTITIES = [1, 10, 25, 50, 100, 200, 500, 1000, 2500, 5000]
ALLOWED_DIMS = [dim for tup in
                [[(width, height), (height, width)]
                 for width, height in [(5.5, 8.5),
                                       (8.5, 11.0),
                                       (9.0, 12.0),
                                       (8.5, 14.0),
                                       (11.0, 17.0),
                                       (11.0, 25.5)]] for dim in tup]
QUANTITY_AREA_RATIOS = pd.DataFrame(
    data=[[1.0, 1.0, 1.0, 1.0],
          [0.94095665, 1.0, 1.05904335, 1.20852018],
          [0.89446367, 1.0, 1.16032295, 1.48096886],
          [0.81620626, 1.0, 1.27624309, 1.84972376],
          [0.72941672, 1.0, 1.44914719, 2.34870499],
          [0.62731481, 1.0, 1.62061404, 2.86184211],
          [0.51778144, 1.0, 1.80322657, 3.40997409],
          [0.4653636, 1.0, 1.89108869, 3.67326608],
          [0.42795306, 1.0, 1.95328847, 3.85994748],
          [0.4143125, 1.0, 1.9761297, 3.92838909]],
    columns=[width * height
             for width, height, in ALLOWED_DIMS[:4:2] + ALLOWED_DIMS[-4::2]],
    index=QUANTITIES
)

rng = np.random.default_rng(SEED)
engine = create_engine(
    URL.create(**CONN_PARAMS),
    connect_args={'sslmode': 'prefer'}
)

def estimate_demand_curve(
    df: pd.DataFrame,
    var: str,
    nbins=10
) -> tuple[float, float]:
    """Calculates the optimal price based on historical sales data and plots the
    underlying estimated demand curve and revenue function.

    :param pd.DataFrame df: The wide DataFrame of sales data.
    :param str var: The variable to group by while constructing the histogram.
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
    W: float,
    H: float,
    bleed: float = 0.125,
    edge: float = 0.125,
    visualize: bool = False
) -> int:
    rectangles = [(w + bleed + edge, h + bleed + edge)] * 100

    algo_results = []

    for orientation in [rectangles, [(r[1], r[0]) for r in rectangles]]:
        packer = newPacker(pack_algo=MaxRectsBaf, rotation=False)
        packer.add_bin(W, H)
        for r in orientation:
            packer.add_rect(*r)
        packer.pack()

        algo_results.append(packer.rect_list())

    best_fit = max(algo_results, key=len)

    if visualize:
        fig, ax = plt.subplots(figsize=(10.0, 6.0))

        for rect in best_fit:
            ax.add_patch(Rectangle(rect[1:3], rect[3], rect[4]))

        ax.set_xlim(0, 29.4375)
        ax.set_ylim(0, 20.75)

        plt.show()

    return len(best_fit)


def get_press_sheet_ratios(engine: Engine) -> pd.DataFrame:
    with engine.begin() as connection:
        return pd.read_sql_query(
            text("""WITH cte AS (SELECT pjcps.default_selected,
                                        pjcps.press_sheet_id,
                                        CAST(ps.width AS FLOAT)
                                            AS unfinished_width,
                                        CAST(ps.height AS FLOAT)
                                            AS unfinished_height,
                                        pst.name AS press_sheet_type_name,
                                        psw.name AS press_sheet_weight_name,
                                        CAST(ct.price AS FLOAT)
                                            AS press_sheet_cost,
                                        ct.updated_on
                                 FROM coll_src.system_offering_site_shares
                                     AS soss
                                 INNER JOIN coll_src.print_job_classifications
                                     AS pjc
                                 ON soss.pricing_print_job_classification_id
                                    = pjc.id
                                 INNER JOIN
                                 coll_src.print_job_classification_press_sheets
                                     AS pjcps
                                 ON pjc.id = pjcps.print_job_classification_id
                                 INNER JOIN coll_src.press_sheets AS ps
                                 ON pjcps.press_sheet_id = ps.id
                                 INNER JOIN coll_src.press_sheet_types AS pst
                                 ON ps.type_id = pst.id
                                 INNER JOIN coll_src.press_sheet_weights AS psw
                                 ON ps.weight_id = psw.id
                                 INNER JOIN coll_src.cost_tables AS ct
                                 ON ps.cost_table_id = ct.id
                                 WHERE soss.id = 10)
                    
                    SELECT cte.updated_on,
                           cte.default_selected,
                           cte.press_sheet_id,
                           cte.unfinished_width,
                           cte.unfinished_height,
                           cte.press_sheet_type_name,
                           cte.press_sheet_weight_name,
                           cte.press_sheet_cost,
                           cte.press_sheet_cost / t.default_press_sheet_cost
                               AS press_sheet_cost_ratio
                    FROM cte
                    CROSS JOIN (SELECT press_sheet_cost
                                    AS default_press_sheet_cost
                                FROM cte
                                WHERE default_selected) AS t;"""),
            con=connection
        ).fillna(np.nan)


def regress_press_sheet_cost(
    engine: Engine, start_date: datetime | str,
    end_date: datetime | str,
    visualize: bool = False
) -> float:
    with engine.begin() as connection:
        dim_df = pd.read_sql_query(
            text(f"""SELECT o.id AS order_id,
                            o.number AS order_number,
                            oi.job_number,
                            CAST(oipjc.finished_width AS FLOAT)
                                AS finished_width,
                            CAST(oipjc.finished_height AS FLOAT)
                                AS finished_height,
                            CAST(oipjc.finished_width AS FLOAT)
                                * CAST(oipjc.finished_height AS FLOAT)
                                AS finished_area,
                            CAST(oipjc.press_sheet_price AS FLOAT)
                                AS press_sheet_cost
                     FROM coll_src.order_items AS oi
                     INNER JOIN coll_src.system_offering_site_shares AS soss
                     ON oi.system_offering_site_share_id = soss.id
                         AND soss.id = 10
                     INNER JOIN coll_src.orders AS o
                     ON oi.order_id = o.id AND o.status_id != 4
                         AND o.created_on BETWEEN
                             '{start_date
                               if isinstance(start_date, str)
                               else start_date.strftime("%Y-%m-%d")}'
                         AND '{end_date
                               if isinstance(end_date, str)
                               else end_date.strftime("%Y-%m-%d")}'
                         AND DATEDIFF(SECOND, o.created_on, oi.created_on) < 2
                     INNER JOIN coll_src.order_item_print_job_classifications
                         AS oipjc
                     ON oi.order_item_print_job_classification_id = oipjc.id
                         AND oipjc.press_sheet_id = 894 AND oipjc.qty = 100
                     WHERE NOT oi.cancelled
                          AND COALESCE(oi.project_id, oi.sales_estimate_id)
                              IS NULL
                          AND oi.order_item_type = 'OI'
                          AND CAST(oi.price AS FLOAT) > 0.0
                          AND (oi.product_id != 'SAMPLES'
                               OR oi.product_id IS NULL);"""),
            con=connection
        ).fillna(np.nan)

    reg = LinearRegression(fit_intercept=False).fit(
                                                    dim_df[['finished_area']],
                                                    dim_df['press_sheet_cost']
                                                )

    if visualize:
        plt.scatter(dim_df['finished_area'], dim_df['press_sheet_cost'])
        plt.plot(
            np.linspace(0.0, dim_df['finished_area'].max(), 2),
            reg.coef_.item()
                * np.linspace(0.0, dim_df['finished_area'].max(), 2)
        )
        plt.xlabel('Finished Area')
        plt.ylabel('Press Sheet Cost')
        score = reg.score(dim_df[['finished_area']], dim_df['press_sheet_cost'])
        plt.annotate(
            f'$R^2={score:.2f}$',
            xy=(200.0, 10.0),
            xytext=(200.0, 10.0)
        )
        plt.show()

    return reg.coef_.item()


def get_ink_ratios(engine: Engine) -> pd.DataFrame:
    with engine.begin() as connection:
        return pd.read_sql_query(
            text("""WITH cte AS (SELECT pjcs1i.default_selected,
                                        pjcs1i.ink_id,
                                        it.name AS ink_type_name,
                                        CAST(ct.price AS FLOAT) AS ink_cost,
                                        ct.updated_on
                                 FROM coll_src.system_offering_site_shares
                                     AS soss
                                 INNER JOIN coll_src.print_job_classifications
                                     AS pjc
                                 ON soss.pricing_print_job_classification_id
                                        = pjc.id
                                 INNER JOIN
                                 coll_src.print_job_classification_side1_inks
                                     AS pjcs1i
                                 ON pjc.id = pjcs1i.print_job_classification_id
                                 INNER JOIN coll_src.inks AS i
                                 ON pjcs1i.ink_id = i.id
                                 INNER JOIN coll_src.ink_types AS it
                                 ON i.type_id = it.id
                                 INNER JOIN coll_src.cost_tables AS ct
                                 ON i.cost_table_id = ct.id
                                 WHERE soss.id = 10)

                    SELECT cte.updated_on,
                           cte.default_selected,
                           cte.ink_id,
                           cte.ink_type_name,
                           cte.ink_cost, cte.ink_cost / t.default_ink_cost
                               AS ink_cost_ratio
                    FROM cte
                    CROSS JOIN (SELECT ink_cost AS default_ink_cost
                                FROM cte
                                WHERE default_selected) AS t;"""),
            con=connection
        ).fillna(np.nan)


def regress_ink_cost(
    engine: Engine,
    start_date: str,
    end_date: str,
    visualize: bool = False
) -> float:
    with engine.begin() as connection:
        dim_df = pd.read_sql_query(
            text(f"""SELECT o.id AS order_id,
                            CAST(oipjc.finished_width AS FLOAT)
                                AS finished_width,
                            CAST(oipjc.finished_height AS FLOAT)
                                AS finished_height,
                            CAST(oipjc.finished_width AS FLOAT)
                                * CAST(oipjc.finished_height AS FLOAT)
                                AS finished_area,
                            CAST(oipjc.side1_ink_price AS FLOAT)
                                AS side1_ink_cost,
                            CAST(oipjc.side2_ink_price AS FLOAT)
                                AS side2_ink_cost
                     FROM coll_src.order_items AS oi
                     INNER JOIN coll_src.system_offering_site_shares AS soss
                     ON oi.system_offering_site_share_id = soss.id
                         AND soss.id = 10
                     INNER JOIN coll_src.orders AS o
                     ON oi.order_id = o.id AND o.status_id != 4
                         AND o.created_on BETWEEN 
                             '{start_date
                               if isinstance(start_date, str)
                               else start_date.strftime("%Y-%m-%d")}'
                         AND '{end_date
                               if isinstance(end_date, str)
                               else end_date.strftime("%Y-%m-%d")}'
                         AND DATEDIFF(SECOND, o.created_on, oi.created_on) < 2
                     INNER JOIN coll_src.order_item_print_job_classifications
                         AS oipjc
                     ON oi.order_item_print_job_classification_id = oipjc.id
                         AND oipjc.side1_ink_id = 146
                         AND oipjc.side2_ink_id = 146
                         AND oipjc.qty = 100
                     WHERE NOT oi.cancelled
                         AND COALESCE(oi.project_id, oi.sales_estimate_id)
                             IS NULL
                         AND oi.order_item_type = 'OI'
                         AND CAST(oi.price AS FLOAT) > 0.0
                         AND (oi.product_id != 'SAMPLES' 
                              OR oi.product_id IS NULL);"""),
            con=connection
        ).fillna(np.nan)

    reg = LinearRegression(fit_intercept=False).fit(
                                                    dim_df[['finished_area']],
                                                    dim_df['side1_ink_cost']
                                                )

    if visualize:
        plt.scatter(dim_df['finished_area'], dim_df['ink_cost'])
        plt.plot(
            np.linspace(0.0, dim_df['finished_area'].max(), 2),
            reg.coef_.item()
            * np.linspace(0.0, dim_df['finished_area'].max(), 2)
        )
        plt.xlabel('Finished Area')
        plt.ylabel('Press Sheet Cost')
        score = reg.score(dim_df[['finished_area']], dim_df['ink_cost'])
        plt.annotate(
            f'$R^2={score:.2f}$',
            xy=(200.0, 10.0), xytext=(200.0, 10.0))
        plt.show()

    return reg.coef_.item() / 2


def get_operations_costs(engine: Engine) -> pd.DataFrame:
    with engine.begin() as connection:
        op_df = pd.read_sql_query(
            text("""SELECT o.id AS operation_id,
                           o.name AS operation_name,
                           CASE o.item_question_modifier_type_id
                               WHEN 1 THEN 'Cost Number of Pieces'
                               WHEN 2 THEN
                               'Cost Number of Pieces after Dividing by Answer'
                               WHEN 3 THEN
                             'Cost Number of Pieces after Multiplying by Answer'
                               WHEN 4 THEN 'Cost Number of Sheets'
                               WHEN 5 THEN
                         'Cost Number of Pieces after Setting Finished Quantity'
                               WHEN 6 THEN 'Cost Area of Sheets'
                               WHEN 7 THEN 'Cost Area of Pieces'
                               WHEN 8 THEN 'Cost Perimeter of Pieces'
                               WHEN 9 THEN 'Cost Add-On'
                               WHEN 10 THEN 'Direct Add-On'
                               WHEN 11 THEN 'Cost Add-On Per Piece'
                           END AS cost_basis,
                           oi.id AS operation_item_id,
                           oi.name AS operation_item_name,
                           COALESCE(CAST(ct.price AS FLOAT), 0.0) AS run_cost,
                           COALESCE(CAST(ctp.price AS FLOAT), 0.0) AS setup_cost
                    FROM coll_src.system_offering_site_shares AS soss
                    INNER JOIN coll_src.print_job_classifications AS pjc
                    ON soss.pricing_print_job_classification_id = pjc.id
                    INNER JOIN coll_src.print_job_classification_operations
                        AS pjco
                    ON pjc.id = pjco.print_job_classification_id AND pjco.show
                    INNER JOIN coll_src.operations AS o
                    ON pjco.operation_id = o.id
                    INNER JOIN coll_src.print_job_classification_operation_items
                        AS pjcoi
                    ON pjco.id = pjcoi.print_job_classification_operation_id
                    INNER JOIN coll_src.operation_items AS oi
                    ON pjcoi.operation_item_id = oi.id
                    INNER JOIN coll_src.cost_tables AS ct
                    ON oi.cost_table_id = ct.id
                    LEFT JOIN coll_src.cost_table_points AS ctp
                    ON ct.id = ctp.cost_table_id AND ctp.endpoint = 1
                    WHERE soss.id = 10
                    ORDER BY o.id, oi.id, ctp.endpoint;"""),
            con=connection
        ).fillna(np.nan)

    return op_df


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
    # Custom Size Surcharge is not specified by the function call;
    # it is added on if the user enters a non-standard Brochure size
    if (finished_width, finished_height) not in ALLOWED_DIMS:
        operations['Custom Size Surcharge'] = 'Brochure Surcharge'
    finished_area = finished_width * finished_height

    try:
        press_sheet_correction = 0.20466173962 * (ps_df.loc[(ps_df['press_sheet_type_name'] == press_sheet_type) & (ps_df['press_sheet_weight_name'] == press_sheet_weight), 'press_sheet_cost_ratio'].item() - 1.0)
    except ValueError as e:
        print(type(e))
        print('Not a valid stock/weight combination')
        return

    # press_sheet_correction = press_sheet_coef * press_sheet_data['press_sheet_cost_ratio'] - 1.0
    #
    # press_sheets = pieces_per_sheet(finished_width, finished_height, press_sheet_data['unfinished_width'], press_sheet_data['unfinished_height'])

    ink_correction = 0.12279704377 * (i_df.loc[i_df['ink_type_name'] == side1_ink_type, 'ink_cost_ratio'].item() + (i_df.loc[i_df['ink_type_name'] == side2_ink_type, 'ink_cost_ratio'].item() if side2_ink_type else 0.0) - 2.0)
    # The rescaling step to adjust for finished area is a little complicated, I wrote out the formula in the markdown cell below
    qa_adjustment = RegularGridInterpolator((QUANTITY_AREA_RATIOS.columns, QUANTITY_AREA_RATIOS.index), QUANTITY_AREA_RATIOS.T.to_numpy(), bounds_error=False, fill_value=None)((finished_area, quantity)).item()
    unit_price = optimal_unit_price * (1.0 + press_sheet_correction + ink_correction) * qa_adjustment

    # For all additional operations, calculate the operation cost and apply a markup
    # Markups are ONLY applied to operations costs
    # Quantity unit-price adjustments are NOT applied to operations costs
    operations_prices = 0.0
    for operation, operation_item in operations.items():
        if operation_item:  # only care about operations with items that are not None
            if not isinstance(operation_item, str):
                operation_item, answer = operation_item  # specifically shrink wrapping asks how many pieces per bundle the customer wants; how this information is passed to the function is something I can coordinate with the Tiger Team
            operation_data = op_df.loc[(op_df['operation_name' if isinstance(
                operation, str) else 'operation_id'] == operation)
                                       & (op_df[
                                              'operation_item_name' if isinstance(
                                                  operation_item,
                                                  str) else 'operation_item_id'] == operation_item), [
                'cost_basis', 'run_cost', 'setup_cost']].squeeze()

            # scale by either the number of press sheets or the number of pieces
            if operation_data['cost_basis'] == 'Cost Number of Sheets':
                operation_cost = press_sheet_coef * finished_area * \
                                 operation_data['run_cost'] + operation_data[
                                     'setup_cost']
            elif operation_data['cost_basis'] == 'Cost Number of Pieces':
                operation_cost = quantity * operation_data['run_cost'] + \
                                 operation_data['setup_cost']
            elif operation_data[
                'cost_basis'] == 'Cost Number of Pieces after Dividing by Answer':
                operation_cost = quantity * operation_data['run_cost'] / answer + operation_data['setup_cost']
            else:
                operation_cost = 0.0

            # apply a markup and add to the running operations price total
            operations_prices += operation_cost * interp1d(
                                                            list(
                                                                QUANTITY_ADJUSTMENTS.keys()),
                                                            [1.71, 1.26, 1.46,
                                                             1.97, 2.7, 3.22,
                                                             3.45, 2.68, 2.35,
                                                             2.04], fill_value='extrapolate')(quantity)

    # return unit price times (in parentheses) times the quantity unit-price adjustment plus the operations prices
    return operations_prices + (unit_price * quantity)  # * interp1d(list(QUANTITY_ADJUSTMENTS.keys()), list(QUANTITY_ADJUSTMENTS.values()))(quantity).item() + operations_prices


rng = np.random.default_rng(SEED)

engine = create_engine(
    URL.create(**CONN_PARAMS),
    connect_args={'sslmode': 'prefer'}
)

optimal_unit_price, lower_ci90, upper_ci90 = 0.4405877130399652, 0.4336458064125138, 0.44754382956751

if any(val is None for val in [optimal_unit_price, lower_ci90, upper_ci90]):
    optimal_unit_price, lower_ci90, upper_ci90 = get_optimal_price(engine)

try:
    ps_df = pd.read_csv('press_sheet_costs.csv')
except FileNotFoundError:
    ps_df = get_press_sheet_ratios(engine)

try:
    i_df = pd.read_csv('../ink_costs.csv')
except FileNotFoundError:
    i_df = get_ink_ratios(engine)

press_sheet_cost_coef, ink_cost_coef, press_sheet_coef = 0.04001075970565253, 0.012371656216164926, 0.289956366216348

if press_sheet_cost_coef is None:
    press_sheet_cost_coef = regress_press_sheet_cost(engine, ps_df.loc[ps_df['default_selected'], 'updated_on'].item(), datetime.today())

if ink_cost_coef is None:
    ink_cost_coef = regress_ink_cost(engine, i_df.loc[i_df['default_selected'], 'updated_on'].item(), datetime.today())

if press_sheet_coef is None:
    press_sheet_coef = regress_press_sheets(engine, '2024-01-01', datetime.today())

try:
    op_df = pd.read_csv('operations_costs.csv')
except FileNotFoundError:
    op_df = get_operations_costs(engine)

quantities = [1, 10, 25, 50, 100, 200, 500, 1000, 2500, 5000]
dims = ALLOWED_DIMS
stocks_weights = ps_df[['press_sheet_type_name', 'press_sheet_weight_name']].to_records(index=False).tolist()
inks = ['Full Color', 'Black Only', None]
operations = defaultdict(list)
for operation, operation_item in op_df.loc[~op_df['operation_name'].isin(['Shrink Wrap', 'Custom Size Surcharge']), ['operation_name', 'operation_item_name']].to_records(index=False).tolist():
    operations[operation].append(operation_item)
for operation in operations:
    operations[operation] = [None] + operations[operation]
operations = [dict(combination) for combination in product(*[[(operation, operation_item) for operation_item in operations[operation]] for operation in operations])]

if __name__ == '__main__':
    print(calculate_price(quantity=100, finished_width=8.5, finished_height=11.0))