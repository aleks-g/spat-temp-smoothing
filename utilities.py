# SPDX-FileCopyrightText: 2022 Aleksander Grochowicz
#
# SPDX-License-Identifier: GPL-3.0-or-later

from opcode import hasconst
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
import numpy as np
import statsmodels
import statsmodels.api as sm


# from itertools import product
# import xarray as xr
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import gamma, expon, poisson
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import pacf as PACF
from typing import Collection
import copy


# Functions to estimate the generation data


def seasonal_pattern(data: pd.DataFrame, period: int = 365, plot: bool = True):
    """Finds a seasonality function for `data`.

    Parameters
    ----------
    data: Input data to be deseasonalised [pd.DataFrame]
    period: Period of seasonality (default = 365) [int]
    plot: Include plot of seasonality function (default = True) [bool]

    Returns
    -------
    seasonality: Seasonality
    deseasonalised: Deseasonalised data."""

    def average_value(data, period):
        cap = 0
        reps = len(data) // period
        for i in range(reps):
            cap += data.loc[i * period : (i + 1) * period - 1].values
        cap /= reps
        return pd.DataFrame(cap, columns=data.columns)

    deseasonalised = copy.deepcopy(data)
    seasonality = pd.DataFrame(columns=data.columns)
    avg_val = average_value(data, period)
    # Prepare OLS
    x1 = np.linspace(0, period, period)
    x2 = [np.sin(2 * np.pi * x / period) for x in x1]
    x3 = [np.cos(2 * np.pi * x / period) for x in x1]

    X = np.array([np.array(x2), np.array(x3)])
    X = sm.add_constant(X.T)

    # OLS to estimate the seasonality function
    params = {}
    for i in data.columns:
        model = sm.OLS(avg_val[i], X, hasconst=1)
        res = model.fit()
        params[i] = res.params
    # Create seasonality dataframe
    seas_fct = (
        lambda t, par: par["const"]
        + par["x1"] * np.sin((np.pi * 2 * t) / period)
        + par["x2"] * np.cos((np.pi * 2 * t) / period)
    )
    for i in seasonality.columns:
        seasonality[i] = pd.DataFrame(seas_fct(t, params[i]) for t in x1)

    # Deseasonalise
    reps = len(data) // period
    deseasonalised -= pd.concat(reps * [seasonality], ignore_index=True)
    if plot:
        seasonality.plot()
    return seasonality, deseasonalised


def estimate_partialacf(data: pd.DataFrame):
    """Compute and plot partial ACF. Note that we should use deseasonalised data here."""
    pacf = {}
    for c in data.columns:
        pacf[c] = PACF(data[c])

    fig, ax = plt.subplots()
    pd.DataFrame.from_dict(pacf)[:10].plot(kind="bar", ax=ax)
    for v in [0, 0.2, -0.2]:
        ax.hlines(v, 0, 10, color="grey", ls=":")
    plt.show()


def create_ar_model(data: pd.DataFrame, lag: int):
    """Estimate an AR(`lag`) model based on `data`.

    Parameters
    ----------
    data: Underlying data, should be deseasonalised [pd.DataFrame]
    lag: p in AR(p) model [int]

    Returns
    -------
    ar_params: np.array
    residuals: pd.DataFrame"""

    ar_params = {}
    residuals = {}
    for c in data.columns:
        model = AutoReg(data[c], lags=lag, trend="n").fit()
        ar_params[c] = model.params
        # Plot diagnostics
        model.plot_diagnostics(lags=10)
        print(c, model.aic)
        residuals[c] = model.resid
        # Need to fill up the first `lag` values.
        for i in range(lag):
            residuals[c].loc[i] = 0
    residual = pd.DataFrame.from_dict(residuals).sort_index()
    return ar_params, residual


def estimate_variance(data: pd.DataFrame, period: int, plot: bool = True):
    """Estimate seasonal variance which leads to heteroskedasticity.

    Parameters
    ----------
    data: pd.DataFrame
    period: int
    plot: bool

    Returns
    -------
    seasonal_variance: pd.DataFrame"""
    variability = pd.DataFrame(columns=data.columns)
    reps = len(data) // period
    for i in range(period):
        variability.loc[i] = sum(data.loc[i + period * j] ** 2 for j in range(reps))
    variability /= reps

    variance_seasonal, _ = seasonal_pattern(variability, period, plot=False)
    homosked = variability / variance_seasonal

    if plot:
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        variability.plot(ax=ax[0])
        variance_seasonal.plot(ax=ax[0])
        ax[0].set_title("Seasonal variance")
        homosked.plot(ax=ax[1])
        ax[1].set_title("Deseasonalised")
        homosked.plot.density(ax=ax[2])
        ax[2].set_title("Error distribution")
        plt.show()
    return variance_seasonal, variability


# Optimisation and simulation functions

## Shared functions


def penalty(overprod: pd.DataFrame, underprod: pd.DataFrame):
    penalties = (overprod + underprod) ** 2
    return penalties


def mismatch_values(overprod: pd.DataFrame, underprod: pd.DataFrame):
    mismatches = overprod + underprod
    return mismatches


def compute_prod(x: np.array, capacity_factors: pd.DataFrame):
    prod = [c * x for c in capacity_factors]
    return prod


def initiate_years(batch_size: int, capacity_factors: pd.DataFrame, load: pd.DataFrame):
    """ """
    sample_size = len(capacity_factors) // 365
    pick_years = lambda batch_size: np.random.choice(
        a=sample_size, size=batch_size, replace=False
    )
    batch_years = pick_years(batch_size)
    cap_data = [
        capacity_factors.loc[b * 365 : ((b + 1) * 365) - 1] for b in batch_years
    ]
    load_data = [load.loc[b * 365 : ((b + 1) * 365) - 1] for b in batch_years]
    return cap_data, load_data, batch_years


def compute_transmission(
    production: Collection, load: Collection, transmission_cap: float
):
    overproduction = [(p - l).clip(0) for (p, l) in zip(production, load)]
    underproduction = [(l - p).clip(0) for (p, l) in zip(production, load)]
    poss_exports = [op.clip(upper=transmission_cap) for op in overproduction]
    poss_imports = [up.clip(upper=transmission_cap) for up in underproduction]
    trans = [
        pd.DataFrame(
            np.min([np.max(exp, axis=1), np.max(imp, axis=1)], axis=0),
            index=exp.index,
            columns=["trans"],
        )
        for exp, imp in zip(poss_exports, poss_imports)
    ]
    return trans, overproduction, underproduction


def compute_storage(
    production: Collection,
    load: Collection,
    storage_cap: np.array,
    charge_cap: np.array,
    discharge_cap: np.array,
    initial_storage_level: np.array = np.array([0, 0]),
    eff_c: float = 0.75,
    eff_d: float = 0.9,
):
    overproduction = [(p - l).clip(0) for (p, l) in zip(production, load)]
    underproduction = [(l - p).clip(0) for (p, l) in zip(production, load)]
    poss_charge = [op.clip(upper=charge_cap, axis=1) for op in overproduction]
    poss_discharge = [up.clip(upper=discharge_cap, axis=1) for up in underproduction]
    usages = []
    levels = []
    for ch, dc in zip(poss_charge, poss_discharge):
        lv, us = determine_storage_year(
            poss_charge=ch,
            poss_discharge=dc,
            storage_cap=storage_cap,
            initial_storage_level=initial_storage_level,
            eff_c=eff_c,
            eff_d=eff_d,
        )
        usages.append(us)
        levels.append(lv)
    return usages, levels, overproduction, underproduction


def determine_storage_year(
    poss_charge: Collection,
    poss_discharge: Collection,
    storage_cap: np.array,
    initial_storage_level: np.array,
    eff_c: float = 0.75,
    eff_d: float = 0.9,
):
    storage_level = [initial_storage_level]
    storage_usage = []
    for i in range(len(poss_charge)):
        level = storage_level[-1]
        usage = np.minimum(
            np.array(storage_cap - level), poss_charge.iloc[i].to_numpy()
        ) - np.minimum(np.array(level), poss_discharge.iloc[i].to_numpy())
        storage_usage.append(usage)
        storage_level.append(
            level + eff_c * np.maximum(0, usage) + eff_d * np.minimum(0, usage)
        )
    stor_level = pd.DataFrame(
        storage_level[1:], index=poss_charge.index, columns=poss_charge.columns
    )
    stor_usage = pd.DataFrame(
        storage_usage, index=poss_charge.index, columns=poss_charge.columns
    )
    return stor_level, stor_usage


def compute_storage_transmission(
    charge_potential,
    discharge_need,
    exports,
    imports,
    storage_cap,
    charge_cap,
    discharge_cap,
    transmission_cap,
    initial_storage_level,
    eff_c: float = 0.75,
    eff_d: float = 0.9,
):
    usages = []
    levels = []
    imported = []
    exported = []
    for ch, dc, ex, im in zip(charge_potential, discharge_need, exports, imports):
        us, lv, exp, imp = determine_storage_and_trans_year(
            charge_potential=ch,
            discharge_need=dc,
            exports=ex,
            imports=im,
            storage_cap=storage_cap,
            charge_cap=charge_cap,
            discharge_cap=discharge_cap,
            transmission_cap=transmission_cap,
            initial_storage_level=initial_storage_level,
            eff_c=eff_c,
            eff_d=eff_d,
        )
        usages.append(us)
        levels.append(lv)
        exported.append(exp)
        imported.append(imp)
    return usages, levels, exported, imported


def determine_storage_and_trans_year(
    charge_potential,
    discharge_need,
    exports,
    imports,
    storage_cap,
    charge_cap,
    discharge_cap,
    transmission_cap,
    initial_storage_level,
    eff_c: float = 0.75,
    eff_d: float = 0.9,
):
    storage_level = [initial_storage_level]
    storage_usage = []
    trans = exports.sum(
        axis=1
    )  # either imports or exports is fine, can only go way, so can sum it up; transmission_cap is a float, so this needs to be a series
    for i in range(len(charge_potential)):
        usage = np.array([0.0, 0.0])
        level = storage_level[-1]
        charge, discharge = 0, 0
        # if transmission was limited by exports, i.e. no more overproduction
        if all(charge_potential.iloc[i] == 0):
            discharge = np.min(
                [discharge_need.iloc[i].to_numpy(), level, discharge_cap], axis=0
            )
            discharge_need.iloc[i] -= discharge
            usage -= discharge
            # have discharged locally as much as possible
            if any(discharge_need.iloc[i] > 0) and not all(discharge_need.iloc[i] > 0):
                # still underproduction in one location, if we have it in both, we will have a mismatch, as above will have exhausted the possibilities
                tapping_other_battery = np.minimum(
                    np.min(
                        [
                            discharge_cap - discharge,
                            level,
                            discharge_need[discharge_need.columns[::-1]]
                            .iloc[i]
                            .to_numpy(),
                        ],
                        axis=0,
                    ),
                    transmission_cap - trans.iloc[i],
                )
                # need to flip discharge_need to fill up the other battery and now need to flip tapping other battery to make the usage correct:
                # we want tapping_other_battery to be positive, if the other battery had additional discharge_cap, transmission_cap and we have positive discharge_need
                usage -= tapping_other_battery
                # since we have discharged a battery we get eff_d * discharge
                imports.iloc[i] += eff_d * np.flip(
                    tapping_other_battery
                )  # need to flip the values from the columns
                exports.iloc[i] += eff_d * tapping_other_battery

        elif all(discharge_need.iloc[i] == 0):
            # transmission limited by imports, i.e. no more underproduction
            charge = np.min(
                [
                    charge_potential.iloc[i].to_numpy(),
                    storage_cap - level,
                    charge_cap,
                ],
                axis=0,
            )
            usage += charge
            # have charged locally as much as possible
            if any(charge_potential.iloc[i] > 0) and not all(
                charge_potential.iloc[i] > 0
            ):
                # still some overproduction in one location
                charging_other_battery = np.minimum(
                    np.min(
                        [
                            charge_cap - charge,
                            storage_cap - level,
                            charge_potential[charge_potential.columns[::-1]]
                            .iloc[i]
                            .to_numpy(),
                        ],
                        axis=0,
                    ),
                    transmission_cap - trans.iloc[i],
                )
                # assume country 1 has charging and storage capacities left, there is still transmission available and country 2 still has charging_potential (that it locally exhausted); then we can charge storage 1 (without flipping)
                usage += charging_other_battery
                # since we are charging we do not use efficiencies in the electricity we put in
                imports.iloc[i] += charging_other_battery
                exports.iloc[i] += np.flip(charging_other_battery)
        storage_level.append(
            level + eff_d * np.minimum(0, usage) + eff_c * np.maximum(0, usage)
        )
        storage_usage.append(usage)
        # add additional exports
        # add additional imports
    stor_usage = pd.DataFrame(
        storage_usage,
        index=charge_potential.index,
        columns=charge_potential.columns,
    )
    stor_level = pd.DataFrame(
        storage_level[1:],
        index=charge_potential.index,
        columns=charge_potential.columns,
    )
    return stor_usage, stor_level, exports, imports


def sim_no_flex(
    x: np.array,
    batch_size: int,
    cap_factors_data: pd.DataFrame,
    demand_data: pd.DataFrame,
):
    """Simulates a two-node system, without transmission and storage.
    The penalty function is sum_t abs(demand(t) - prod(t))**2.

    Parameters
    ----------
    x: Capacities
    batch_size: Number of years to simulate
    cap_factors_data: Capacity factors with numeric index
    demand_data: Demand data with numeric index

    Returns
    -------
    penalties: Penalty values
    mismatches: Mismatch between demand and production
    """
    if len(cap_factors_data) != len(demand_data):
        raise ValueError("Demand and capacity factor time series do not match!")

    cap_data, load_data, batch_years = initiate_years(
        batch_size, cap_factors_data, demand_data
    )
    prod = compute_prod(x, cap_data)

    # Calculate penalty values and save iteration data
    overproduction = [(p - l).clip(0) for (p, l) in zip(prod, load_data)]
    underproduction = [(l - p).clip(0) for (p, l) in zip(prod, load_data)]

    # Initiate dataframes for results
    penalties = {loc: pd.DataFrame(index=range(365)) for loc in ["NO-N", "NO-S"]}
    mismatches = {loc: pd.DataFrame(index=range(365)) for loc in ["NO-N", "NO-S"]}
    for i, year in enumerate(batch_years):
        penalty_res = penalty(overproduction[i], underproduction[i]).reset_index(
            drop=True
        )
        mismatch_res = mismatch_values(
            overproduction[i], underproduction[i]
        ).reset_index(drop=True)
        for loc in ["NO-N", "NO-S"]:
            penalties[loc][f"year {i}"] = penalty_res[loc]
            mismatches[loc][f"year {i}"] = mismatch_res[loc]
    return penalties, mismatches


def sim_trans(
    x: np.array,
    transmission_cap: float,
    batch_size: int,
    cap_factors_data: pd.DataFrame,
    demand_data: pd.DataFrame,
):
    """Simulates a two-node system with transmission between them.
    The penalty function is sum_t abs([demand(t) + exports(t) - [prod(t) + imports(t)]).

    Parameters
    ----------
    x: Capacities
    transmission_cap: Capacity of transmission
    batch_size: Number of years to simulate
    cap_factors_data: Capacity factors
    demand_data: Demand

    Returns
    -------
    penalties: Penalty values
    mismatches: Mismatch between demand and production
    imports: Imports
    exports: Exports
    """
    if len(cap_factors_data) != len(demand_data):
        raise ValueError("Demand and capacity factor time series do not match!")

    cap_data, load_data, batch_years = initiate_years(
        batch_size, cap_factors_data, demand_data
    )
    prod = compute_prod(x, cap_data)

    # Initiate dataframes
    penalties = {loc: pd.DataFrame(index=range(365)) for loc in ["NO-N", "NO-S"]}
    mismatches = {loc: pd.DataFrame(index=range(365)) for loc in ["NO-N", "NO-S"]}
    imports = {loc: pd.DataFrame(index=range(365)) for loc in ["NO-N", "NO-S"]}
    exports = {loc: pd.DataFrame(index=range(365)) for loc in ["NO-N", "NO-S"]}

    # Compute transmission and populate dataframes
    trans, overproduction, underproduction = compute_transmission(
        prod, load_data, transmission_cap
    )
    for i, year in enumerate(batch_years):
        overprod = overproduction[i].subtract(trans[i]["trans"], axis=0).clip(0)
        underprod = underproduction[i].subtract(trans[i]["trans"], axis=0).clip(0)
        exports_year = (overproduction[i] - overprod).reset_index(drop=True)
        imports_year = (underproduction[i] - underprod).reset_index(drop=True)
        penalty_res = penalty(overprod, underprod).reset_index(drop=True)
        mismatch_res = mismatch_values(overprod, underprod).reset_index(drop=True)
        for loc in ["NO-N", "NO-S"]:
            penalties[loc][f"year {i}"] = penalty_res[loc]
            mismatches[loc][f"year {i}"] = mismatch_res[loc]
            imports[loc][f"year {i}"] = imports_year[loc]
            exports[loc][f"year {i}"] = exports_year[loc]
    return penalties, mismatches, imports, exports


# Need empty storage at the beginning.
def sim_storage(
    x: np.array,
    storage_cap: np.array,
    charge_cap: np.array,
    discharge_cap: np.array,
    batch_size: int,
    cap_factors_data: pd.DataFrame,
    demand_data: pd.DataFrame,
    eff_c: float = 0.75,
    eff_d: float = 0.9,
):
    """Simulates a two-node system with storage in each.
    The penalty function is sum_t abs([demand(t) + charging(t) - [prod(t) + discharging(t)])**2.

    Parameters
    ----------
    x: Capacities
    storage_cap: Capacity of storage
    charge_cap: Charging capacity
    discharge_cap: Discharging capacity
    batch_size: Number of years to simulate
    cap_factors_data: Capacity factors
    demand_data: Demand
    eff_c: Efficiency of charging
    eff_d: Efficiency of discharging

    Returns
    -------
    penalties: Penalty values
    mismatches: Mismatch values
    storage_usage: Storage usage
    storage_level: Storage levels
    """
    initial_storage_level = np.array([0, 0])

    for i in range(len(storage_cap)):
        if np.any(charge_cap[i] > storage_cap[i]):
            charge_cap[i] = storage_cap[i]
            print(
                "Charging capacity cannot be higher than storage capacity. Set them equal."
            )
        if np.any(discharge_cap[i] > storage_cap[i]):
            discharge_cap[i] = storage_cap[i]
            print(
                "Discharging capacity cannot be higher than storage capacity. Set them equal."
            )

    if len(cap_factors_data) != len(demand_data):
        raise ValueError("Demand and capacity factor time series do not match!")

    cap_data, load_data, batch_years = initiate_years(
        batch_size, cap_factors_data, demand_data
    )
    prod = compute_prod(x, cap_data)

    # Initiate dataframes
    penalties = {loc: pd.DataFrame(index=range(365)) for loc in ["NO-N", "NO-S"]}
    mismatches = {loc: pd.DataFrame(index=range(365)) for loc in ["NO-N", "NO-S"]}
    storage_usages = {loc: pd.DataFrame(index=range(365)) for loc in ["NO-N", "NO-S"]}
    storage_levels = {loc: pd.DataFrame(index=range(365)) for loc in ["NO-N", "NO-S"]}

    usages, levels, overproduction, underproduction = compute_storage(
        prod,
        load_data,
        storage_cap,
        charge_cap,
        discharge_cap,
        initial_storage_level,
        eff_c,
    )
    # Populate dataframes
    for i, year in enumerate(batch_years):
        overprod = overproduction[i].subtract(eff_d * usages[i].clip(0))
        underprod = underproduction[i].add(usages[i].clip(upper=0))
        penalty_res = penalty(overprod, underprod).reset_index(drop=True)
        mismatch_res = mismatch_values(overprod, underprod).reset_index(drop=True)
        for loc in ["NO-N", "NO-S"]:
            penalties[loc][f"year {i}"] = penalty_res[loc]
            mismatches[loc][f"year {i}"] = mismatch_res[loc]
            storage_usages[loc][f"year {i}"] = usages[i][loc].reset_index(drop=True)
            storage_levels[loc][f"year {i}"] = levels[i][loc].reset_index(drop=True)
    return penalties, mismatches, storage_usages, storage_levels


def sim_full_flex(
    x: np.array,
    transmission_cap: float,
    storage_cap: np.array,
    charge_cap: np.array,
    discharge_cap: np.array,
    batch_size: int,
    cap_factors_data: pd.DataFrame,
    demand_data: pd.DataFrame,
    eff_c: float = 0.75,
    eff_d: float = 0.9,
):
    """Simulates a two-node system with storage in each node and transmission between them.
    The penalty function is sum_t abs([demand(t) + charging(t) + exports(t) - [prod(t) + discharging(t) + imports(t)])**2.

    Parameters
    ----------
    x: Capacities
    transmission_cap: Capacity of transmission
    storage_cap: Capacity of storage
    charge_cap: Charging capacity
    discharge_cap: Discharging capacity
    batch_size: Number of years to simulate
    cap_factors_data: Capacity factors
    demand_data: Demand
    eff_c: Efficiency of charging
    eff_d: Efficiency of discharging

    Returns
    -------
    penalties: Penalty values
    mismatches: Mismatch values
    imports: Imports
    exports: Exports
    storage_usage: Storage usage
    storage_level: Storage levels"""

    cap_data, load_data, batch_years = initiate_years(
        batch_size, cap_factors_data, demand_data
    )
    prod = compute_prod(x, cap_data)
    penalties = {loc: pd.DataFrame(index=range(365)) for loc in ["NO-N", "NO-S"]}
    mismatches = {loc: pd.DataFrame(index=range(365)) for loc in ["NO-N", "NO-S"]}
    imports = {loc: pd.DataFrame(index=range(365)) for loc in ["NO-N", "NO-S"]}
    exports = {loc: pd.DataFrame(index=range(365)) for loc in ["NO-N", "NO-S"]}
    storage_usages = {loc: pd.DataFrame(index=range(365)) for loc in ["NO-N", "NO-S"]}
    storage_levels = {loc: pd.DataFrame(index=range(365)) for loc in ["NO-N", "NO-S"]}

    trans, overproduction, underproduction = compute_transmission(
        prod, load_data, transmission_cap
    )
    charge_potential = [
        op.subtract(tr["trans"], axis=0).clip(0)
        for op, tr in zip(overproduction, trans)
    ]
    discharge_need = [
        up.subtract(tr["trans"], axis=0).clip(0)
        for up, tr in zip(underproduction, trans)
    ]
    exported = [old_op - op for old_op, op in zip(overproduction, charge_potential)]
    imported = [old_up - up for old_up, up in zip(underproduction, discharge_need)]
    st_usage, st_levels, exported, imported = compute_storage_transmission(
        charge_potential,
        discharge_need,
        exported,
        imported,
        storage_cap,
        charge_cap,
        discharge_cap,
        transmission_cap,
        initial_storage_level=np.array([0, 0]),
    )
    for i, year in enumerate(batch_years):
        generation = prod[i].subtract(eff_d * st_usage[i].clip(upper=0) - imported[i])
        consumption = load_data[i].add(st_usage[i].clip(0) + exported[i])
        overprod = generation.subtract(consumption).clip(0)
        underprod = consumption.subtract(generation).clip(0)
        penalty_res = penalty(overprod, underprod).reset_index(drop=True)
        mismatch_res = mismatch_values(overprod, underprod).reset_index(drop=True)
        for loc in ["NO-N", "NO-S"]:
            penalties[loc][f"year {i}"] = penalty_res[loc]
            mismatches[loc][f"year {i}"] = mismatch_res[loc]
            imports[loc][f"year {i}"] = imported[i][loc].reset_index(drop=True)
            exports[loc][f"year {i}"] = exported[i][loc].reset_index(drop=True)
            storage_usages[loc][f"year {i}"] = st_usage[i][loc].reset_index(drop=True)
            storage_levels[loc][f"year {i}"] = st_levels[i][loc].reset_index(drop=True)
    return penalties, mismatches, imports, exports, storage_usages, storage_levels


def plot_iterations(xs, ys):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    nb_iters = len(ys)
    for i in range(1, nb_iters):
        ax[0].plot([xs[i - 1], xs[i]], [ys[i - 1], ys[i]], "ro-")
        ax[1].plot(
            [xs[i - 1].sum(), xs[i].sum()], [ys[i - 1].sum(), ys[i].sum()], "ko-"
        )
    fig.show()
