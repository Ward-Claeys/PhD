#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 09:24:15 2026

@author: wardclaeys
"""

# Import modules
import os, sys
import numpy as np
import pandas as pd
from scipy import optimize

# Avoid warnings
import warnings

warnings.filterwarnings("ignore")

os.chdir("/Users/wardclaeys/Documents/Github/PhD/Parameter_recovery")


def softmax(
    values=np.array([0.5, 0.5, 0.5]),
    PEs=np.array([0.5, 0.5, 0.5]),
    LPs=np.array([0.5, 0.5, 0.5]),
    Novs=np.array([0.5, 0.5, 0.5]),
):

    numerator_1 = np.exp(values[0] * PEs[0] + values[1] * LPs[0] + values[2] * Novs[0])
    numerator_2 = np.exp(values[0] * PEs[1] + values[1] * LPs[1] + values[2] * Novs[1])
    numerator_3 = np.exp(values[0] * PEs[2] + values[1] * LPs[2] + values[2] * Novs[2])

    denom = np.sum([numerator_1, numerator_2, numerator_3])

    response_probabilities = [
        numerator_1 / denom,
        numerator_2 / denom,
        numerator_3 / denom,
    ]

    # response_probabilities = np.exp(values[0][0] * PEs + values[0][1] * LPs + values[0][2] * Novs) / np.sum(np.exp(values[0][0] * PEs + values[0][1] * LPs + values[0][2] * Novs))

    return response_probabilities


def delta_rule(previous_value=0.0, obtained_reward=1.0, LR=0.1):

    # Calculate the prediction error:
    PE = obtained_reward - previous_value  # PE = R(t-1) - V(s, a)(t-1)
    # calculate the new value for this stimulus-response pair
    updated_value = np.sum(
        [previous_value, np.multiply(PE, LR)]
    )  # V(s, a)t = V(s, a)(t-1) + PE*LR
    return PE, updated_value


# Likelihood function for empirical data
def likelihood(parameter_set, file="Check_this_file.csv"):

    df = pd.read_csv(file)  # Read data
    ntrials = df.shape[0]  # Extract number of trials

    # Start the likelihood estimation process: summed_logL = log(L(parameter set|data))
    # log(L(parameter set|data)) = sum( log( L(parameter set|response) ) for trial in trials)
    summed_logL = 0  # this is calculated by summing over trials the log( L(parameter set|response on that trial) )

    # trial-loop: calculate log(L(parameter set|response)) on each trial
    for trial in range(ntrials):
        # Define the variables that are important for the likelihood estimation process
        response = int(df.loc[trial, "Choice"]) - 1  # the stimulus shown on this trial

        PEs = np.array(
            [df.loc[trial, "PE_1"], df.loc[trial, "PE_2"], df.loc[trial, "PE_3"]]
        )
        LPs = np.array(
            [df.loc[trial, "LP_1"], df.loc[trial, "LP_2"], df.loc[trial, "LP_3"]]
        )
        Novs = np.array(
            [df.loc[trial, "Nov_1"], df.loc[trial, "Nov_2"], df.loc[trial, "Nov_3"]]
        )

        loglikelihoods = np.log(softmax(parameter_set, PEs, LPs, Novs))

        # then select the probability of the actual response given the parameter set
        current_loglikelihood = loglikelihoods[response]

        # Add L(parameter set|current response) to the total log likelihood
        summed_logL = summed_logL + current_loglikelihood

    return -summed_logL


if __name__ == "__main__":
    # Extract path to data folder

    directory = "/Users/wardclaeys/Documents/Github/PhD/Parameter_recovery"

    # Get a list of files in the folder and filter on csv files that are not previous fitting results
    filelist = os.listdir(directory)
    filtered_filelist = [x for x in filelist if x[-3::] == "csv"]
    filtered_filelist = [x for x in filtered_filelist if x != "Fitting_results.csv"]
    filtered_filelist = [x for x in filtered_filelist if x[-13::] != "Simulated.csv"]

    # Go to that directory
    os.chdir(directory)

    # Define the starting parameters for the likelihood
    start_params = (
        np.random.uniform(0, 1),
        np.random.uniform(0, 1),
        np.random.uniform(0, 1),
    )

    # Define columns for output file
    column_list = [
        "Current_PE",
        "Current_LP",
        "Current_Nov",
        "Estimated_PE",
        "Estimated_LP",
        "Estimated_Nov",
        "Negative_LogL",
    ]
    estimated_data = pd.DataFrame(columns=column_list)

    idx = -1
    for file in filtered_filelist:
        idx += 1

        # Minimize negative loglikelihood
        # df = pd.read_csv("Check_this_file.csv")
        # optimization_output = optimize.minimize(fun = likelihood , x0 = [start_params, df]) #, options = {'maxfev':10000, 'xatol':0.00001, 'return_all':0})

        optimization_output = optimize.minimize(
            fun=likelihood,
            x0=start_params,
            args=file,
            options={"maxfev": 10000, "xatol": 0.00001, "return_all": 0},
        )

        # Get minimum log likelihood and parameter estimations
        LL = optimization_output["fun"]
        estimated_parameters = optimization_output["x"]

        # print("estimated learning rate is: {0} and estimated inverse temperature is: {1}.\n\n".format(lr, inv_temp))
        # Store everything in output file
        estimated_data.loc[idx, ["Current_PE", "Current_LP", "Current_Nov"]] = [
            df.loc[0, "weight_PE"],
            df.loc[0, "weight_LP"],
            df.loc[0, "weight_Nov"],
        ]
        estimated_data.loc[
            idx, ["Estimated_PE", "Estimated_LP", "Estimated_Nov", "Negative_LogL"]
        ] = [
            estimated_parameters[0],
            estimated_parameters[1],
            estimated_parameters[2],
            LL,
        ]

        print("Simulated data")

    # Write results of parameter fitting
    estimated_data.to_csv(
        "Fitting_results.csv", columns=column_list, float_format="%.3f"
    )
    print("End of fitting procedure")
