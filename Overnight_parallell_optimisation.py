import numpy as np
import nolds
from matplotlib import pyplot as plt
from numpy import linalg as la
import os
from scipy import stats as st
from scipy import spatial as sp
from scipy import sparse
from sklearn import linear_model
from scipy import linalg as sla

import lyapynov as le

import pandas as pd
import networkx as nx

import SystemsSolved as s
import Methods as m

import Sparse_Reservoir_Computing as rc

import RecurrencePlotsCode as rp

import multiprocess as mp

import Parallel_Grid_Search as pgs
import openpyxl as op 

if __name__ == '__main__':
    mu_pts = np.linspace(0.0, 0.1, 6); nu = 0.02
    sigma_pts = np.linspace(0.01, 0.95, 10); rho_pts = np.linspace(0.01, 0.95, 10); gamma_pts = [1]
    params = pgs.parameters_reshape_gridsearch(sigma_pts, rho_pts, gamma_pts)
    file = "C:\\Users\\jackr\\JackRoss_FYP_2024\\Python_Files\\OutputData\\ReservoirParameters.xlsx"

    k = 1
    for i in range(1, len(mu_pts)):
        Ry = s.yCoupledRossler(a = 0.2, b = 0.2, c = 5.7, 
                            nu1 = nu, mu1 = mu_pts[i], nu2 = nu, mu2 = mu_pts[i])
        Ry_ts = Ry.solve(t0 = 0, tFinal = 1000, x0_1 = [-1.5, 1.5, 20], x0_2 = [-1, -1, 10], dt = 0.01)

        #Splits ts and removes transient time 
        A0 = Ry_ts[0:3, :]; B0 = Ry_ts[3:6, :] 
        A = pgs.gridSearchRC_parallell(params, A0, processes = 6, t_evolve = 1000, t_listen = 700, t_train = 500, Nodes = 100, dt = 0.01)
        B = pgs.gridSearchRC_parallell(params, B0, processes = 6, t_evolve = 1000, t_listen = 700, t_train = 500, Nodes = 100, dt = 0.01)
        output = [A + B + [mu_pts[i].tolist()]]

    
        wb = op.load_workbook(file); ws = wb.active
        for row in output:
            ws.append(row)
        wb.save(file)
        print(k)
        k = k+1