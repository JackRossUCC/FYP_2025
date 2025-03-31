#Reservoir Computing Prediction File 
import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg as la
import os
from scipy import stats
from sklearn import linear_model
from scipy import sparse
import sqlite3
import csv
import pandas as pd
import networkx as nx

import Methods as m


class Reservoir: #Creates Reservoir 
    def __init__(self, Nodes, InDim, seed = 12):
        self.Nodes = Nodes
        self.seed = seed 
        self.InDim = InDim
        self.r0 = np.zeros(Nodes)

    def Weighted_Erdos_R(self, p = 0.04, SpecRadius=0.9): #Sets Network by weighted erdos-reyni topology
        Net = nx.fast_gnp_random_graph(self.Nodes, p, seed=self.seed) #Graph ER

        #Number of edges and generating weights
        edges1 = nx.number_of_edges(Net) 
        np.random.seed(self.seed) 
        wts = np.random.uniform(-1, 1, size=edges1)

        #Getting weights as tuples
        edget = tuple(Net.edges) #list of edges 
        wEdges = []
        for i in range(0, edges1):
            wEdges.append(edget[i] + (wts[i],))
        
        #Weighted Network 
        Net.add_weighted_edges_from(wEdges) 
        #As sparse network 
        Net1 = nx.to_scipy_sparse_array(Net, format='csc') #Reservoir Generated
        
        #Spectral Radius (Normalising)
        sr = sparse.linalg.norm(Net1, 2) #could be error here ask about it
        M = (SpecRadius/sr)*Net1 
        self.R = M

    def Set_W_in_Matrix(self):
        W = np.zeros((self.Nodes, self.InDim)) #zeros in shape 

        np.random.seed(self.seed)
        Col = np.random.randint(0, self.InDim, self.Nodes) #Selecting column

        np.random.seed(self.seed)
        Val = np.random.uniform(-1, 1, self.Nodes) #values 

        for i in range(0, self.Nodes):
            W[i, Col[i]] = Val[i]

        self.W_in = sparse.csc_matrix(W)

    def set_rdot(self, sigma, gamma):
        self.sigma = sigma
        self.gamma = gamma 

    def rdot(self, r, u, t): #Reservoir Update Function
          #r reservoir state 
          #u drive

          a = self.R @ r
          b = self.sigma*(self.W_in @ u)

          out = self.gamma*(-r + np.tanh(a+b))
          return out

class RC: 
     def __init__(self, SparseReservoir, U, dt):
         self.R = SparseReservoir
         self.U = U
         self.dt = dt

     def rdot_listening(self, r, t): #listing reservoir update function
         i = int(t/self.dt)
         return self.R.rdot(r, self.U[:, i], t)
         

     def Listening(self, time = 100):
        iListen = int(time/self.dt)

        rListening = np.zeros((self.R.Nodes, iListen)) #Blank Array
        rListening[:, 0] = self.R.r0
        #Iterating Reservoir
        for i in range(1, iListen):
            rListening[:, i] = m.rk4_step(rListening[:, i-1], self.dt*i, self.dt, self.rdot_listening)
        self.rListenFinal = rListening[:, iListen-1]
        self.rListening = rListening

     def Training(self, t0, tEnd, lam = 10**-6):
         iStart = int(t0/self.dt)
         iEnd = int(tEnd/self.dt)
         iTrain = iEnd - iStart

         #Training Data 
         uTrain = self.U[:, iStart:iEnd]
         rTrain = self.rListening[:, iStart:iEnd]

         #Break Symmetry in Data 
         qTrain = np.zeros((2*self.R.Nodes, iTrain))
         qTrain[0:self.R.Nodes, :] = rTrain
         qTrain[self.R.Nodes:2*self.R.Nodes, :] = rTrain*rTrain

         #Q = np.concatenate([np.identity(self.R.Nodes), np.diag(rTrain)])
         #qTrain = Q @ rTrain

         A1 = np.matmul(qTrain, np.transpose(qTrain)) + np.identity(2*self.R.Nodes)*(lam)
         A2 = la.inv(A1)
         A3 = np.matmul(np.transpose(qTrain), A2)
         W = np.matmul(uTrain, A3)
         self.W_out = W 

     def rdot_prediction(self, r, t): #Reservoir Update Function
          #r reservoir state 
          #u drive
            #W = self.R.W_in @ 
            
            q = np.zeros((2*self.R.Nodes))
    
            temp2 = r*r; temp1 = r
            q[0:self.R.Nodes] = temp1; q[self.R.Nodes:2*self.R.Nodes] = temp2

            c = self.W_out @ q 
            a = self.R.R @ r
            b = self.R.sigma*(self.R.W_in @ c)

            out = self.R.gamma*(-r + np.tanh(a+b))
            return out

     def Jac(self, r, t): 
            N = self.R.R.shape[0]

            K = self.R.sigma*(self.R.W_in @ self.W_out)
            K1 = K[:, 0:N]
            #K2 = K[:, N:2*N]

           # A = r
            #for i in range(0, N-1):
            #    A = np.row_stack([A, r])

            #K3 = self.R.sigma*(K2*A)

            J = -(self.R.gamma)*np.identity(N) + self.R.gamma*( self.R.R + K1) #+ K3)
            return J

     def Predicting(self, t0, tFinal, return_r_only=False, return_u_only=False, return_both=False):
        #Indices
        iPredict = int((tFinal-t0)/self.dt) 
        iEnd = int((tFinal)/self.dt)
        iStart = int((t0)/self.dt)

        #Blank Array with correct dimensions 

        uPredict = np.zeros((self.R.InDim, iPredict))
        rPredict = np.zeros((self.R.Nodes, iPredict))
       
       #Setting rIntial 

        #uPredict[:, 0] = np.zeros(uPredict[:, 0].shape)
        rPredict[:, 0] = self.rListenFinal
        #def rdot_predicting(r, t): #predicting reservoir update function
        #   i = int(t/self.dt)
        #    return self.R.rdot(r, uPredict[:, i-1], t)


        #Getting first u value 
        q = np.zeros((2*self.R.Nodes))
        temp2 = rPredict[:, 0]*rPredict[:, 0]
        temp1 = rPredict[:, 0]
        q[0:self.R.Nodes] = temp1; q[self.R.Nodes:2*self.R.Nodes] = temp2
        uPredict[:, 0] = np.matmul(self.W_out, q)

        #Evolving
        for i in range(1, iPredict):
            #Evolving
            rPredict[:, i] = m.rk4_step(rPredict[:, i-1], self.dt*i, self.dt, self.rdot_prediction)
            #Predicting U
            #Q = np.concatenate([np.identity(self.R.Nodes), np.diag(rPredict[:, i])])
            #q = Q @ rPredict[:, i]
            q = np.zeros((2*self.R.Nodes))

            temp2 = rPredict[:, i]*rPredict[:, i]
            temp1 = rPredict[:, i]
            q[0:self.R.Nodes] = temp1; q[self.R.Nodes:2*self.R.Nodes] = temp2
            uPredict[:, i] = np.matmul(self.W_out, q)
           
        #Saving values
        self.rPrediction = rPredict
        self.uPrediction = uPredict
        if return_r_only == True: 
            return  rPredict
        if return_u_only == True: 
            return  uPredict
        if return_both == True: 
            return  uPredict, rPredict


   
def gridSearchRC(sigma_pts, rho_pts, gamma_pts, U, t_evolve = 1000, t_listen = 700, t_train = 500, Nodes = 100, dt = 0.01 ):
    L = []
    Lmean = []
    Lvar = []
    Lauto = []
    Lindex = []

    N = int(len(sigma_pts))
    M = int(len(rho_pts))
    K = len(gamma_pts)

    Nodes = 100
    dt = 0.01

    t_listen = 700
    t_train = 500
    t_evolve = 1000


    q = 1

    for i in range(0, N):
        for j in range(0, M):
            for k in range(0, K):
                #parameters
                sigma = sigma_pts[i] #Drive
                rho = rho_pts[j] #Spectral Radius
                gamma = gamma_pts[k] #gamma 

                #setting reservoir
                r = Reservoir(Nodes = Nodes, InDim=3, seed=12)
                r.Weighted_Erdos_R(SpecRadius=rho, p=0.04)
                r.set_rdot(sigma = sigma, gamma = gamma)
                r.Set_W_in_Matrix() #Error from normalising specral radius

                #Training Reservoir 
                rc1 = RC(SparseReservoir=r, U=U, dt = dt)
                rc1.Listening(time=t_listen)
                rc1.Training(t0 = t_train, tEnd = t_listen)
                rc1.Predicting(t0 = t_listen, tFinal = (t_listen+t_evolve) )

                r1_auto = np.correlate(rc1.uPrediction[0, :], rc1.uPrediction[0, :], mode='full')
                U_auto = np.correlate(U[0, :], U[0, :], mode='full')
                Auto_stat = stats.pearsonr(U_auto, r1_auto).statistic

                L.append(rc1.uPrediction)
                Lindex.append((i, j, k))
                Lmean.append(np.mean(rc1.uPrediction))
                Lvar.append(np.var(rc1.uPrediction))
                Lauto.append(Auto_stat)
                print(q); q = q+1

    Lmean = np.array(Lmean); Lvar = np.array(Lvar); Lauto = np.array(Lauto)      

    U_mean = np.mean(U) - Lmean
    U_var = np.var(U) - Lvar
    i1 = np.argmin(U_mean); i2 = np.argmin(U_var); i3 = np.argmax(Lauto); I = [i1, i2, i3]

    #print(Lindex[i1]); print(Lindex[i2]); print(Lindex[i3])
    return L, I, Lindex
        


#r = Reservoir(Nodes = 100, InDim=3, seed=12)
#r.Weighted_Erdos_R(SpecRadius=0.9, p=0.04)
#r.Set_W_in_Matrix()

#rc = RC(Reservoir=r, U=U, dt = 0.01)
#rc.Listening(time=700)
#rc.Training(t0 = 500, tEnd = 700)
#rc.Predicting(t0 = 700, tFinal = 900)