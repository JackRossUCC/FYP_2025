#Recurrance Plots
import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg as la
from scipy import stats
from scipy import spatial as sp
import pandas as pd


#Recurrence Matrix
def RecurrenceMatrix(Data, p = 2, tol=0.1):
    Dist_M = sp.distance_matrix(Data.T, Data.T, p = p)
    Bool_M = (Dist_M < tol) #true/false matrix
    return Bool_M

def CrossRecurrrenceMatrix(Data1, Data2, p = 2, tol=0.1):
    Dist_M = sp.distance_matrix(Data1.T, Data2.T, p = p)
    Bool_M = (Dist_M < tol) #true/false matrix
    return Bool_M

def JoinRecurrenceMatrix(Matrix1, Matrix2): #Takes two recurrence matrices as input
    return np.multiply(Matrix1, Matrix2)


#For Plotting of Recurrence Matrices
class RP: 
    xlabel = "Time"
    ylabel = "Time"
    title = "Recurrence Plot"
    def figure(size = (6, 6)):
        RP.fig = plt.figure(figsize=(6,6))


    def plot(Data): #SIngle plot
        plt.title(RP.title)
        plt.xlabel(RP.xlabel)
        plt.ylabel(RP.xlabel)
        plt.imshow(Data, cmap='binary', origin='lower')
    
    def set_Lim(x, y): #Setting limit for single plot
        plt.xlim(x)
        plt.ylim(y)

    def subfig(size = (8, 6), nrows = 1, ncols = 2): #setting subplots
        RP.fig, RP.ax = plt.subplots(nrows, ncols, figsize=size)
    
    def subplot(Data, plotNum, title=title, xlab=xlabel, ylab=ylabel): #subplotting
        RP.ax[plotNum].imshow(Data, cmap='binary', origin='lower')
        RP.ax[plotNum].set_title(title)
        RP.ax[plotNum].set_xlabel(xlab)
        RP.ax[plotNum].set_ylabel(ylab)
    
    def sub_set_lim(x,y, plotNum):
        RP.ax[plotNum].set_xlim(x)
        RP.ax[plotNum].set_ylim(y)

class recurrence_functions: #Calculate probability of recurrence from recurrence matrix. 
    def __init__(self, RM): #Takes Recurrence Matrix
        self.RM = RM
        self.N = len(RM[:, 0])

    def p_tau(self, tau): #Probability of recurrence after tau 
        L = int(self.N - tau)
        w1 = np.diagonal(self.RM, offset=tau) #diagonal row 
        w2 = sum(w1) 
        L = len(w1)
        out = w2/L
        
        return out
    
    def p_tau_dist(self): #distribution prob of recurrence for each two (outputs vector)
        out = np.zeros(self.N)
        for i in range(0, self.N): #Calculates for all tau
           out[i] = self.p_tau(i)
        
        return out

    def recurrence_rate(self):
        total = 0 
        for i in range(0, self.N):
            total = total + sum(self.RM[i, :])
        
        out = (1/((self.N)**2))*(total)
        return out
    

            