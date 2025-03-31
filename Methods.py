#Methods
import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg as la
from scipy import stats
from scipy import spatial as sp
from scipy import interpolate as ip
import pandas as pd
import RecurrencePlotsCode as rp


#Rk4 Solving 

#Step
def rk4_step(x0, t0, dt, f):
        f1 = f(x0, t0)
        f2 = f(x0 + (dt/2)*f1, t0 + (dt/2))
        f3 = f(x0 + (dt/2)*f2, t0 + (dt/2))
        f4 = f(x0 + dt*f3, t0 + dt)
        xk_1 = x0 + (dt/6)*(f1 + 2*f2 + 2*f3 + f4)
        return xk_1

#Solve integration 
def solve(xdot, x0, t0, tFinal, dt, Dim = 3):
    N = int((tFinal-t0)/dt)
    sol = np.zeros((Dim, N))

    sol[:, 0] = x0
    for i in range(0, N-1):
        sol[:, i+1] = rk4_step(x0 = sol[:, i], t0=dt*i, dt=dt, f = xdot) 
        print(sol[0, i])
        
        return np.array(sol)
    

#Euler Maryuma Method 

def em_Iterate(f, g, x0, t0, dt, dW): #f deterministic, g stochastic
        return x0 + f(x0, t0)*dt + g(x0, t0)*dW


def em_Solve(f, g, x0, t0, tFinal, dt=0.01):
       numSteps = int(tFinal/dt)
       xOut = np.zeros(numSteps)
       xOut[0] = x0
    
       for i in range(0, numSteps-1):
           t_i = t0 + dt*i
           dW = np.random.normal(0, np.sqrt(dt))
           xOut[i+1] = em_Iterate(f, g, xOut[i], t_i, dt, dW)
       return xOut


#EM method for solving stochatic differential equations

def em2D_Iterate(f1, f2, g1, g2, x10, x20, t0, dt, dW1, dW2):
       x1_new = x10 + f1(x10, x20, t0)*dt + g1(x10, x20, t0)*dW1
       x2_new = x20 + f2(x10, x20, t0)*dt + g2(x10, x20, t0)*dW2
       

       return np.array([x1_new, x2_new])
    
def em2D_Solve(f1, f2, g1, g2, x10, x20, t0, tFinal, dt=0.01): #f1, f2 deterministic, g1, g2 stochastic
       numSteps = int((tFinal-t0)/dt)
       xOut = np.zeros((numSteps, 2))
       xOut[0, 0] = x10
       xOut[0, 1] = x20
    
       for i in range(0, numSteps-1):
           t_i = t0 + dt*i
           dW1 = np.random.normal(0, np.sqrt(dt))
           dW2 = np.random.normal(0, np.sqrt(dt))
           xOut[i+1, :] = em2D_Iterate(f1, f2, g1, g2, xOut[i, 0], xOut[i, 1], t_i, dt, dW1, dW2)
    
       return xOut

#Delay Co-ordinate Embedding for univarate data
def DCembedding(ts, m, tau):
    n = len(ts) 
    X = np.zeros((m, n-tau*m))
    for i in range(0, n - tau*m): 
        coord = ts[i:m+i]
        X[:, i] = coord
    return X

#Plotting 
class plot3D: #Must plot in same block 
    def __init__(self, cols=1, rows=1, figsize = (10, 10)):
        self.figRow = rows #Rows
        self.figCol = cols #Cols
        self.figSize = figsize #figsize 

        #Creating figure 
        self.fig = plt.figure(figsize=self.figSize)
        self.pltColours = ["royalblue", "green", "red", "orchid"]
    
    def plot3D(self, U, plotNum, title="Plot", axlab = ['x', 'y', 'z'], title_font_size=10, axes_fontsize = [10, 10, 10], tick_label_size=10, tick_label_color = 'k'):
        self.ax = self.fig.add_subplot(self.figRow, self.figCol, plotNum, projection='3d') #setting which plot 

        #Setting plot attributes
        self.ax.set_title(title, fontsize=title_font_size)
        self.ax.set_xlabel(axlab[0], fontsize = axes_fontsize[0])
        self.ax.set_ylabel(axlab[1], fontsize = axes_fontsize[1])
        self.ax.set_zlabel(axlab[2], fontsize = axes_fontsize[2])
        self.ax.tick_params(direction='out', length=6, width=2, colors=tick_label_color,
                                grid_color=tick_label_color, grid_alpha=1, labelsize=tick_label_size)
        #Plotting 
        plt.plot(U[0, :], U[1, :], U[2, :], color=self.pltColours[(plotNum-1) % len(self.pltColours)])
    
    def plot2D(self, x, y, plotNum, title="Plot", axlab = ['x', 'y'], title_font_size=10, axes_fontsize = [10, 10], tick_label_size=0, tick_label_color='k', label="1"):
        
        ax = self.fig.add_subplot(self.figRow, self.figCol, plotNum) #setting which plot 
        
        #Setting plot attributes
        ax.set_title(title, fontsize=title_font_size)
        ax.set_xlabel(axlab[0], fontsize=axes_fontsize[0])
        ax.set_ylabel(axlab[1], fontsize=axes_fontsize[1])
        ax.tick_params(direction='out', length=6, width=2, colors=tick_label_color,
                                grid_color=tick_label_color, grid_alpha=1, labelsize=tick_label_size)
    
        #Plotting 
        plt.plot(x, y, color=self.pltColours[(plotNum-1) % len(self.pltColours)], label=label)

    def add_plot2D(self, x, y, title="Plot", axlab = ['x', 'y'], title_font_size=10, axes_fontsize = [10, 10], tick_label_size=0, tick_label_color='k', label="1"):
        
        #Setting plot attributes
        #ax.set_title(title, fontsize=title_font_size)
        #ax.set_xlabel(axlab[0], fontsize=axes_fontsize[0])
        #ax.set_ylabel(axlab[1], fontsize=axes_fontsize[1])
        #ax.tick_params(direction='out', length=6, width=2, colors=tick_label_color,
         #                       grid_color=tick_label_color, grid_alpha=1, labelsize=tick_label_size)
    
        #Plotting 
        plt.plot(x, y, color=self.pltColours[(plotNum-1) % len(self.pltColours)], label=label)

    def plot_recurrence(self, Matrix, plotNum, title = "Heat Map", axlab = ['x', 'y'] ): #Heat Map / Recurrence Plot
        ax = self.fig.add_subplot(self.figRow, self.figCol, plotNum)

        ax.set_title(title)
        ax.set_xlabel(axlab[0])
        ax.set_ylabel(axlab[1])
        
        plt.imshow(Matrix, cmap='binary', origin='lower')


#Computer Largest Lyapunov Exponent by fixed time Wolfs Algorithim, univariate time series embedded by delay co-ordinate embedding
def LLEwolfs(TimeSeries, theta_max = np.pi/6, epsilon = 5.95, EmbeddingDim = 10, NN_max = 50, Max_iterations = 10000, TimeSeries_dt=0.01, ReplaceTime = 10):
    Lprime = []
    L = []
    T = []
    NN_T = []
    ReplaceTimeIndex = int(ReplaceTime/TimeSeries_dt)

    #epsilon = 5.95 #maximum distance between points 
    #NN_max = 50 #maximum nearest neightbours checked before failure 
    k = i = 0
    theta = 0
    #theta_max = np.pi/6 

    #Delay co-ordinate embedding of time series 
    Ts = DCembedding(ts = TimeSeries, m = EmbeddingDim, tau=1)
    Tree = sp.KDTree(Ts.T)
    dist, index = Tree.query(Ts.T[1, :], k=2, eps=0, p=2)

    def angle(a, b): #angle between two vectors 
        y1 = np.dot(a, b)
        y2 = np.sqrt(np.dot(a, a))*np.sqrt(np.dot(b, b))
        return np.arccos(y1/y2)
    
    dist, index = Tree.query(Ts.T[0, :], k=2, eps=0, p=2)
    L.append(dist[1]); NN_T.append(index[1]); T.append(i)

    for j in range(0, Max_iterations):
        if (L[j] >= epsilon) or (theta >= theta_max):
            print("No valid nearest neighbour iteration j = ", j, ". Distance = ", dist[1])
            break
        else:
            k = 0
            d = L[j]
            for k in range(0, ReplaceTimeIndex):
                if (T[j]+k >= Ts.T.shape[0]):
                    break
                if (NN_T[j]+k >= Ts.T.shape[0]):
                    k = k-1
                    break
                else: 
                    d = sp.distance.pdist( np.array([Ts.T[T[j]+k, :], Ts.T[NN_T[j]+k, :]]) )
                    k = k + 1
                
            Lprime.append(d)

        i = i + k 
        T.append(i) #new time index 
        if i >= Ts.T.shape[0]:
            print("End of Time Series: i = ", i)
            
            break
        else:
            if Lprime[j] <= epsilon:
                L.append(Lprime[j]); NN_T.append(NN_T[j]+k)
                #print("Neighbour Found j = ", j)
            else: 
                for v in range(0, NN_max):
                    dist, index = Tree.query(Ts.T[T[j+1], :], k=v+2, eps=0, p=2) # getting nearest neightbour to new point
                    #finding angle: 
                    a = Ts.T[NN_T[j]+k, :] - Ts.T[T[j+1], :]; b = Ts.T[index[v+1], :] - Ts.T[T[j+1], :]
                    theta = angle(a, b)
                    if (theta < theta_max) and (dist[1]<epsilon):
                        L.append(dist[1]); NN_T.append(index[1])
                        #print("New Neighbour Found j = ", j)
                        break
                if (theta >= theta_max) or (dist[1]>=epsilon):
                    print("No valid nearest neighbour on iteration j = ", j)
                    break 

    M = len(Lprime)
    N = T[-1]*0.01
    sum = 0 
    for i in range(0, M-1):
        sum = sum + np.log2(Lprime[i]/L[i])
    
    return sum/N








#linear interpolation
#takes points x1, x2. 
#Current time 
#dt is time step for time series. 
def lin_interpolate(x1, x2, t, dt): 
    if len(x1) != len(x2):
        raise ValueError("Points not same dimension")
    
    t_step = t % dt
    

    Dim = len(x1)
    out = np.zeros(Dim)
    slope = x2 - x1
    step = slope*t_step
    out = step + x1
    
    return out

#function from regularly spaces time series. 
#Linearly interpolates multivarible data points
#Creates object with the time series.  
class TS_CtsFunction:
    def __init__(self, ts, dt):
        self.V = ts
        self.dt = dt

    def f(self, t):
        if t % self.dt == 0: 
            index = int(t/self.dt)
            return self.V[:, index]
        
        else:
            index = int(t // self.dt)
            step = t % self.dt
            x1 = self.V[:, index]
            x2 = self.V[:, index+1]
            out = lin_interpolate(x1, x2, step, dt=self.dt)
            return out

#Compute Statisics for Phase Synchronisation given time series 
def PSstat1(A0, B0):
    RM_A1 = rp.RecurrenceMatrix(A0, p = 2, tol = 3)
    RM_B1 = rp.RecurrenceMatrix(B0, p = 2, tol = 3) #Recurrence Matrices for each 

    #Class to calculate properties from recurrence matrices 
    a1 = rp.recurrence_functions(RM_A1)
    b1 = rp.recurrence_functions(RM_B1) 

    #Probability of recurrence after tau for every tau. 
    p_a1 = a1.p_tau_dist()
    p_b1 = b1.p_tau_dist()

    return stats.pearsonr(p_a1,p_b1).statistic


#S_A and S_B must have shape (Dim, Length)
def getSurrogates(S_A, S_B, N = 10, Time = 100, dt = 0.01):
    L = int(Time/dt)
    SurrogatesA = []
    SurrogatesB = []
    if (L*N > S_A.shape[1]):
        print("Insuffient Data to Generate Surrogates: Total length of surrogates exceeds time series provided")
        return 0
    else: 
        for i in range(0, N):
            SurrogatesA.append(S_A[:, (i*L):((i+1)*(L))])
            SurrogatesB.append(S_B[:, (i*L):((i+1)*(L))])
        return SurrogatesA, SurrogatesB