import os
import numpy as np
import scipy as sp
import scipy.io as io
import pandas as pd
from scipy.optimize import minimize


class BLP():
    def __init__(self):
        
        # Use the two files ps2.mat and iv.mat
        # ps2.mat contains the matrices v, demogr, x1, x2, s_jt, id_demo
        
        ps2 = io.loadmat('ps2.mat')
        iv = io.loadmat('iv.mat')
        
        self.ns = 20       # number of simulated "indviduals" per market 
        self.nmkt = 94     # number of markets = (# of cities)*(# of quarters)  
        self.nbrn = 24     # number of brands per market. if the numebr differs by market this requires some "accounting" vector
                           # Default is ns=20, nmkt=94, nbrn=24. Thus we have 24*94=2256 observations
                
        # x1 variables enter the linear part of the estimation and include the 
        # price variable and 24 brand dummies.         
        self.x1 = ps2['x1']
        
        # x2 variables enter the non-linear part and include a constant, price,
        # sugar content, a mushiness dummy. 
        self.x2 = ps2['x2']
        
        # the market share of brand j in market t
        self.s_jt = ps2['s_jt']

        # Random draws. For each market 80 iid normal draws are provided.
        # They correspond to 20 "individuals", where for each individual
        # there is a different draw for each column of x2. 
        self.v = ps2['v']

        # draws of demographic variables from the CPS for 20 individuals in each
        # market. The first 20 columns give the income, the next 20 columns the
        # income squared, columns 41 through 60 are age and 61 through 80 are a
        # child dummy variable. 
        self.demogr = ps2['demogr']

        self.vfull = np.array([np.repeat(i, 24).reshape(80, 24) for i in self.v]).swapaxes(1, 2).reshape(2256, 80)
        self.dfull = np.array([np.repeat(i, 24).reshape(80, 24) for i in self.demogr]).swapaxes(1, 2).reshape(2256, 80)
        # the matrix v has 80 iid normal random numbers for each of the 94 observations
        # the matrix demogr has random draws from the CPS for 20 individuals per obs.
        
        self.IV = np.matrix(np.concatenate((iv['iv'][:, 1:], self.x1[:, 1:].todense()), 1))
        # The previous line creates a matrix IV from the instruments and the x's.
        # 20 columns of instruments for price by default, and nbrn brand dummies
        # 20 columns from iv, 24 columns from x1. 24 is brand dummies 

        
        # this vector relates each observation to the market it is in
        self.cdid = np.kron(np.array([i for i in range(self.nmkt)], ndmin=2).T, np.ones((self.nbrn, 1)))
        self.cdid = self.cdid.reshape(self.cdid.shape[0]).astype('int')

        ## this vector provides for each index the of the last observation 
        ## in the data used here all brands appear in all markets. if this 
        ## is not the case the two vectors, cdid and cdindex, have to be   
        ## created in a different fashion but the rest of the program works fine.
        ## cdindex = [nbrn:nbrn:nbrn*nmkt]';
        self.cdindex = np.array([i for i in range((self.nbrn - 1),self.nbrn * self.nmkt, self.nbrn)])
        
        
        ## starting values for theta2. zero elements in the following matrix 
        ## correspond to coeff that will not be max over,i.e are fixed at zero.
        ## The rows are coefficients for the constant, price, sugar, and mushy
        ## dummy, respectively. The columns represent F, and interactions with
        ## income, income squared, age, and child.
        
        
        ##                           S.D   Income    Inc^2       Age   Child      
        self.theta2w =  np.array([[0.3302,  5.4819,       0,   0.2037,      0],         #constant
                                 [ 2.4526, 15.8935, -1.2000,        0, 2.6342],         #price
                                 [ 0.0163, -0.2506,       0,   0.0511,      0],         #suger
                                 [ 0.2441,  1.2650,       0,  -0.8091,      0]])        #mushy


        ## create a vector of the non-zero elements in the above matrix, and the 
        ## corresponding row and column indices. this facilitates passing values  
        ## to the functions below. 
        
        self.theti, self.thetj = list(np.where(self.theta2w != 0))
        # the above line returns the (i,j) cordinate for nonzero estimates 
        
        self.theta2_init = self.theta2w[np.where(self.theta2w != 0)]
        # theta2 is the initial nonlinear parameter guess 
        
        
        ## create initial weight matrix
        self.invA = np.linalg.inv(np.dot(self.IV.T, self.IV))
        
        
        ## Logit results and save the mean utility as initial values for the
        ## search below.
        ## compute the outside good market share by market
        temp = np.cumsum(self.s_jt)
        sum1 = temp[self.cdindex]
        sum1[1:] = np.diff(sum1)
        outshr = np.array([np.repeat(1 - i, 24) for i in sum1]).reshape(len(temp), 1)
        
        ## compute logit results and save the mean utility as initial values for the search below
        y = np.log(self.s_jt) - np.log(outshr)
        mid = self.x1.T @ self.IV @ self.invA @ self.IV.T
        t = np.linalg.inv(mid @ self.x1) @ mid @ y
        self.d_old = self.x1 @ t
        self.d_old = np.exp(self.d_old)

        self.gmmvalold = 0
        self.gmmdiff = 1

        self.iter = 0
        
    def mktsh(self):
        # compute the market share for each product
        temp = self.ind_sh().T
        f = sum(temp) / float(self.ns)
        return f.T
    
    def ind_sh(self):
        self.expmu = np.exp(self.mufunc(self.theta2w))
        eg = np.multiply(self.expmu, np.kron(np.ones((1, self.ns)), self.d_old))
        temp = np.cumsum(eg, 0)
        sum1 = temp[self.cdindex, :]
        sum2 = sum1
        sum2[1:sum2.shape[0], :] = np.diff(sum1.T).T
        denom1 = 1. / (1. + sum2)
        denom = denom1[self.cdid, :]
        return np.multiply(eg, denom)   
    
    def meanval(self, theta2, theti, thetj):
        if self.gmmdiff < 1e-6:
            etol = self.etol = 1e-13
        elif self.gmmdiff < 1e-3:
            etol = self.etol = 1e-12
        else:
            etol = self.etol = 1e-9
        norm = 1
        avgnorm = 1
        i = 0
        theta2w = np.zeros((max(theti) + 1,max(thetj) + 1))
        for ind in range(len(theti)):
            self.theta2w[theti[ind], thetj[ind]] = theta2[ind]
        self.expmu = np.exp(self.mufunc(theta2w))
        while norm > etol:
            pred_s_jt = self.mktsh()
            self.d_new = np.multiply(self.d_old,self.s_jt) / pred_s_jt
            t = np.abs(self.d_new - self.d_old)
            norm = np.max(t)
            avgnorm = np.mean(t)
            self.d_old = self.d_new
            i += 1
        #print ('# of iterations for delta convergence:', i)
        return np.log(self.d_new)       
    
    def mufunc(self, theta2w):
        n,k = self.x2.shape
        j = theta2w.shape[1]-1
        mu = np.zeros((n, self.ns))
        for i in range(self.ns):
            v_i = self.vfull[:, np.arange(i, k * self.ns, self.ns)]
            d_i = self.dfull[:, np.arange(i, j * self.ns, self.ns)]
            temp = d_i * np.matrix(theta2w[:, 1:(j+1)].T)
            mu[:, i]=(((np.multiply(self.x2, v_i) * theta2w[:, 0]) +
                      np.multiply(self.x2, temp)) * np.ones((k, 1))).flatten()
        return mu

    
    def jacob(self, *args):
        theta2w = np.zeros((max(self.theti) + 1, max(self.thetj) + 1))
        for ind in range(len(self.theti)):
            theta2w[self.theti[ind], self.thetj[ind]] = self.theta2[ind]
        self.expmu = np.exp(self.mufunc(theta2w))
        shares = self.ind_sh()
        n,K = self.x2.shape
        J = theta2w.shape[1] - 1
        f1 = np.zeros((self.cdid.shape[0] , K * (J + 1)))

        for i in range(K):
            xv = np.multiply(self.x2[:, i].reshape(self.nbrn*self.nmkt, 1) * np.ones((1, self.ns)),
                             self.v[self.cdid, self.ns*i:self.ns * (i + 1)])
            temp = np.cumsum(np.multiply(xv, shares), 0)
            sum1 = temp[self.cdindex, :]
            sum1[1:sum1.shape[0], :] = np.diff(sum1.T).T
            f1[:,i] = (np.mean((np.multiply(shares, xv - sum1[self.cdid,:])).T,0).T).flatten()

        for j in range(J):
            d = self.demogr[self.cdid, self.ns * j:(self.ns * (j+1))]   
            temp1 = np.zeros((self.cdid.shape[0], K))
            for i in range(K):
                xd = np.multiply(self.x2[:, i].reshape(self.nbrn * self.nmkt, 1) * np.ones((1, self.ns)), d)
                temp = np.cumsum(np.multiply(xd, shares), 0)
                sum1 = temp[self.cdindex, :]
                sum1[1:sum1.shape[0], :] = np.diff(sum1.T).T
                temp1[:, i] = (np.mean((np.multiply(shares, xd-sum1[self.cdid, :])).T, 0).T).flatten()
            f1[:, K *(j + 1):K * (j + 2)] = temp1
            
            
        # computing (partial delta)/(partial theta2)
        rel = self.theti + (self.thetj - 1) * max(self.theti)
        rel = np.array([0, 1 , 2 , 3 , 4 , 5 , 6 , 7 , 9 , 12, 14, 15, 17]).flatten()
        f = np.zeros((self.cdid.shape[0], rel.shape[0]))
        n = 0

        for i in range(self.cdindex.shape[0]):
            temp = shares[n:(self.cdindex[i] + 1), :]
            H1 = temp * temp.T
            H = (np.diag(np.array(sum(temp.T)).flatten()) - H1) / self.ns
            f[n:(self.cdindex[i]+1), :] = -np.linalg.inv(H)*f1[n:(self.cdindex[i] + 1), rel]
            n = self.cdindex[i] + 1
        return f

    def varcov(self):
        N = self.x1.shape[0]
        Z = self.IV.shape[1]
        theta2w = np.zeros((max(self.theti) + 1, max(self.thetj) + 1))
        for ind in range(len(self.theti)):
            theta2w[self.theti[ind], self.thetj[ind]] = self.theta2[ind]
        Q = self.IV.T @ np.hstack((self.x1.todense() , self.jacob())) 
        
        
        a =  np.linalg.inv(Q.T @ self.invA @ Q)
        IVres = np.multiply(self.IV, self.gmmresid * np.ones((1, Z)))
        omega = IVres.T @ IVres
        
        f =  a @ Q.T @ self.invA @ omega @ self.invA @ Q @ a
        return f
         
    def gradient_GMM (self):
        # Return gradient of GMM objective function
        jacob = self.jacob()
        gradient = 2*jacob.T @ self.IV @ self.invA @ self.IV.T @ self.gmmresid
        return gradient
    
    def gmmobj(self, theta2, theti, thetj):
        # compute GMM objective function
        self.delta = self.meanval(theta2, theti, thetj)
        self.theta2 = theta2
        ##% the following deals with cases where the min algorithm drifts into region where the objective is not defined
        if max(np.isnan(self.delta)) == 1:
            f = 1e+10
        else:
            temp1 = self.x1.T @ self.IV
            temp2 = self.delta.T @ self.IV
            #self.theta1 = np.linalg.inv(temp1 @ self.invA @ temp1.T) @ temp1 @ self.invA @ temp2.T
            self.theta1 = sp.linalg.solve(temp1 @ self.invA @ temp1.T, temp1 @ self.invA @ temp2.T)
            self.gmmresid = self.delta - self.x1 @ self.theta1
            temp1 = self.gmmresid.T @ self.IV

            f = temp1 @ self.invA @ temp1.T

        

        self.gmmvalnew = f[0,0]
        if self.gmmvalnew < self.gmmvalold:
            self.iter += 1
            
            if self.iter % 10 ==0:
                print ('# of valuations:', self.iter)
                print ('gmm objective:', self.gmmvalnew)
            
        self.gmmdiff = np.abs(self.gmmvalold - self.gmmvalnew)
        self.gmmvalold = self.gmmvalnew
        return(f[0, 0])
    
    def result(self, theta1, rex):
        D_names = ['Mean', 'SD', 'Income', 'Income^2', 'Age', 'Child']
        V_names = ['Constant', 'Price', 'Sugar', 'Mushy']


        x = np.delete(self.x2[:24,:], 1,1)
        d = theta1[1:]
        Var = self.varcov()
        v = Var[1:self.theta1.shape[0], 1:self.theta1.shape[0]]
        v_inv = np.linalg.inv(v)
        y = x.T @ v_inv @ d
        L = np.linalg.inv(x.T @ v_inv @ x)
        t = L @ y 
        
        beta_se = np.sqrt(L.diagonal())
        
        a = np.zeros((4,5))
        t = np.insert(t, 1, theta1[0], axis=0) 
        k = 0
        for i in range(len(theti)):
            a[theti[i]][thetj[i]] = rex[k]
            k+=1

 
        df1 = pd.DataFrame(t, index = V_names )
        df2 = pd.DataFrame(a, index = V_names )
        result = pd.concat([df1, df2], axis=1)
        result.columns = D_names
        
        se_all = self.cal_se(Var)
        beta_se = np.insert(beta_se, 1, se_all[:,0])
        other_se = np.zeros((4,5))
        j = 0
        for p in range(len(theti)):
            other_se[theti[p]][thetj[p]] = se_all[:,25+j]
            j+=1
            
            
        df3 = pd.DataFrame(beta_se.T, index = V_names )
        df4 = pd.DataFrame(other_se, index = V_names )
        se = pd.concat([df3, df4], axis=1)
        se.columns = D_names
        self.se = se 
        
        return result 
    
    def cal_se(self, varcov):
        varcov = self.varcov()
        se_all = np.sqrt(varcov.diagonal())
        return se_all
    
if __name__ == '__main__':
    import time
    start_time = time.time()
    
    blp = BLP()
    init_theta = blp.theta2_init
    theti = blp.theti
    thetj = blp.thetj
    #gradient = blp.gradient_GMM()
    res = minimize(blp.gmmobj, init_theta, args=(theti, thetj), method='BFGS', options={'maxiter':40, 'disp' : True})
    print('Mean Estimates:')
    print(blp.result(blp.theta1, res.x))
    
    print('Standard Errors:')
    print(blp.se)
    # Nelder-Mead
    print("--- %s seconds ---" % (time.time() - start_time))
