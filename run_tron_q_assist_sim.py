#%% -*- coding: utf-8 -*-
"""NC submission Neurotron q assist

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Grqd8YloStHVD0eoAnUtOJ3jSxX1A8rA

#Introduction
"""

# %% Import packages

import numpy as np
import random 
from random import sample
import pandas as pd
from numpy.linalg import  matrix_rank as rank

from sim_setup import output_path

# %% Set seed

np.random.seed(seed=1001)

#%% An example of the basic ploting mechanism  

#type(data),np.shape(data)

y = [[i**2 for i in range(1,10)],[i**3 for i in range(1,10)] ]
x = [i for i in range(1,10)]

# %% Defining the SGD & Stochastic_NeuroTron class

class SGD_NeuroTron:
    def __init__(self,w_star,d,eta,b,width,filter,q,M,C):
        assert(len(w_star) == filter)
        self.w_true = w_star
        self.dim = d
        self.w_now = np.ones((filter,1)) #np.random.randn(filter,1) #random normal initialization 
        self.step = eta
        self.minibatch = b 
        self.w = width  #the w in the paper is the width of the net 
        self.r = filter #the r in the paper - the filter dimension << dim
        self.q = q
        self.M = M
        self.C = C
        
        self.A_list = []

        c = 0 
        k = width/2 
        for i in range (width+1): 
            factor =  (-k+c)
            #print ("factor = ",factor)
            if factor != 0:
               Z = self.M + factor*self.C
               Z = Z/1
               self.A_list.append(Z)

            c+=1 

        #Elements of A_list are r x dim 
        #Here we check that average(A_list) = M = something of full rank = r 
        sum = 0 
        for i in range(width):
            sum += 1*self.A_list[i] 

        avg = sum/width

        # print ("Filter dimension = ",self.r," Input dimension = ",self.dim," Shape of Avg-A = ",np.shape(avg)," Shape of M = ",np.shape(self.M))
        # print ("Rank of the (M,average  A_i) =(", rank(self.M), ",", rank(avg), ") ||Avg-A - M|| =",np.linalg.norm(avg - self.M)) 
        #print ("True weight =",self.w_true)
        #print ("Intiial weight =",self.w_now)

    def err(self):
        return np.linalg.norm(self.w_true-self.w_now)    

    def sample(self,mu,sigma):    
        return mu + sigma*np.random.randn(self.minibatch,self.dim)

    def attack(self,bound,beta):
        b = self.minibatch
        u = np.random.uniform(0,1,b)
        v = u <= beta
        x = v* np.resize([-1*bound,1*bound], b)
        return x

    #A_i in filter x dim 
    #weight is filter x 1
    #data is b x dim  
    def net (self,data,weight):
        sum = 0 
        for i in range(self.w):
            #print ("Shape of data^T =",np.shape(data.transpose()))
            #print ("Shape of A[i] =",np.shape(self.A_list[i]))
            #print ("Shape of weight^T =",np.shape(weight.transpose()))
            y_fake_now = np.matmul(weight.transpose(),np.matmul(self.A_list[i],data.transpose())) #= w^TA_ix 
            #y_fake_now is 1 x b 
            indi = (y_fake_now > 0).astype(float) 
            sum += self.q*indi*y_fake_now #(= max(0, xA_i^Tw))  #random.uniform(5, 10) does the job of qi

        return (sum/self.w).flatten() 

    def net_der (self,data,weight):
        sum = 0 
        for i in range(self.w):
            ##print ("Shape of data^T =",np.shape(data.transpose())) #data^T is dim x b ? 
            #print ("Shape of A[i] =",np.shape(self.A_list[i]))
            #print ("Shape of weight^T =",np.shape(weight.transpose()))
            Aix = np.matmul(self.A_list[i],data.transpose())
            ##print ("Shape of A_ix =",np.shape(Aix)) #A_ix is r x b ? 
            y_fake_now = np.matmul(weight.transpose(),Aix) #= w^TA_ix 
            #y_fake_now is 1 x b 
            indi = (y_fake_now > 0).astype(float) #1(w^TA_ix >0) is 1 x b 
            indi = np.diag(indi[0])
            
            ##print ("Shape of indi=",np.shape(indi)) # b x b ? 
            ##print (indi) 

            indAix = self.q*np.matmul(Aix,indi) 
            ##print ("Shape of indi*A_ix =",np.shape(indAix)) #ind*A_ix is r x b ? 
            ##print (indAix) 
            sum += indAix

        #final =  (sum/self.w).flatten() 
        final =  (sum/self.w) # r x b 
        ##print ("Shape of final =",np.shape(final)) #final is r x b ? 
        return final   


    #M is rx dim 
    #w_now is a r x 1 current point 
    #inputs are 1 x dim  
    def update_neuro (self,w_now,mu,sigma,bound,beta): 
        data = self.sample(mu,sigma) #b x dim sized data matrix sampled from N(mu,sigma)
        y_oracle = self.net(data,self.w_true)
        poison = self.attack(bound,beta)
        
        #print ("Shape of poison =",np.shape(poison),poison) # [b,] 
        #print ("Shape of y_oracle =",np.shape(y_oracle),y_oracle) # 1 x b 
        y_oracle += poison #np.reshape(poison,(self.minibatch,1)) 
        #print ("Shape of y_oracle post-attack =",np.shape(y_oracle),y_oracle)
        y_now = self.net(data,self.w_now)
        #print ("Shape of y_now =",np.shape(y_now),y_now) # 1 x b 
        
        sum = 0
        for i in range(0,self.minibatch):
            #print ("y_oracle[i] = ",y_oracle[i])
            #print ("y_now[i] = ", y_now[i])
            sum += (y_oracle[i] -y_now[i])*data[i,:]
        
        g_tron = (1/self.minibatch)*np.matmul(self.M,sum.reshape(self.dim,1))
        self.w_now += self.step * g_tron 
        return self.err()
        
    def update_sgd (self,w_now,mu,sigma,bound,beta): 
        data = self.sample(mu,sigma) #b x dim sized data matrix sampled from N(mu,sigma)
        y_oracle = self.net(data,self.w_true)
        poison = self.attack(bound,beta)
        
        #print ("Shape of poison =",np.shape(poison),poison) # [b,] 
        #print ("Shape of y_oracle =",np.shape(y_oracle),y_oracle) # 1 x b 
        y_oracle += poison #np.reshape(poison,(self.minibatch,1)) 
        #print ("Shape of y_oracle post-attack =",np.shape(y_oracle),y_oracle)
        y_now = self.net(data,self.w_now)
        #print ("Shape of y_now =",np.shape(y_now),y_now) # 1 x b 
        net_der_now = self.net_der(data,self.w_now) 

        sum = 0
        for i in range(0,self.minibatch):
            #print ("y_oracle[i] = ",y_oracle[i])
            #print ("y_now[i] = ", y_now[i])
            sum += (y_oracle[i] -y_now[i])*np.reshape(net_der_now[:,0],(self.r,1))
        
        g_sgd = (1/self.minibatch)*sum 
        self.w_now += self.step * g_sgd 
        return self.err()


#%% Running SGD & Stochastic NeuroTron : fixed beta, varying theta*

#Choose w_initial as the all ones vector but sample w_star from a normal 
#theta* = 0 works : dim = 100, filter = 5, width = 4, eta = 10^(-4), b = 2^4 
#All theta* works : dim = 50, filter = 20, width = 100, eta = 10^(-4), b = 2^6  

dlist = [50] #data dimension in 100, 50, 20, 10

width = 100 #choose "width" as an even number

etalist = [0.0001] #step-lengths in 0.05,0.1,0.2
blist = [2**6] #[2**2,2**4,2**6,2**8,2**10] #mini-batch  
mu = 0 #mu of the Gaussian data distribution N(mu,sigma) 
sigma = 1 #sigma of the Gaussian data distribution N(mu,sigma)  

#works for dlist 50, filter 20, width 100,eta = 10^(-4), b = 2^6  
#betalist = [0.005,0.5] #[0,0.005,0.05,0.1,0.2,0.5,0.9] 
#boundlist =  [0] #[0,2**(-1),2**0,2**1]   #[0,2**(-3),2**(-2),2**0,2**1,2**2] 

betalist = [0.05] #[0.005,0.5] #[0,0.005,0.05,0.1,0.2,0.5,0.9] 
boundlist =  [0,2**(-1),2**0]   #[0,2**(-3),2**(-2),2**0,2**1,2**2] 
filterlist = [20]

d0=50
filter0=20
#Choosing the "M" matrix 
M_X = np.random.randn(filter0,filter0)
M_Y = np.random.randn(filter0,d0-filter0) 
M = np.concatenate((M_X,M_Y),axis=1)
C = np.random.randn(filter0,d0)

samples = 1 # 5
iterations = 4*(10**4)

k = 0

for filter in filterlist:
    w_star = np.random.randn(filter,1)  #Choosing the w_* from a Normal distribution 
    #print(w_star)
    for d in dlist:
        for bound in boundlist: 
            for beta in betalist: 
                for eta in etalist:  
                    for b  in blist:    
     
                        #err_final_sgd = []
                        err_final_neuro1 = []
                        err_final_neuro10 = [] 
                        for s in range(samples):
                            err_list_sgd = []
                            err_list_neuro1 = []
                            err_list_neuro10 = []
                            #(self,w_star,d,eta,b,width,filter)
                            q0 = 10
                            SN_neuro10 = SGD_NeuroTron(w_star,d,eta/5,b,width,filter,10,M,C)
                            SN_neuro1 = SGD_NeuroTron(w_star,d,eta,b,width,filter,1,M,C)  
                            #SN_sgd = SGD_NeuroTron(w_star,d,eta/10,b,width,filter,q0) 

                            for i in range(iterations):
                                #(SN.w_now,mu,sigma,bound,beta)  
                                err_list_neuro1.append(SN_neuro1.update_neuro(SN_neuro1.w_now,mu,sigma,bound,beta))
                                err_list_neuro10.append(SN_neuro10.update_neuro(SN_neuro1.w_now,mu,sigma,bound,beta))
                                #err_list_sgd.append(SN_sgd.update_sgd(SN_sgd.w_now,mu,sigma,bound,beta))

                            #print ("At sample index =",s," the error =",err_list[iterations-1])
                            #err_final_sgd = np.sum(np.array([err_final_sgd,err_list_sgd]),axis=0)
                            err_final_neuro1 = np.sum(np.array([err_final_neuro1,err_list_neuro1]),axis=0)
                            err_final_neuro10 = np.sum(np.array([err_final_neuro10,err_list_neuro10]),axis=0)

                        np.savetxt(output_path.joinpath('q_assist_neuro1_'+str(k)+'_tron.csv'), err_final_neuro1, delimiter=',')
                        np.savetxt(output_path.joinpath('q_assist_neuro10_'+str(k)+'_tron.csv'), err_final_neuro10, delimiter=',')
                        # np.savetxt(output_path.joinpath('q_assist_neuro1_'+str(k)+'_tron.csv'), err_final_neuro1/samples, delimiter=',')
                        # np.savetxt(output_path.joinpath('q_assist_neuro10_'+str(k)+'_tron.csv'), err_final_neuro10/samples, delimiter=',')

                        k = k + 1

                            #print(s)

                        # print ("(dim,iterations,eta,b,sigma,(beta,attack-bound)) =", (d,iterations,eta,b,filter,(beta,bound)))
                        #print ("final sample averaged error for SGD =", err_final_sgd[iterations-1]/samples) 
                        # print ("final sample averaged error for NeuroTron, (q=1)=", err_final_neuro1[iterations-1]/samples) 
                        # print ("final sample averaged error for NeuroTron, (q=10)=", err_final_neuro10[iterations-1]/samples) 
