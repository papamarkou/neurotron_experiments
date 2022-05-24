import numpy as np

class NeuroTron:
    def __init__(self, sample_data=None, w_star=None, d=None, eta=None, b=None, width=None, filter=None):
        self.sample_data = sample_data

        self.reset(w_star, d, eta, b, width, filter)

    def reset(self, w_star, d, eta, b, width, filter):
        if (w_star is not None) and (filter is not None):
            assert(len(w_star) == filter)

        self.w_true = w_star.copy() if (w_star is not None) else w_star
        self.dim = d
        self.step = eta
        self.minibatch = b
        self.w = width  # The w in the paper is the width of the net
        self.r = filter # The r in the paper - the filter dimension < dim

        if (d is not None) and (width is not None) and (filter is not None):
            self.w_now_tron = np.ones((filter, 1)) if (filter is not None) else None # Initial point for NeuroTron
            self.w_now_sgd = np.ones((filter, 1)) if (filter is not None) else None # Initial point for SGD

            # Choosing the M matrix
            M_X = np.random.randn(filter, filter)
            M_Y = np.random.randn(filter, d-filter)
            self.M = np.concatenate((M_X, M_Y), axis=1)

            # Fixing the neural net
            self.A_list = []
            C = np.random.randn(filter, d)

            c = 0
            k = width/2
            for i in range (width+1):
                factor =  (-k+c)
                if factor != 0:
                    Z = self.M+factor*C
                    self.A_list.append(Z)

                c+=1

    def err_tron(self):
        return np.linalg.norm(self.w_true-self.w_now_tron)

    def err_sgd(self):
        return np.linalg.norm(self.w_true-self.w_now_sgd)

    def attack(self,bound,beta):
        b = self.minibatch
        u = np.random.uniform(0,1,b)
        v = u <= beta
        x = v* np.resize([-1*bound, 1*bound], b)
        return x

    # A_i in filter x dim 
    # weight is filter x 1
    # data is b x dim  
    def net(self, data, weight):
        sum = 0 
        for i in range(self.w):
            y_fake_now = np.matmul(weight.transpose(), np.matmul(self.A_list[i], data.transpose())) # = w^TA_ix 
            # y_fake_now is 1 x b 
            indi = (y_fake_now > 0).astype(float) 
            sum += indi*y_fake_now # (= max(0, xA_i^Tw))

        return (sum/self.w).flatten() 

    def net_der(self, data,weight):
        sum = 0 
        for i in range(self.w):
            # data^T is dim x b 
            Aix = np.matmul(self.A_list[i], data.transpose())
            # A_ix is r x b 
            y_fake_now = np.matmul(weight.transpose(), Aix) # = w^TA_ix 
            # y_fake_now is 1 x b 
            indi = (y_fake_now > 0).astype(float) 
            # 1(w^TA_ix >0) is 1 x b 
            indi = np.diag(indi[0])
            
            # indi is b x b 
            indAix = np.matmul(Aix, indi) 
            # ind*A_ix is r x b 
            sum += indAix

        
        final =  (sum/self.w) # r x b 
        return final   

    # M is r x dim 
    # w_now_tron is a r x 1 current point
    # inputs are 1 x dim  
    def update_tron(self, bound, beta):
        data = self.sample_data([self.minibatch, self.dim])
        y_oracle = self.net(data, self.w_true)
        poison = self.attack(bound, beta)
        
        y_oracle += poison 
        y_now = self.net(data, self.w_now_tron) # 1 x b
        
        sum = 0
        for i in range(0, self.minibatch):
            sum += (y_oracle[i]-y_now[i])*data[i, :]
        
        g_tron = (1/self.minibatch)*np.matmul(self.M, sum.reshape(self.dim, 1))
        self.w_now_tron += self.step*g_tron
        return self.err_tron()

    def update_sgd(self, bound, beta):
        data = self.sample_data([self.minibatch, self.dim])
        y_oracle = self.net(data, self.w_true)
        poison = self.attack(bound, beta)
        
        y_oracle += poison 
        y_now = self.net(data, self.w_now_sgd) # 1 x b
        net_der_now = self.net_der(data, self.w_now_sgd)

        sum = 0
        for i in range(0,self.minibatch):
            sum += (y_oracle[i]-y_now[i])*np.reshape(net_der_now[:, 0], (self.r, 1))
        
        g_sgd = (1/self.minibatch)*sum 
        self.w_now_sgd += self.step*g_sgd
        return self.err_sgd()

    def run(self, filterlist, dlist, boundlist, betalist, etalist, blist, width, num_iters, run_sgd=False, verbose=True):
        filter_len = len(filterlist)
        d_len = len(dlist)
        bound_len = len(boundlist)
        beta_len = len(betalist)
        eta_len = len(etalist)
        b_len = len(blist)

        if verbose:
            num_runs = filter_len * d_len * bound_len * beta_len * eta_len * b_len
            ir = 0
            msg = 'Iteration {:' + str(len(str(num_runs))) + '} out of ' + str(num_runs)

        tron_error = np.empty([num_iters, filter_len, d_len, bound_len, beta_len, eta_len, b_len])

        if run_sgd:
            sgd_error = np.empty([num_iters, filter_len, d_len, bound_len, beta_len, eta_len, b_len])
        else:
            sgd_error = None

        for ifilter, filter in enumerate(filterlist):
            # Choosing the ground truth w_* from a Normal distribution
            w_star = np.random.randn(filter, 1)

            for id, d in enumerate(dlist):
                for ibound, bound in enumerate(boundlist):
                    for ibeta, beta in enumerate(betalist):
                        for ieta, eta in enumerate(etalist):
                            for ib, b in enumerate(blist):
                                if verbose:
                                    ir += 1
                                    print(msg.format(ir, num_runs))

                                self.reset(w_star, d, eta, b, width, filter)

                                for i in range(num_iters):
                                    tron_error[i, ifilter, id, ibound, ibeta, ieta, ib] = self.update_tron(bound, beta)

                                    if run_sgd:
                                        sgd_error[i, ifilter, id, ibound, ibeta, ieta, ib] = self.update_sgd(bound, beta)

        return tron_error, sgd_error
