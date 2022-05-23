class NeuroTron:
    def __init__(self, w_star, d, eta, b, width, filter):
        assert(len(w_star) == filter)

        self.w_true = w_star
        self.dim = d
        self.w_now = np.ones((filter, 1)) # Fixed initial point for all experiments 
        self.step = eta
        self.minibatch = b 
        self.w = width  # The w in the paper is the width of the net 
        self.r = filter # The r in the paper - the filter dimension < dim

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

        # Elements of A_list are r x dim 
        # One can check that average(A_list) = M = something of full rank = r 
        sum = 0 
        for i in range(width):
            sum += self.A_list[i] 

        avg = sum/width  

    def err(self):
        return np.linalg.norm(self.w_true-self.w_now)    

    def sample(self,mu,sigma):    
        return mu + sigma*np.random.randn(self.minibatch, self.dim)

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
    # w_now is a r x 1 current point 
    # inputs are 1 x dim  
    def update_neuro(self, mu, sigma, bound, beta): 
        data = self.sample(mu, sigma) # b x dim sized data matrix sampled from N(mu,sigma)
        y_oracle = self.net(data, self.w_true)
        poison = self.attack(bound, beta)
        
        y_oracle += poison 
        y_now = self.net(data, self.w_now) # 1 x b 
        
        sum = 0
        for i in range(0, self.minibatch):
            sum += (y_oracle[i]-y_now[i])*data[i, :]
        
        g_tron = (1/self.minibatch)*np.matmul(self.M, sum.reshape(self.dim, 1))
        self.w_now += self.step*g_tron
        return self.err()

    def update_sgd(self, mu, sigma, bound, beta): 
        data = self.sample(mu, sigma) # b x dim sized data matrix sampled from N(mu, sigma)
        y_oracle = self.net(data, self.w_true)
        poison = self.attack(bound, beta)
        
        y_oracle += poison 
        y_now = self.net(data, self.w_now) # 1 x b 
        net_der_now = self.net_der(data, self.w_now) 

        sum = 0
        for i in range(0,self.minibatch):
            sum += (y_oracle[i]-y_now[i])*np.reshape(net_der_now[:, 0], (self.r, 1))
        
        g_sgd = (1/self.minibatch)*sum 
        self.w_now += self.step*g_sgd
        return self.err()
