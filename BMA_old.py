# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 14:20:04 2016

@author: Michael Zhang
"""

#import pdb
import time
import numpy
import scipy
import itertools
import datetime
import weiszfeld
#import matplotlib.pyplot as plt
import random
from sklearn.cross_validation import train_test_split
#from sklearn import preprocessing
#from sklearn.metrics.pairwise import rbf_kernel
from mpi4py import MPI
#from scipy.spatial.distance import cdist, euclidean
#from pyemd import emd
from sklearn.datasets import make_regression
random.seed(1)

class BMA(object):

    def __init__(self, N=1000000, sigma=1., outlier_num=0, D=10, num_predict=3,
                 prior_a = .1, prior_b = 1./.1, outlier_val = 1000.,
                 verbose=True, predict_N=50, ss=False, diabetes=False, test=.1,
                 model_probs= True, rand_state=111):
        self.comm = MPI.COMM_WORLD
        self.today = datetime.datetime.today()
        self._P = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.verbose = verbose
        self._prior_a = prior_a
        self._prior_b = prior_b
        self._sigma = sigma
        self.rand_state = rand_state

        if diabetes:
            self.diabetes = numpy.loadtxt("diabetes.txt", delimiter=",")
            self._X, self._predict_X, self._Y, self._predict_Y  = train_test_split(self.diabetes[:,1:], self.diabetes[:,0]-self.diabetes[:,0].mean(), test_size = test, random_state = self.rand_state)
            assert(self._predict_X.shape[0] == self._predict_Y.size)
            self._predict_N = self._predict_X.shape[0]
            self.diabetes = True
            self._num_predict = None
            self._beta = None

        else:
            self.diabetes = False
            self._num_predict = num_predict  # number of true predictors in linear model
            self._N = N
            self.outlier_val = outlier_val
            self._outlier_num = outlier_num
            self._D = D
            self._predict_N = predict_N # number of values to predict
            assert(self._D >= self._num_predict)
            if self.comm.rank == 0:
                self.data_generate()
            else:
                self._beta = None
                self._X = None
                self._Y = None
                self._predict_X = None
                self._predict_Y = None
#                self._predict_N = None
        if self.diabetes:
            self.fname_foot = "_" + "diabetes" + "_" + str(self._P ) + "_" + self.today.strftime("%Y-%m-%d-%f") + ".txt"
        else:
            self.fname_foot = "_" + "synthetic" + "_"+ str(self._P ) + "_" + self.today.strftime("%Y-%m-%d-%f") + ".txt"

        self._X = self.comm.bcast(self._X)
        self._beta = self.comm.bcast(self._beta)
        self._Y = self.comm.bcast(self._Y)
        self._predict_X = self.comm.bcast(self._predict_X)
        self._predict_Y = self.comm.bcast(self._predict_Y)
#        self._predict_N = self.comm.bcast(self._predict_N)
        (self._N, self._D) = self._X.shape
        assert(self._N == self._Y.size)

        self.data_partition = self.partition_tuple()
        self.part_size_X = tuple([j * self._D for j in self.partition_tuple()])
        self.part_size_Y = tuple([j for j in self.partition_tuple()])
        self.data_displace_X = self.displacement(self.part_size_X)
        self.data_displace_Y = self.displacement(self.part_size_Y)
        self._X_local = numpy.zeros(self.data_partition[self.rank] * self._D)
        self._Y_local = numpy.zeros(self.data_partition[self.rank])

        self.comm.Scatterv([self._Y, self.part_size_Y, self.data_displace_Y, MPI.DOUBLE], self._Y_local)
        self.comm.Scatterv([self._X, self.part_size_X, self.data_displace_X, MPI.DOUBLE], self._X_local)
        self._X_local = self._X_local.reshape((self.data_partition[self.rank], self._D))
        self._N_p = self._X_local.shape[0]

        if ss == False:
#            print("Creating Models")
            possible_models = self.all_combinations(xrange(self._D))
            full_model_list = {counter:model for counter,model in enumerate(possible_models)}
            reduced_model_enum = enumerate([model for model in full_model_list.values() if (len(model) <= num_predict)])
            self._X_model_names = {counter:model for counter,model in reduced_model_enum}
#            self._X_model_names = {counter:model for counter,model in enumerate(possible_models)}

#            if self._D >= 5:
#                true_model = set(self._beta.nonzero()[0])
#                print("True Model: %s" % true_model)
#                reduced_model_enum = enumerate([model for model in full_model_list.values() if bool(set(model) >= true_model)])
#                self._X_model_names = full_model_list
#            else:
#                print("Model size less than 10")
#                self._X_model_names = full_model_list

            self._K = len(self._X_model_names) # total number of models
            self.AIC = numpy.zeros(self._K)
            self.BIC = numpy.zeros(self._K)
            self.post_beta = numpy.zeros((self._K,self._D))
            self.post_sigma = numpy.random.gamma(self._prior_a,self._prior_b, size=self._K)
            self.prior_model = numpy.ones(self._K)
            self.prior_model /= self.prior_model.sum()
            self._X_model_local = {k:self._X_local[:,self._X_model_names[k]] for k in xrange(self._K)}
#                self.model_prob = numpy.zeros(self._K)
#                for k in xrange(self._K):
#                    if self.rank==0:
#                        print(k)
#                    X = self._X_model_local[k]
#                    self.model_prob[k] = self.log_likelihood_y(X)
            if model_probs:
                self.model_prob = numpy.array([self.log_likelihood_y(self._X_model_local[k]) for k in xrange(self._K)])
                self.model_prob += numpy.log(self.prior_model)
                self.model_prob -= scipy.misc.logsumexp(self.model_prob)
                self.model_prob = numpy.exp(self.model_prob)
                self.model_prob /= numpy.sum(self.model_prob) # posterior model prob, P(M_k|Y)
            self.log_likelihood = numpy.zeros(self._K)
            self.Y_hat = numpy.zeros((self._K, self._predict_N))

        assert(self._N_p > 0)

        self._X = None
#        self._true_X = None
        self._Y = None
        if self.verbose and self.rank ==0:
            print("Initialization complete")
            print("True Model: %s" % str(self._beta.nonzero()[0]))

    def all_combinations(self,any_list):
        return itertools.chain.from_iterable(
            itertools.combinations(any_list, i + 1)
            for i in xrange(len(any_list)))

    def data_generate(self):
        total_X, total_Y, self._beta = make_regression(n_samples = self._N + self._predict_N,
                                                       n_features = self._D,
                                                       n_informative = self._num_predict,
                                                       noise = numpy.sqrt(self._sigma), coef=True, random_state = self.rand_state)
        self._X, self._predict_X, self._Y, self._predict_Y  = train_test_split(total_X, total_Y, test_size=self._predict_N, random_state = self.rand_state)
        if self._outlier_num > 0.:
#            outlier_choice = numpy.random.choice(xrange(self._N), self._outlier_num)
            outlier_choice = numpy.argsort(numpy.abs(self._Y))[-self._outlier_num:]
            outliers = (-self.outlier_val* (self._Y[outlier_choice] < 0) + (self.outlier_val* (self._Y[outlier_choice] >= 0)))
            self._Y[outlier_choice] += outliers

    def partition_tuple(self):
        base = [int(self._N / self._P) for k in range(self._P)]
        remainder = self._N % self._P
        assert(len(xrange(remainder)) <= len(base))
        if remainder:
            for r in xrange(remainder):
                base[r] += 1
        assert(sum(base) == self._N)
        assert(len(base) == self._P)
        return(tuple(base))

    def displacement(self, partition):
        if self._P > 1:
            displace = numpy.append([0], numpy.cumsum(partition[:-1])).astype(int)
        else:
            displace = [0]
        return(tuple(displace))

    def mvn_t(self, x, mu, nu, p, sigma):
        numerator = scipy.special.gammaln((nu+p)/2.0)
        diff = x-mu
        dot_prods = numpy.dot(diff.T, numpy.linalg.inv(sigma))
        dot_prods = numpy.dot(dot_prods, diff)
        denom_inner = 1.0 + (1.0/nu) * dot_prods
        denominator = scipy.special.gammaln(nu / 2.0)
        denominator += (p/2.0) * numpy.log(nu) + (p/2.0) * numpy.log(numpy.pi)
        (sgn, log_det_sigma) = numpy.linalg.slogdet(sigma)
        denominator += .5 * log_det_sigma
        denominator += (.5*(nu+p)) * numpy.log(denom_inner)
        pdf = numerator - denominator
        return(pdf)

    def log_likelihood_y(self,X):
        N,D = X.shape
        prior_sigma = numpy.eye(D)
        likelihood = self._prior_a * numpy.log(self._prior_b)
        likelihood += scipy.special.gammaln(.5*self._N + self._prior_a)
        likelihood -= scipy.special.gammaln(self._prior_a)
        likelihood += (.5*self._N) * numpy.log(self._P / (2*numpy.pi))
#        X_Sigma_XT = self._P * numpy.dot(numpy.dot(X, prior_sigma),X.T)
#        assert(X_Sigma_XT.shape == (self._N_p, self._N_p))
        #I_n = numpy.eye(self._N_p)
        rank_d_matrix = (1./self._P)*prior_sigma + numpy.dot(X.T,X)
        rank_d_inv = numpy.linalg.inv(rank_d_matrix)
        #woodbury_inv = I_n - numpy.dot(numpy.dot(X, rank_d_inv),X.T)
        #woodbury_inv[numpy.diag_indices(self._N_p)] += 1. 

        Y_T_X = numpy.dot(self._Y_local.T, X)
        Y_T_Y =numpy.dot(self._Y_local.T,self._Y_local)
        Y_T_woodbury_inv_Y = Y_T_Y - numpy.dot(numpy.dot(Y_T_X, rank_d_inv),Y_T_X.T)

#        full_inv = numpy.linalg.inv(I_n + X_Sigma_XT)
#        assert(numpy.allclose(full_inv, woodbury_inv))
#        (sgn, logdet) = numpy.linalg.slogdet(I_n + X_Sigma_XT)
        (sgn_w,woodbury_logdet_1) = numpy.linalg.slogdet(rank_d_matrix)
        woodbury_logdet_2 = D*numpy.log(self._P)
        woodbury_logdet = woodbury_logdet_1 + woodbury_logdet_2
#        assert(numpy.allclose(logdet, woodbury_logdet))
#        likelihood -= .5 * logdet
        likelihood -= .5 * woodbury_logdet
#        marg_b = self._prior_b + (self._P * .5) * numpy.dot(numpy.dot(self._Y_local.T, numpy.linalg.inv(I_n + X_Sigma_XT)),self._Y_local)
        #marg_b = self._prior_b + (self._P * .5) * numpy.dot(numpy.dot(self._Y_local.T, woodbury_inv),self._Y_local)
        marg_b = self._prior_b + (self._P * .5) * Y_T_woodbury_inv_Y
        #assert(numpy.allclose(marg_b, marg_b_2))

        likelihood -= (.5*self._N + self._prior_a)*numpy.log(marg_b)
        return(likelihood)

    def spike_slab(self, iters = 1000, nu0 = .00005, a_t = .01, b_t = 100., B=5, q=2, burnin=None):
        assert(B*q == self._D)
        if burnin == None:
            self.burnin = iters//2
        else:
            self.burnin = burnin
        blocks = numpy.array([tuple(range(i, i+q)) for i in xrange(0,self._D,q)])
        assert(len(blocks) == B)
        self.J = numpy.ones(self._D)
        self.tau = .5*numpy.ones(self._D)
        self.Gamma = self.J * self.tau * numpy.eye(self._D)
        Gamma_inv = 1./numpy.diag(self.Gamma) * numpy.eye(self._D)
        self.w = numpy.random.uniform()
        self.w_k = numpy.random.dirichlet([1,1], size=(self._D))
        burnin = iters//2
        XT_X = numpy.dot(self._X_local.T,self._X_local)
        OLS_local = numpy.linalg.inv(XT_X)
        OLS_local = numpy.dot(OLS_local, numpy.dot(self._X_local.T, self._Y_local))
        OLS_Y_hat = numpy.dot(self._X_local, OLS_local)
        error = self._Y_local - OLS_Y_hat
        sigma_hat = numpy.dot(error.T, error) / (self._N_p - self._D)
        self._Y_local *= numpy.sqrt(self._N_p/sigma_hat)
        self.ss_beta_trace = numpy.zeros((iters,self._D))
        self.ss_sigma_trace = numpy.zeros(iters)
        self.ss_Y_trace = numpy.zeros((iters, self._predict_N))

        sigma = numpy.copy(self._sigma)
        beta = numpy.copy(OLS_local)
        for it in xrange(iters):
            for j in blocks:
                assert(len(j) == q)
                not_j = [h for h in xrange(B*q) if h not in j]
                X_j = XT_X[j[0]:j[-1]+1, j[0]:j[-1]+1]
                X_not_j = XT_X[j[0]:j[-1]+1, not_j]
                beta_not_j = beta[not_j]
                var_j = numpy.linalg.inv(Gamma_inv[j[0]:j[-1]+1,j[0]:j[-1]+1] + (self._P/(sigma * self._N_p))*X_j)
                var_j_chol = numpy.linalg.cholesky( (self._P * var_j) /self._N_p )
                mu_j = numpy.dot(self._X_local[:,j].T, self._Y_local) - numpy.dot(X_not_j, beta_not_j)
                mu_j = self._P*numpy.dot(var_j, mu_j )/(sigma * self._N_p)
                beta[j] = numpy.dot(var_j_chol, numpy.random.normal(size=q)) + mu_j

            self.ss_beta_trace[it] = beta
            X_beta = numpy.dot(self._predict_X, beta)  / numpy.sqrt(self._N_p/sigma_hat)
#            self.ss_Y_trace[it,:] = numpy.random.normal(X_beta, numpy.sqrt((self._N_p * sigma) /self._P))
            self.ss_Y_trace[it,:] = numpy.random.normal(X_beta, numpy.sqrt(sigma/self._P))
            error = self._Y_local - numpy.dot(self._X_local, beta)
            error = numpy.dot(error.T,error)

            self.w_k[:,0] = numpy.log(1.-self.w) -.5*numpy.log(nu0) - (beta**2 / (2.* nu0 *self.tau))
            self.w_k[:,1] = numpy.log(self.w) - (beta**2 / (2.*self.tau))
            self.w_k -= numpy.tile(scipy.misc.logsumexp(self.w_k, axis=1),(2,1)).T
            bool_nu0 = self.w_k[:,0] > numpy.log(numpy.random.uniform(size=self._D))
            bool_1 = 1 - bool_nu0
            num_nu0 = bool_nu0.sum()
            num_1 = bool_1.sum()
            self.J[numpy.where(bool_nu0)] = nu0
            self.J[numpy.where(bool_1)] = 1

            tau_a = a_t + .5
            tau_b = 1./(b_t + (beta**2 / (2. * self.J)))
            self.tau = 1./numpy.random.gamma(tau_a, tau_b)

            w_a = 1. + num_1
            w_b = 1. + num_nu0
            self.w = numpy.random.beta(w_a,w_b)

            if self.diabetes:
                sigma_a = self._prior_a + .5*self._N
                sigma_b = 1./(self._prior_b + (self._P / (2.*self._N_p))*error)
                sigma = 1./numpy.random.gamma(sigma_a, sigma_b)
            self.ss_sigma_trace[it] = sigma

            self.Gamma = self.J * self.tau * numpy.eye(self._D)
            Gamma_inv = 1./numpy.diag(self.Gamma) * numpy.eye(self._D)

        self.ss_beta_trace = self.ss_beta_trace[burnin:] / numpy.sqrt(self._N_p/sigma_hat)
        self.ss_sigma_trace = self.ss_sigma_trace[burnin:]
        self.ss_post_beta = self.ss_beta_trace.mean(axis=0) #/ numpy.sqrt(self._N_p/sigma_hat)
        self.ss_post_sigma = self.ss_sigma_trace.mean(axis=0)
        self.ss_Y_trace = self.ss_Y_trace[burnin:]
        joint_trace_1 = numpy.hstack((self.ss_post_beta,self.ss_post_sigma))
        joint_trace_2 = numpy.vstack((self.ss_beta_trace.T,self.ss_sigma_trace)).T
        joint_trace_gather1 = numpy.array(self.comm.gather(joint_trace_1))
        joint_trace_gather2 = numpy.array(self.comm.gather(joint_trace_2))
        ss_Y_trace_gather = numpy.array(self.comm.gather(self.ss_Y_trace))
        self._Y_local /= numpy.sqrt(self._N_p/sigma_hat)

        if self.rank == 0:
            joint_median_1 = weiszfeld.geometric_median(joint_trace_gather1)
            joint_median_2 = weiszfeld.geometric_median(joint_trace_gather2)
            self.ss_Y_hat = numpy.median(ss_Y_trace_gather,axis=0)
            self.ss_median_beta, self.ss_median_sigma = joint_median_1[:-1], joint_median_1[-1]
            self.ss_median_beta_trace, self.ss_median_sigma_trace = joint_median_2[:,:-1], joint_median_2[:,-1]
            X_beta = numpy.dot(self._predict_X, self.ss_median_beta)
            X_beta2 = numpy.dot(self._predict_X, self.ss_median_beta_trace.mean(axis=0))
            self.ss_Y_hat2 = numpy.random.normal(numpy.dot(self._predict_X, self.ss_median_beta_trace.T) ,numpy.sqrt(self.ss_sigma_trace/self._P)).T

            RMSE = (X_beta - self._predict_Y)**2
            RMSE2 = (X_beta2 - self._predict_Y)**2
            self.ss_RMSE = numpy.sqrt(RMSE.mean())
            self.ss_RMSE2 = numpy.sqrt(RMSE2.mean())

        else:
            self.ss_Y_hat = None
            self.ss_median_beta = None
            self.ss_median_sigma = None
            self.ss_median_beta_trace = None
            self.ss_median_sigma_trace = None
            self.ss_Y_hat2 = None
            self.ss_RMSE = None
            self.ss_RMSE2 = None

        self.ss_Y_hat = self.comm.bcast(self.ss_Y_hat)
        self.ss_median_beta = self.comm.bcast(self.ss_median_beta)
        self.ss_median_sigma = self.comm.bcast(self.ss_median_sigma)
        self.ss_median_beta_trace = self.comm.bcast(self.ss_median_beta_trace)
        self.ss_median_sigma_trace = self.comm.bcast(self.ss_median_sigma_trace)
        self.ss_Y_hat2 = self.comm.bcast(self.ss_Y_hat2)
        self.ss_RMSE = self.comm.bcast(self.ss_RMSE)
        self.ss_RMSE2 = self.comm.bcast(self.ss_RMSE2)

        if self.rank == 0 and self.verbose:
            print("Spike and Slab RMSE: %f\tSpike and Slab RMSE2: %f" % (self.ss_RMSE,self.ss_RMSE2))

    def save_beta_trace_ss(self):
        if self.rank == 0:
            if self.diabetes:
                ss_1_name_Y = "ss_Y_trace" + self.fname_foot
                ss_1_name_Y2 = "ss_Y_trace2" + self.fname_foot
                ss_1_predict_Y = "ss_predict_Y" + self.fname_foot
                numpy.savetxt(ss_1_name_Y, self.ss_Y_hat, delimiter=",")
                numpy.savetxt(ss_1_name_Y2, self.ss_Y_hat2, delimiter=",")
                numpy.savetxt(ss_1_predict_Y, self._predict_Y, delimiter=",")
            else:
                ss_1_name = "ss_beta_trace" + self.fname_foot
                numpy.savetxt(ss_1_name, self.ss_median_beta_trace.T, delimiter=",")
                beta_fname = "true_beta" + self.fname_foot
                numpy.savetxt(beta_fname, self._beta.T, delimiter=",")

    def linear_model_sample(self, iters=1000, burnin=None):
        if burnin == None:
            self.burnin = iters//2
        else:
            self.burnin = burnin
        beta_trace = numpy.zeros((iters,self._K,self._D))
        sigma_trace = numpy.zeros((iters,self._K))
        Y_trace = numpy.empty((iters,self._K, self._predict_N))

        for k in xrange(self._K):
            X = self._X_model_local[k]
            OLS_beta = numpy.linalg.inv(numpy.dot(X.T,X))
            OLS_beta = numpy.dot(numpy.dot(OLS_beta, X.T),self._Y_local)
            X_OLS_beta = numpy.dot(X, OLS_beta)
            pred_X = self._predict_X[:,self._X_model_names[k]]
            (N,D) = X.shape # local values
            prior_beta = numpy.zeros(D)
            prior_sigma = numpy.eye(D)
            prior_sigma_inv = numpy.linalg.inv(prior_sigma)
            sigma = numpy.copy(self._sigma)
            OLS_sigma = numpy.mean((self._Y_local - X_OLS_beta)**2)
            OLS_log_likelihood = numpy.array([scipy.stats.multivariate_normal.logpdf(self._Y_local[n], mean=X_OLS_beta[n], cov=numpy.sqrt(OLS_sigma)) for n in xrange(N)]).sum()
            self.AIC[k] = (2.*D) - 2.*(OLS_log_likelihood*self._P)
            self.BIC[k] = -2. *(OLS_log_likelihood*self._P) + (D * numpy.log(self._N))

            if D > 1:
                post_var_inv = (prior_sigma_inv + self._P * numpy.dot(X.T, X))
                post_var = numpy.linalg.inv(post_var_inv)
                post_exp = numpy.dot(prior_sigma_inv, prior_beta) + self._P * numpy.dot(X.T, self._Y_local)
                post_exp = numpy.dot(post_var, post_exp)
            else:
                post_var_inv = (prior_sigma_inv + self._P * numpy.dot(X.T, X))
                post_var = 1./post_var_inv
                post_exp = (prior_sigma_inv * prior_beta) + self._P * numpy.dot(X.T, self._Y_local)
                post_exp *= post_var

            post_a = self._prior_a + .5 *(self._N+D)


            if self.diabetes:
                for it in xrange(iters):
                    if D > 1:
                        chol_var = numpy.linalg.cholesky(sigma*post_var)
                        beta = numpy.dot(numpy.random.normal(size=D), chol_var) + post_exp
                    else:
                        beta = numpy.random.normal(loc=post_exp, scale=numpy.sqrt(sigma*post_var), size=(1,))

                    beta_trace[it, k, self._X_model_names[k]] = beta
                    X_beta_k = numpy.dot(X, beta)
                    X_pred_beta_k = numpy.dot(pred_X, beta)
                    error = X_beta_k - self._Y_local
                    error = numpy.dot(error.T, error)
                    beta_diff = beta - prior_beta
                    beta_square = numpy.dot(numpy.dot(beta_diff.T, prior_sigma), beta_diff)
                    post_b = 1./(self._prior_b + .5*((self._P*error)+beta_square))
                    sigma = 1./numpy.random.gamma(post_a, post_b)
                    sigma_trace[it,k] = sigma
                    Y_trace[it,k,:] = numpy.random.normal(X_pred_beta_k, scale=numpy.sqrt(sigma / self._P))
            else:
                if D > 1:
                    chol_var = numpy.linalg.cholesky(sigma*post_var)
                    beta_trace[:,k,self._X_model_names[k]] = numpy.dot(numpy.random.normal(size=(iters,D)), chol_var) + numpy.tile(post_exp, (iters,1))
                else:
                    beta_trace[:,k,self._X_model_names[k]] = numpy.random.normal(loc=post_exp, scale=numpy.sqrt(sigma*post_var), size=(iters,1))
                sigma_trace[:,k] = sigma
                X_pred_beta_trace_k = numpy.dot(beta_trace[:,k,self._X_model_names[k]],pred_X.T)
                Y_trace[:,k,:] = numpy.random.normal(X_pred_beta_trace_k, scale=numpy.sqrt(sigma / self._P))

        self.Y_trace = Y_trace[self.burnin:]
        self.beta_trace = beta_trace[self.burnin:]
        self.sigma_trace = sigma_trace[self.burnin:]
        self.post_beta = self.beta_trace.mean(axis=0)
        self.post_sigma = self.sigma_trace.mean(axis=0)
        if self.verbose and self.rank == 0:
            print("Sampling Complete")

    def aic_bic(self):
        self.comm.Barrier()
        self.AIC_median = numpy.median(numpy.array(self.comm.allgather(self.AIC)),axis=0)
        self.BIC_median =numpy.median(numpy.array(self.comm.allgather(self.BIC)),axis=0)
        AIC_beta_trace = self.beta_trace[:,self.AIC.argmin(),:]
        BIC_beta_trace = self.beta_trace[:,self.BIC.argmin(),:]
        AIC_med_beta_trace = self.beta_trace[:,self.AIC_median.argmin(),:]
        BIC_med_beta_trace = self.beta_trace[:,self.BIC_median.argmin(),:]

        AIC_beta_trace_gather = numpy.array(self.comm.gather(AIC_beta_trace))
        BIC_beta_trace_gather = numpy.array(self.comm.gather(BIC_beta_trace))
        AIC_med_beta_trace_gather = numpy.array(self.comm.gather(AIC_med_beta_trace))
        BIC_med_beta_trace_gather = numpy.array(self.comm.gather(BIC_med_beta_trace))

        post_beta_gather = numpy.array(self.comm.gather(self.post_beta))
        Y_trace_gather =  numpy.array(self.comm.gather(self.Y_trace))
        AIC_Y_trace_gather = numpy.array(self.comm.gather(self.Y_trace[:,self.AIC.argmin(),:]))
        BIC_Y_trace_gather = numpy.array(self.comm.gather(self.Y_trace[:,self.BIC.argmin(),:]))
        post_AIC_beta = numpy.array(self.comm.gather(self.post_beta[self.AIC.argmin(),:]))
        post_BIC_beta = numpy.array(self.comm.gather(self.post_beta[self.BIC.argmin(),:]))

        if self.rank == 0:
            AIC_diff = numpy.dot(self._predict_X, weiszfeld.geometric_median(post_AIC_beta)) - self._predict_Y
            AIC_diff **= 2
            self.AIC_RMSE = numpy.sqrt(AIC_diff.mean())
            BIC_diff = numpy.dot(self._predict_X, weiszfeld.geometric_median(post_BIC_beta)) - self._predict_Y
            BIC_diff **= 2
            self.BIC_RMSE = numpy.sqrt(BIC_diff.mean())
            AIC_beta = weiszfeld.geometric_median(post_beta_gather[:,self.AIC_median.argmin(),:])
            BIC_beta = weiszfeld.geometric_median(post_beta_gather[:,self.BIC_median.argmin(),:])

            AIC_diff2 = numpy.dot(self._predict_X, AIC_beta) - self._predict_Y
            AIC_diff2 **= 2
            BIC_diff2 = numpy.dot(self._predict_X, BIC_beta) - self._predict_Y
            BIC_diff2 **= 2
            self.AIC_RMSE2 = numpy.sqrt(AIC_diff2.mean())
            self.BIC_RMSE2 = numpy.sqrt(BIC_diff2.mean())

            self.AIC_beta_1 = weiszfeld.geometric_median(AIC_beta_trace_gather)
            self.BIC_beta_1 = weiszfeld.geometric_median(BIC_beta_trace_gather)
            self.AIC_beta_2 = weiszfeld.geometric_median(AIC_med_beta_trace_gather)
            self.BIC_beta_2 = weiszfeld.geometric_median(BIC_med_beta_trace_gather)

#            self.AIC_Y_hat = weiszfeld.geometric_median(AIC_Y_trace_gather)
#            self.BIC_Y_hat = weiszfeld.geometric_median(BIC_Y_trace_gather)
#            self.AIC_Y_hat2 = weiszfeld.geometric_median(Y_trace_gather[:, :, self.AIC_median.argmin(),:])
#            self.BIC_Y_hat2 = weiszfeld.geometric_median(Y_trace_gather[:, :, self.BIC_median.argmin(),:])

            self.AIC_Y_hat = numpy.median(AIC_Y_trace_gather,axis=0)
            self.BIC_Y_hat = numpy.median(BIC_Y_trace_gather,axis=0)
            self.AIC_Y_hat2 = numpy.median(Y_trace_gather[:, :, self.AIC_median.argmin(),:],axis=0)
            self.BIC_Y_hat2 = numpy.median(Y_trace_gather[:, :, self.BIC_median.argmin(),:],axis=0)


        else:
            self.AIC_RMSE = None
            self.BIC_RMSE = None
            self.AIC_beta_1 = None
            self.BIC_beta_1 = None
            self.AIC_beta_2 = None
            self.BIC_beta_2 = None
            self.AIC_Y_hat = None
            self.BIC_Y_hat = None
            self.AIC_Y_hat2 = None
            self.BIC_Y_hat2 = None

        self.AIC_RMSE = self.comm.bcast(self.AIC_RMSE)
        self.BIC_RMSE = self.comm.bcast(self.BIC_RMSE)
        self.AIC_beta_1 = self.comm.bcast(self.AIC_beta_1)
        self.BIC_beta_1 = self.comm.bcast(self.BIC_beta_1)
        self.AIC_beta_2 = self.comm.bcast(self.AIC_beta_2)
        self.BIC_beta_2 = self.comm.bcast(self.BIC_beta_2)
        self.AIC_Y_hat = self.comm.bcast(self.AIC_Y_hat)
        self.BIC_Y_hat = self.comm.bcast(self.BIC_Y_hat)
        self.AIC_Y_hat2 = self.comm.bcast(self.AIC_Y_hat2)
        self.BIC_Y_hat2 = self.comm.bcast(self.BIC_Y_hat2)

        if self.rank == 0 and self.verbose:
            print('Min AIC: %.2f\t Optimal AIC Model: %s' % (self.AIC_median.min(), str(self._X_model_names[self.AIC_median.argmin()])))
            print('Min BIC: %.2f\t Optimal BIC Model: %s' % (self.BIC_median.min(), str(self._X_model_names[self.BIC_median.argmin()])))
            print('AIC RMSE: %.3f\tBIC RMSE: %.3f' % (self.AIC_RMSE,self.BIC_RMSE))
            print('AIC RMSE2: %.3f\tBIC RMSE2: %.3f' % (self.AIC_RMSE2,self.BIC_RMSE2))

    def bayes_ma(self, eps = .001): # model averaging script
        self.comm.Barrier()
        local_trunc_model_prob = numpy.copy(self.model_prob)
        local_trunc_model_prob[local_trunc_model_prob < eps] = 0 # set models of prob < .001 to zero
        local_trunc_model_prob /= local_trunc_model_prob.sum()
        gather_trunc_prob = numpy.array(self.comm.allreduce(local_trunc_model_prob))
        nnz_gather_trunc_prob = numpy.where(gather_trunc_prob > 0)[0]
        iter_n, K,D = self.beta_trace.shape
        joint_posterior = numpy.vstack((self.post_beta[nnz_gather_trunc_prob,:].T, self.post_sigma[nnz_gather_trunc_prob].T, local_trunc_model_prob[nnz_gather_trunc_prob])).T
        joint_posterior_gather = numpy.array(self.comm.gather(joint_posterior))
        trace_joint = numpy.dstack((self.beta_trace, self.sigma_trace, numpy.tile(local_trunc_model_prob, (iter_n,1))))
        trace_joint = trace_joint[:,nnz_gather_trunc_prob,:]
        iter_n, K,D_joint = trace_joint.shape
        trace_joint = trace_joint.reshape(iter_n, K*D_joint)
        local_Y_trace = numpy.dot(local_trunc_model_prob[nnz_gather_trunc_prob], self.Y_trace[:,nnz_gather_trunc_prob,:])
        trunc_y_trace_gather = numpy.array(self.comm.gather(local_Y_trace))
        trace_joint_gather = numpy.array(self.comm.gather(trace_joint))

        if self.rank == 0:
            med_joint = weiszfeld.geometric_median(joint_posterior_gather)
            self.med_marg_model_prob, self.med_marg_sigma, self.med_marg_beta = med_joint[:,-1],med_joint[:,-2],med_joint[:,:-2]
            marg_X_beta = numpy.dot(self.med_marg_model_prob, numpy.dot(self.med_marg_beta, self._predict_X.T))
            med_trace_joint = weiszfeld.geometric_median(trace_joint_gather).reshape(iter_n, K,D_joint)

            self.marg_Y_hat = numpy.median(trunc_y_trace_gather, axis=0)
            self.med_marg_model_prob2, self.med_marg_sigma_trace, self.med_marg_beta_trace = med_trace_joint[0,:,-1], med_trace_joint[:,:,-2], med_trace_joint[:,:,:-2]
            self.med_marg_model_prob2[self.med_marg_model_prob2 < eps] = 0
            self.med_marg_model_prob2 /= self.med_marg_model_prob2.sum()
            nnz_med_marg_model_prob2 = numpy.where(self.med_marg_model_prob2 > 0)[0]

            XB_marg2 = numpy.dot(self.med_marg_beta_trace[:,nnz_med_marg_model_prob2,:],self._predict_X.T)
            marg_X_beta_2 = numpy.dot(self.med_marg_model_prob2[nnz_med_marg_model_prob2],numpy.dot(self.med_marg_beta_trace[:,nnz_med_marg_model_prob2,:].mean(axis=0), self._predict_X.T))
            tile_sigma = numpy.tile(self.med_marg_sigma_trace[:,nnz_med_marg_model_prob2], (self._predict_N,1,1)).reshape(XB_marg2.shape)
            self.marg_Y_hat2 = numpy.dot(self.med_marg_model_prob2[nnz_med_marg_model_prob2],numpy.random.normal(XB_marg2,numpy.sqrt(tile_sigma / self._P)).T)

            RMSE = (self._predict_Y - marg_X_beta)**2
            RMSE2 = (self._predict_Y - marg_X_beta_2)**2
            self.bma_RMSE = numpy.sqrt(RMSE.mean()) # joint beta/model prob RMSE
            self.bma_RMSE2 = numpy.sqrt(RMSE2.mean()) # joint beta/model prob RMSE
        else:
            self.med_marg_model_prob = None
            self.med_marg_sigma = None
            self.med_marg_beta = None
            self.marg_Y_hat = None
            self.med_marg_model_prob2 = None
            self.med_marg_sigma_trace = None
            self.med_marg_beta_trace = None
            self.marg_Y_hat2 = None
            self.bma_RMSE = None
            self.bma_RMSE2 = None

        self.med_marg_model_prob = self.comm.bcast(self.med_marg_model_prob)
        self.med_marg_sigma = self.comm.bcast(self.med_marg_sigma)
        self.med_marg_beta = self.comm.bcast(self.med_marg_beta)
        self.marg_Y_hat = self.comm.bcast(self.marg_Y_hat)
        self.med_marg_model_prob2 = self.comm.bcast(self.med_marg_model_prob2)
        self.med_marg_sigma_trace = self.comm.bcast(self.med_marg_sigma_trace)
        self.med_marg_beta_trace = self.comm.bcast(self.med_marg_beta_trace)
        self.marg_Y_hat2 = self.comm.bcast(self.marg_Y_hat2)
        self.bma_RMSE = self.comm.bcast(self.bma_RMSE)
        self.bma_RMSE2 = self.comm.bcast(self.bma_RMSE2)

        if self.comm.rank == 0 and self.verbose:
            print('Model Averaging RMSE: %.3f\tModel Averaging RMSE2: %.3f' % (self.bma_RMSE, self.bma_RMSE2))

    def median_selection(self): # model averaging script
        self.comm.Barrier()
        (K,D) = self.post_beta.shape
        predictor_prob = numpy.zeros(D)
        for d in xrange(D):
            models_p = self.post_beta[:,d].nonzero()
            predictor_prob[d] = self.model_prob[models_p].sum()
        predictor_select = numpy.where(predictor_prob >= .5)
        model_select = tuple(predictor_select[0])
        optimal_model = [v for v,k in self._X_model_names.items() if k==model_select]
        if optimal_model:
            optimal_model = optimal_model[0]
            local_beta = self.post_beta[optimal_model]
            local_sigma = self.post_sigma[optimal_model]
            local_Y = self.Y_trace[:,optimal_model,:]
            local_beta_trace = self.beta_trace[:,optimal_model,:]
            local_sigma_trace = self.sigma_trace[:, optimal_model]
        else:
            local_beta = numpy.random.normal(size=D)
            local_Y = numpy.random.normal(scale = numpy.sqrt(self._sigma / self._P), size=self.Y_trace.shape[0])
            local_beta_trace = numpy.random.normal(size=(self.burnin,D))
            if self.diabetes:
                post_a = self._prior_a+ .5*(self._N+D)
                post_b = 1./(self._prior_b + .5* numpy.diag((self._P*numpy.dot(local_Y, local_Y.T))+numpy.dot(local_beta_trace, local_beta_trace.T)))
                local_sigma_trace = numpy.random.gamma(post_a, post_b)
                local_sigma = local_sigma_trace.mean()
            else:
                local_sigma_trace = self._sigma * numpy.ones(self.burnin)
                local_sigma = self._sigma

        med_predictor_prob = numpy.median(numpy.array(self.comm.allgather(predictor_prob)),axis=0)
        med_predictor_select = numpy.where(med_predictor_prob >= .5)
        med_model_select = tuple(med_predictor_select[0])
        med_optimal_model = [v for v,k in self._X_model_names.items() if k==med_model_select]
        if med_optimal_model:
            med_optimal_model = med_optimal_model[0]
            med_local_beta = self.post_beta[med_optimal_model]
            med_local_sigma = self.post_sigma[med_optimal_model]
            med_local_Y = self.Y_trace[:,med_optimal_model,:]
            med_local_beta_trace = self.beta_trace[:,med_optimal_model,:]
            med_local_sigma_trace = self.sigma_trace[:, med_optimal_model]
        else:
            med_local_beta = numpy.random.normal(size=D)
            med_local_Y = numpy.random.normal(scale = numpy.sqrt(self._sigma / self._P), size=self.Y_trace.shape[0])
            med_local_beta_trace = numpy.random.normal(size=(self.burnin,D))
            if self.diabetes:
                post_a = self._prior_a+ .5*(self._N+D)
                post_b = 1./(self._prior_b + .5* numpy.diag((self._P*numpy.dot(med_local_Y, med_local_Y.T))+numpy.dot(med_local_beta_trace, med_local_beta_trace.T)))
                med_local_sigma_trace = numpy.random.gamma(post_a, post_b)
                med_local_sigma = med_local_sigma_trace.mean()
            else:
                med_local_sigma_trace = self._sigma * numpy.ones(self.burnin)
                med_local_sigma = self._sigma
        assert(local_beta_trace.shape == med_local_beta_trace.shape)
        assert(local_Y.shape == med_local_Y.shape)
        assert(local_sigma_trace.shape == med_local_sigma_trace.shape)
        n_iter, K, D = self.beta_trace.shape
        joint_trace1 = numpy.vstack((local_beta_trace.T,local_sigma_trace)).T
        joint_trace2 = numpy.vstack((med_local_beta_trace.T,med_local_sigma_trace)).T
        joint_post1 = numpy.hstack((local_beta, local_sigma))
        joint_post2 = numpy.hstack((med_local_beta, med_local_sigma))
        joint_post_gather1 = numpy.array(self.comm.gather(joint_post1))
        joint_post_gather2 = numpy.array(self.comm.gather(joint_post2))
        joint_trace_gather1 = numpy.array(self.comm.gather(joint_trace1))
        joint_trace_gather2 = numpy.array(self.comm.gather(joint_trace2))
        local_Y_gather = numpy.array(self.comm.gather(local_Y))
        med_local_Y_gather = numpy.array(self.comm.gather(med_local_Y))

        if self.rank == 0:
            joint_med_1 = weiszfeld.geometric_median(joint_post_gather1)
            joint_med_2 = weiszfeld.geometric_median(joint_post_gather2)
            self.median_beta, self.median_sigma = joint_med_1[:-1],joint_med_1[-1]
            self.median_beta2, self.median_sigma2 = joint_med_2[:-1],joint_med_2[-1]
            joint_trace_med_1 = weiszfeld.geometric_median(joint_trace_gather1)
            joint_trace_med_2 = weiszfeld.geometric_median(joint_trace_gather2)
            self.median_beta_trace, self.med_model_sigma_trace = joint_trace_med_1[:,:-1], joint_trace_med_1[:,-1]
            self.median_beta_trace2, self.med_model_sigma_trace2 = joint_trace_med_2[:,:-1], joint_trace_med_2[:,-1]
            self.med_Y_hat = numpy.median(local_Y_gather,axis=0)
            self.med_Y_hat2 = numpy.median(med_local_Y_gather,axis=0)
            X_beta = numpy.dot(self._predict_X, self.median_beta)
            X_beta2 = numpy.dot(self._predict_X, self.median_beta2)
            RMSE = (self._predict_Y - X_beta)**2
            RMSE2 = (self._predict_Y - X_beta2)**2
            self.med_RMSE = numpy.sqrt(RMSE.mean())
            self.med_RMSE2 = numpy.sqrt(RMSE2.mean())
        else:
            self.median_beta = None
            self.median_sigma = None
            self.median_beta2 = None
            self.median_sigma2 = None
            self.median_beta_trace = None
            self.med_model_sigma_trace = None
            self.median_beta_trace2 = None
            self.med_model_sigma_trace2 = None
            self.med_Y_hat = None
            self.med_Y_hat2 = None
            self.med_RMSE = None
            self.med_RMSE2 = None

        self.median_beta = self.comm.bcast(self.median_beta)
        self.median_sigma = self.comm.bcast(self.median_sigma)
        self.median_beta2 = self.comm.bcast(self.median_beta2)
        self.median_sigma2 = self.comm.bcast(self.median_sigma2)
        self.median_beta_trace = self.comm.bcast(self.median_beta_trace)
        self.med_model_sigma_trace = self.comm.bcast(self.med_model_sigma_trace)
        self.median_beta_trace2 = self.comm.bcast(self.median_beta_trace2)
        self.med_model_sigma_trace2 = self.comm.bcast(self.med_model_sigma_trace2)
        self.med_Y_hat = self.comm.bcast(self.med_Y_hat)
        self.med_Y_hat2 = self.comm.bcast(self.med_Y_hat2)
        self.med_RMSE = self.comm.bcast(self.med_RMSE)
        self.med_RMSE2 = self.comm.bcast(self.med_RMSE2)

        if self.comm.rank == 0 and self.verbose:
            print('Median Model RMSE: %.3f\tMedian Model RMSE2: %.3f' % (self.med_RMSE, self.med_RMSE2))
            print("Optimal Median Model: %s" % str(self._X_model_names[med_optimal_model]))


    def save_beta_trace(self):
        if self.rank == 0:
            if self.diabetes:
                med_1_Y = "median_Y_1" + self.fname_foot
                numpy.savetxt(med_1_Y, self.med_Y_hat.T, delimiter=",")
                med_2_Y = "median_Y_2" + self.fname_foot
                numpy.savetxt(med_2_Y, self.med_Y_hat2.T, delimiter=",")
                AIC_1_Y = "AIC_Y_1" + self.fname_foot
                numpy.savetxt(AIC_1_Y, self.AIC_Y_hat.T, delimiter=",")
                AIC_2_Y = "AIC_Y_2" + self.fname_foot
                numpy.savetxt(AIC_2_Y, self.AIC_Y_hat2.T, delimiter=",")
                BIC_1_Y = "BIC_Y_1" + self.fname_foot
                numpy.savetxt(BIC_1_Y, self.BIC_Y_hat.T, delimiter=",")
                BIC_2_Y = "BIC_Y_2" + self.fname_foot
                numpy.savetxt(BIC_2_Y, self.BIC_Y_hat2.T, delimiter=",")
                bma_1_Y = "BMA_Y_1" + self.fname_foot
                numpy.savetxt(bma_1_Y, self.marg_Y_hat.T, delimiter=",")
                bma_2_Y = "BMA_Y_2" + self.fname_foot
                numpy.savetxt(bma_2_Y, self.marg_Y_hat2.T, delimiter=",")
                Y_trace = "Y_trace"  + self.fname_foot
                numpy.savetxt(Y_trace, self._predict_Y, delimiter=",")

            else:
                med_1_name = "median_beta_trace1" + self.fname_foot
                numpy.savetxt(med_1_name, self.median_beta_trace.T, delimiter=",")
                med_2_name = "median_beta_trace2" + self.fname_foot
                numpy.savetxt(med_2_name, self.median_beta_trace2.T, delimiter=",")
                file_AIC_beta_1 = "AIC_beta_1" + self.fname_foot
                numpy.savetxt(file_AIC_beta_1, self.AIC_beta_1.T, delimiter=",")
                file_AIC_beta_2 = "AIC_beta_2" + self.fname_foot
                numpy.savetxt(file_AIC_beta_2, self.AIC_beta_2.T, delimiter=",")
                file_BIC_beta_1 = "BIC_beta_1" + self.fname_foot
                numpy.savetxt(file_BIC_beta_1, self.BIC_beta_1.T, delimiter=",")
                file_BIC_beta_2 = "BIC_beta_2" + self.fname_foot
                numpy.savetxt(file_BIC_beta_2, self.BIC_beta_2.T, delimiter=",")
#                beta_fname = "true_beta" + self.fname_foot
#                numpy.savetxt(beta_fname, self._beta.T, delimiter=",")


    def contamination_test(self, trials=10, contam_num = 50, by=10, file_output="contamination_output",start=0):
        file_output += self.fname_foot
        self.contam_data = []
        for c in xrange(start,contam_num+1, by):
            if self.rank  == 0:
                print("Contamination Number: %i" % c)
            self.__init__(outlier_num=c,model_probs=True,outlier_val =10000)
            for t in xrange(trials):
                self.linear_model_sample()
                self.aic_bic()
                self.bayes_ma()
                self.median_selection()
                if self.rank == 0:
                    self.contam_data.append([self._P, self._outlier_num, self.med_RMSE, self.med_RMSE2, self.bma_RMSE, self.bma_RMSE2, self.AIC_RMSE, self.AIC_RMSE2, self.BIC_RMSE, self.BIC_RMSE2])
                    self._predict_X = numpy.random.normal(size=(self._predict_N,self._D))
                    self._predict_Y = numpy.dot(self._predict_X, self._beta)
                else:
                    self._predict_X = None
                    self._predict_Y = None
                self._predict_X = self.comm.bcast(self._predict_X)
                self._predict_Y = self.comm.bcast(self._predict_Y)


        if self.rank == 0:
            self.contam_data = numpy.array(self.contam_data)
            file_header = "[self._P, self._outlier_num, self.med_RMSE, self.med_RMSE2, self.bma_RMSE, self.bma_RMSE2, self.AIC_RMSE, self.AIC_RMSE2, self.BIC_RMSE, self.BIC_RMSE2]"
            numpy.savetxt(file_output, self.contam_data, delimiter=",",header=file_header)

    def contamination_ss(self, trials=10, contam_num = 50, by=10, file_output="contamination_output_ss"):
        file_output += self.fname_foot
        self.contam_data_ss = []
        for c in xrange(0,contam_num+1, by):
            if self.rank  == 0:
                print("Contamination Number: %i" % c)
            for t in xrange(trials):
                self.__init__(outlier_num=c, ss=True,outlier_val =10000)
                if self.rank == 0:
                    self._predict_X = numpy.random.normal(size=(self._predict_N,self._D))
                    self._predict_Y = numpy.dot(self._predict_X, self._beta)
                else:
                    self._predict_X = None
                    self._predict_Y = None
                self._predict_X = self.comm.bcast(self._predict_X)
                self._predict_Y = self.comm.bcast(self._predict_Y)
                self.spike_slab()
                if self.rank==0:
                    self.contam_data_ss.append([self._P, c, self.ss_RMSE,self.ss_RMSE2])


        if self.rank == 0:
            self.contam_data_ss = numpy.array(self.contam_data_ss)
            file_header = "[self._P, c, self.ss_RMSE,self.ss_RMSE2]"
            numpy.savetxt(file_output, self.contam_data_ss, delimiter=",",header=file_header)

    def outlier_magnitude(self, trials=20, min_outlier = 0, max_outlier = 10001, by = 500,file_output="outlier_magnitude"):
        file_output += self.fname_foot
        self.magnitude_data = []
        for out in xrange(min_outlier, max_outlier, by):
            if self.rank  == 0:
                print("Outlier Value: %i" % out)
            self.__init__(outlier_val = out,model_probs=True, outlier_num=1)
            for t in xrange(trials):
                self.linear_model_sample()
                self.aic_bic()
                self.bayes_ma()
                self.median_selection()
                if self.rank == 0:
                    self.magnitude_data.append([self._P, self.outlier_val, self.med_RMSE, self.med_RMSE2, self.bma_RMSE, self.bma_RMSE2, self.AIC_RMSE, self.AIC_RMSE2, self.BIC_RMSE, self.BIC_RMSE2])
                    self._predict_X = numpy.random.normal(size=(self._predict_N,self._D))
                    self._predict_Y = numpy.dot(self._predict_X, self._beta)
                else:
                    self._predict_X = None
                    self._predict_Y = None
                self._predict_X = self.comm.bcast(self._predict_X)
                self._predict_Y = self.comm.bcast(self._predict_Y)


        if self.rank == 0:
            self.magnitude_data = numpy.array(self.magnitude_data)
            file_header = "[self._P, self.outlier_val, self.med_RMSE, self.med_RMSE2, self.bma_RMSE, self.bma_RMSE2, self.AIC_RMSE, self.AIC_RMSE2, self.BIC_RMSE, self.BIC_RMSE2, self.bma_RMSE3]"
            numpy.savetxt(file_output, self.magnitude_data, delimiter=",",header=file_header)

    def outlier_magnitude_ss(self, trials=20, min_outlier = 0, max_outlier = 10001, by = 500, file_output="outlier_magnitude_ss"):
        file_output += self.fname_foot
        self.magnitude_data_ss = []
        for out in xrange(min_outlier, max_outlier, by):
            if self.rank  == 0:
                print("Outlier Value: %i" % out)
            for t in xrange(trials):
                self.__init__(outlier_val = out, ss=True, outlier_num=1)
                if self.rank == 0:
                    self._predict_X = numpy.random.normal(size=(self._predict_N,self._D))
                    self._predict_Y = numpy.dot(self._predict_X, self._beta)
                else:
                    self._predict_X = None
                    self._predict_Y = None
                self._predict_X = self.comm.bcast(self._predict_X)
                self._predict_Y = self.comm.bcast(self._predict_Y)
                self.spike_slab()
                if self.rank==0:
                    self.magnitude_data_ss.append([self._P, out, self.ss_RMSE,self.ss_RMSE2])


        if self.rank == 0:
            self.magnitude_data_ss = numpy.array(self.magnitude_data_ss)
            file_header = "[self._P, out, self.ss_RMSE, self.ss_RMSE2]"
            numpy.savetxt(file_output, self.magnitude_data_ss, delimiter=",",header=file_header)

    def coverage_test(self, outlier_val=1., n_tests = 20):
        self.__init__(outlier_val = outlier_val, predict_N=1, model_probs=True,outlier_num=1)

        if self.rank==0:
            self.bma_coverage = numpy.empty((n_tests))
            self.med_coverage = numpy.empty((n_tests))
            self.AIC_coverage = numpy.empty((n_tests))
            self.BIC_coverage = numpy.empty((n_tests))
            self.bma_coverage2 = numpy.empty((n_tests))
            self.med_coverage2 = numpy.empty((n_tests))
            self.AIC_coverage2 = numpy.empty((n_tests))
            self.BIC_coverage2 = numpy.empty((n_tests))

        for nt in xrange(n_tests):
            self.linear_model_sample()
            true_Y = self._predict_Y[0]
            self.aic_bic()
            self.bayes_ma()
            self.median_selection()

            if self.rank == 0:
                print("True Y: %.2f" % true_Y)
                AIC_CI = numpy.percentile(self.AIC_Y_hat, (2.5,97.5))
                self.AIC_coverage[nt] =  (AIC_CI[0] <= true_Y) & (true_Y <= AIC_CI[1])
                AIC_CI2 = numpy.percentile(self.AIC_Y_hat2, (2.5,97.5))
                self.AIC_coverage2[nt] =  (AIC_CI2[0] <= true_Y) & (true_Y <= AIC_CI2[1])
                print("AIC CI: %s\nAIC2 CI: %s " % (str(AIC_CI), str(AIC_CI2)))
                BIC_CI = numpy.percentile(self.BIC_Y_hat, (2.5,97.5))
                self.BIC_coverage[nt] =  (BIC_CI[0] <= true_Y) & (true_Y <= BIC_CI[1])
                BIC_CI2 = numpy.percentile(self.BIC_Y_hat2, (2.5,97.5))
                self.BIC_coverage2[nt] =  (BIC_CI2[0] <= true_Y) & (true_Y <= BIC_CI2[1])
                print("BIC CI: %s\nBIC2 CI: %s " % (str(BIC_CI), str(BIC_CI2)))
                bma_CI = numpy.percentile(self.marg_Y_hat, (2.5,97.5))
                self.bma_coverage[nt] = (bma_CI[0] <= true_Y) & (true_Y <= bma_CI[1])
                bma_CI2 = numpy.percentile(self.marg_Y_hat2, (2.5,97.5))
                self.bma_coverage2[nt] = (bma_CI2[0] <= true_Y) & (true_Y <= bma_CI2[1])
                print("BMA CI: %s\nBMA2 CI: %s " % (str(bma_CI), str(bma_CI2)))
                med_CI = numpy.percentile(self.med_Y_hat, (2.5,97.5))
                self.med_coverage[nt] = (med_CI[0] <= true_Y) & (true_Y <= med_CI[1])
                med_CI2 = numpy.percentile(self.med_Y_hat2, (2.5,97.5))
                self.med_coverage2[nt] = (med_CI2[0] <= true_Y) & (true_Y <= med_CI2[1])
                print("Med CI: %s\nMed 2 CI: %s " % (str(med_CI), str(med_CI2)))
                self._predict_X = numpy.random.normal(size=(self._predict_N,self._D))
                self._predict_Y = numpy.dot(self._predict_X, self._beta)
            else:
                self._predict_X = None
                self._predict_Y = None

            self._predict_X = self.comm.bcast(self._predict_X)
            self._predict_Y = self.comm.bcast(self._predict_Y)

#                self.bma_coverage_prop = self.bma_coverage.mean()
#                self.med_coverage_prop = self.med_coverage.mean()
#                self.AIC_prop = self.AIC_coverage.mean()
#                self.BIC_prop = self.BIC_coverage.mean()
#                self.bma_coverage_prop2 = self.bma_coverage2.mean()
#                self.med_coverage_prop2 = self.med_coverage2.mean()
#                self.AIC_prop2 = self.AIC_coverage2.mean()
#                self.BIC_prop2 = self.BIC_coverage2.mean()

        if self.rank==0:
            self.bma_coverage_prop = self.bma_coverage.mean()
            self.med_coverage_prop = self.med_coverage.mean()
            self.AIC_prop = self.AIC_coverage.mean()
            self.BIC_prop = self.BIC_coverage.mean()
            self.bma_coverage_prop2 = self.bma_coverage2.mean()
            self.med_coverage_prop2 = self.med_coverage2.mean()
            self.AIC_prop2 = self.AIC_coverage2.mean()
            self.BIC_prop2 = self.BIC_coverage2.mean()
            print('Subsets: %i\t Outlier Number: %i\t Outlier Magnitude: %.1f' % (self._P, self._outlier_num, self.outlier_val))
            print('BMA Coverage: %.2f\t Median Model Coverage: %.2f' % (self.bma_coverage_prop,self.med_coverage_prop))
            print('AIC Coverage: %.2f\t BIC Coverage: %.2f' %  (self.AIC_prop, self.BIC_prop))
            print('BMA Coverage: %.2f\t Median Model Coverage: %.2f' % (self.bma_coverage_prop2,self.med_coverage_prop2))
            print('AIC Coverage: %.2f\t BIC Coverage: %.2f' %  (self.AIC_prop2, self.BIC_prop2))

    def outlier_coverage_test(self, min_outlier = 0, max_outlier = 7001, by = 500, file_output="outlier_coverage", n_tests=20):
        file_output += self.fname_foot
        self.outlier_coverage = []
        for out in xrange(min_outlier, max_outlier, by):
            self.coverage_test(n_tests=n_tests,outlier_val = out)
            if self.rank == 0:
                print("Test Outlier Magnitude: %.1f" % out)
                self.outlier_coverage.append([self._P, out, self.bma_coverage_prop,self.bma_coverage_prop2, self.med_coverage_prop, self.med_coverage_prop2,self.AIC_prop,self.AIC_prop2,self.BIC_prop,self.BIC_prop2])

        if self.rank == 0:
            self.outlier_coverage = numpy.array(self.outlier_coverage)
            file_header = '[self._P, out, self.bma_coverage_prop,self.bma_coverage_prop2, self.med_coverage_prop, self.med_coverage_prop2,self.AIC_prop,self.AIC_prop2,self.BIC_prop,self.BIC_prop2,self.bma_coverage_prop3]'
            numpy.savetxt(file_output, self.outlier_coverage, delimiter=",",header=file_header)

    def outlier_coverage_ss(self, min_outlier = 0, max_outlier = 10001, by = 500, file_output="outlier_coverage_ss", n_tests=20):
        file_output +=self.fname_foot
        self.outlier_coverage_ss = []
        self.ss_coverage = numpy.empty(n_tests)
        self.ss_coverage2 = numpy.empty(n_tests)
        for out in xrange(min_outlier, max_outlier, by):
            for t in xrange(n_tests):
                self.__init__(outlier_val = out, predict_N = 1, ss=True, outlier_num=1)
                if self.rank == 0:
                    self._predict_X = numpy.random.normal(size=(self._predict_N,self._D))
                    self._predict_Y = numpy.dot(self._predict_X, self._beta)
                else:
                    self._predict_X = None
                    self._predict_Y = None
                self._predict_X = self.comm.bcast(self._predict_X)
                self._predict_Y = self.comm.bcast(self._predict_Y)
                self.spike_slab()

                true_Y = self._predict_Y[0]
                ss_CI = numpy.percentile(self.ss_Y_hat, (2.5, 97.5), axis=0)
                ss_bool = (ss_CI[0] <= true_Y) & (true_Y <= ss_CI[1])
                self.ss_coverage[t] = ss_bool
                ss_CI2 = numpy.percentile(self.ss_Y_hat2, (2.5, 97.5), axis=0)
                ss_bool2 = (ss_CI2[0] <= true_Y) & (true_Y <= ss_CI2[1])

                self.ss_coverage2[t] = ss_bool2

                if self.rank == 0:
                    print("True Y: %.2f" % true_Y)
                    print("SS CI: %s\nSS CI2: %s" % (str(ss_CI),str(ss_CI2)))
#                    print("SS CI2: %s" % str(ss_CI2))

            self.ss_coverage_prop = self.ss_coverage.sum() / n_tests
            self.ss_coverage_prop2 = self.ss_coverage2.sum() / n_tests
            if self.rank == 0:
                print("Test Outlier Magnitude: %.1f\tSS Coverage %.2f\tSS Coverage 2 %.2f" % (out,self.ss_coverage_prop, self.ss_coverage_prop2))
                self.outlier_coverage_ss.append([self._P, out, self.ss_coverage_prop, self.ss_coverage_prop2])

        if self.rank == 0:
            self.outlier_coverage_ss = numpy.array(self.outlier_coverage_ss)
            file_header = '[self._P, out, self.ss_coverage_prop, self.ss_coverage_prop2]'
            numpy.savetxt(file_output, self.outlier_coverage_ss, delimiter=",", header=file_header)

if __name__ == "__main__":

#    bma = BMA(diabetes=True)
#    bma = BMA(verbose=False, predict_N = 100)
#    bma = BMA(model_probs=False)
#    bma.coverage_test(outlier_val = 1000)
#    bma.__init__(N=100, D=3, num_predict = 1, predict_N = 100)
#    bma = BMA(D=3, num_predict=2, predict_N=100)
#    bma = BMA(verbose=False)
    bma = BMA()
    bma.contamination_test()
    bma.outlier_magnitude()
#    bma.outlier_coverage_test()

 #    for it in xrange(500):
#    bma.coverage_test(n_tests = 50, outlier_val=100.)
#    ss = BMA(ss=True,outlier_val=100000,outlier_num=1,predict_N=100)
#    ss = BMA(ss=True, model_probs=False)
    #ss.contamination_ss()
#    ss.outlier_magnitude_ss()
#    ss.outlier_coverage_ss()
#    for t in xrange(10):
#    ss.spike_slab()
#    ss.save_beta_trace_ss()
#    plt.plot(ss.ss_post_beta,'o')
#    plt.errorbar(xrange(45), ss._predict_Y, yerr=numpy.percentile(ss.ss_Y_hat2.T, (2.5, 97.5), axis=0), lw=1, ls="None", marker='o')
#    plt.errorbar(xrange(45), ss._predict_Y, yerr=numpy.percentile(ss.ss_Y_hat, (2.5, 97.5), axis=0), lw=1, ls="None", marker='o')
#    plt.xlabel("Test Set Data")
#    plt.xlim([-1,46])
#    plt.ylabel("Test Set Response")
#    plt.plot(ss._beta,'^')
#    ss.save_beta_trace_ss()
#    print(bma.model_prob)
#    gather_model = numpy.array(bma.comm.allgather(bma.model_prob))
#    median_prob = weiszfeld.weiszfeld.geometric_median(gather_model, axis=0)
#    median_prob /= median_prob.sum()
#    if bma.rank==0:
#        print(median_prob)
#    bma = BMA(outlier_val=100000,outlier_num=1,predict_N=100)
    # comm = MPI.COMM_WORLD
    # for t in xrange(20):
        # start_time = time.time()
        # bma = BMA(verbose=False) 
        # bma.linear_model_sample()
        # bma.aic_bic()
        # bma.bayes_ma()
        # bma.median_selection()
        # end_time = time.time() - start_time
        # if comm.rank == 0:
            # print(end_time)
#    bma.save_beta_trace()

#    proportions = []
#    for x in xrange(50):
#        bma.coverage_test(n_tests=50)
#        proportions.append([bma.bma_coverage_prop,bma.med_coverage_prop,bma.AIC_prop, bma.BIC_prop])
#    bma.regression_coverage()
#plt.plot(xrange(45),numpy.percentile(ss.ss_Y_hat, (2.5),axis=0) - ss._predict_Y,'b')
#plt.plot(xrange(45),numpy.percentile(ss.ss_Y_hat, (97.5),axis=0) - ss._predict_Y,'b')
#plt.plot(xrange(45),numpy.mean(ss.ss_Y_hat,axis=0) - ss._predict_Y,'bo')
#plt.plot(xrange(45), numpy.zeros(45), 'r')