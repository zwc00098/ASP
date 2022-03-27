# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10

@author: jaehyuk
"""

import numpy as np
import scipy.stats as ss
import scipy.optimize as sopt
import pyfeng as pf
import scipy.integrate as spint

'''
MC model class for Beta=1
'''
class ModelBsmMC:
    beta = 1.0   # fixed (not used)
    vov, rho = 0.0, 0.0
    sigma, intr, divr = None, None, None
    bsm_model = None
    '''
    You may define more members for MC: time step, etc
    '''
    
    def __init__(self, sigma, vov=0, rho=0.0, beta=1.0, intr=0, divr=0):
        self.sigma = sigma
        self.vov = vov
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.bsm_model = pf.Bsm(sigma, intr=intr, divr=divr)
        
    def bsm_vol(self, strike, spot, texp=None, sigma=None):
        ''''
        From the price from self.price() compute the implied vol
        Use self.bsm_model.impvol() method
        '''
        p = self.price(strike, spot, texp).mean(axis=0)
        return self.bsm_model.impvol(p, strike, spot, texp)
    
    def price(self, strike, spot, texp=None, sigma=None, cp=1):
        '''
        Your MC routine goes here
        Generate paths for vol and price first. Then get prices (vector) for all strikes
        You may fix the random number seed
        '''
        m = pf.BsmNdMc(self.sigma, rn_seed = 12345)
        m.simulate(tobs = [texp], n_path = 10000)
        payoff = lambda x: np.fmax(np.mean(x, axis=1) - strike, 0)
        price = []
        for strike in strike:
            price.append(m.price_european(spot, texp, payoff))
        return np.array(price)

'''
MC model class for Beta=0
'''
class ModelNormalMC:
    beta = 0.0   # fixed (not used)
    vov, rho = 0.0, 0.0
    sigma, intr, divr = None, None, None
    normal_model = None
    
    def __init__(self, sigma, vov=0, rho=0.0, beta=0.0, intr=0, divr=0):
        self.sigma = sigma
        self.vov = vov
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.normal_model = pf.Norm(sigma, intr=intr, divr=divr)
        
    def norm_vol(self, strike, spot, texp=None, sigma=None):
        ''''
        From the price from self.price() compute the implied vol.
        Use self.normal_model.impvol() method        
        '''
        p = self.price(strike, spot, texp).mean(axis=0)
        return self.normal_model.impvol(p, stike, spot, texp)
        
    def price(self, strike, spot, texp=None, sigma=None, cp=1):
        '''
        Your MC routine goes here
        Generate paths for vol and price first. Then get prices (vector) for all strikes
        You may fix the random number seed
        '''
        znorm = np.random.normal(size=10000)
        forward = spot
        prices = []
        for strike in strike:
            price = forward + np.sqrt(texp) * znorm * self.sigma
            price = np.mean(np.fmax(cp*(price - strike), 0))
            prices.append(price)
        return np.array(prices)

'''
Conditional MC model class for Beta=1
'''
class ModelBsmCondMC:
    beta = 1.0   # fixed (not used)
    vov, rho = 0.0, 0.0
    sigma, intr, divr = None, None, None
    bsm_model = None
    '''
    You may define more members for MC: time step, etc
    '''
    
    def __init__(self, sigma, vov=0, rho=0.0, beta=1.0, intr=0, divr=0):
        self.sigma = sigma
        self.vov = vov
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.bsm_model = pf.Bsm(sigma, intr=intr, divr=divr)
        
    def bsm_vol(self, strike, spot, texp=None):
        ''''
        should be same as bsm_vol method in ModelBsmMC (just copy & paste)
        '''
        p = self.price(strike, spot, texp).mean(axis=0)
        return self.bsm_model.impvol(p, strike, spot, texp)
    
    def price(self, strike, spot, texp=None, cp=1):
        '''
        Your MC routine goes here
        Generate paths for vol only. Then compute integrated variance and BSM price.
        Then get prices (vector) for all strikes
        You may fix the random number seed
        '''
        m = pf.BsmNdMc(self.vov, rn_seed=12345)
        tobs = np.arange(0, 101)/100*texp
        _ = m.simulate(tobs = tobs, n_path=1000)
        sigma_path = np.squeeze(m.path)
        sigma_final = sigma_path[-1,:]
        int_var = spint.simps(sigma_path**2, dx=1, axis=0)/100
        price = []
        model = pf.Bsm(sigma = np.sqrt((1 - self.rho ** 2) * np.mean(int_var)) * self.sigma , intr = self.intr, divr = self.divr)
        for strike in strike:
            price.append(model.price(strike, spot * np.exp(self.rho * (np.mean(sigma_final) * self.sigma - self.sigma) / self.vov - (self.rho ** 2) * (self.sigma ** 2) * texp * np.mean(int_var) / 2), texp))
        return np.array(price)

'''
Conditional MC model class for Beta=0
'''
class ModelNormalCondMC:
    beta = 0.0   # fixed (not used)
    vov, rho = 0.0, 0.0
    sigma, intr, divr = None, None, None
    normal_model = None
    
    def __init__(self, sigma, vov=0, rho=0.0, beta=0.0, intr=0, divr=0):
        self.sigma = sigma
        self.vov = vov
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.normal_model = pf.Norm(sigma, intr=intr, divr=divr)
        
    def norm_vol(self, strike, spot, texp=None):
        ''''
        should be same as norm_vol method in ModelNormalMC (just copy & paste)
        '''
        p = self.price(strike, spot, texp).mean(axis=0)
        return self.normal_model.impvol(p, stike, spot, texp)
        
    def price(self, strike, spot, texp=None, cp=1):
        '''
        Your MC routine goes here
        Generate paths for vol only. Then compute integrated variance and normal price.
        You may fix the random number seed
        '''
        m = pf.BsmNdMc(self.vov, rn_seed=12345)
        tobs = np.arange(0, 101)/100*texp
        _ = m.simulate(tobs = tobs, n_path=1000)
        sigma_path = np.squeeze(m.path)
        sigma_final = sigma_path[-1,:]
        int_var = spint.simps(sigma_path**2, dx=1, axis=0)/100
        price = []
        model = pf.Norm(sigma = np.sqrt((1 - self.rho ** 2) * np.mean(int_var)) * self.sigma , intr = self.intr, divr = self.divr)
        for strike in strike:
            price.append(model.price(strike, spot + self.rho * (np.mean(sigma_final) * self.sigma - self.sigma) / self.vov, texp))
        return np.array(price)
