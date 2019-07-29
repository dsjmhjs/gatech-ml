"""
Template for implementing QLearner  (c) 2015 Tucker Balch
"""

import numpy as np
import random as rand
import time

class QLearner(object):

    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        dyna = 0, \
        verbose = False):

        self.verbose = verbose
        self.num_actions = num_actions
        self.s = 0
        self.a = 0
        self.rar = rar
        self.radr = radr
        self.alpha = alpha
        self.gamma = gamma
        self.num_states = num_states
        self.dyna = dyna
        self.Q = np.zeros(shape=(num_states, num_actions))
        self.T = np.full((num_states, num_actions, num_states), 0.0000001)
        self.R = np.zeros(shape=(num_states, num_actions))

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        self.s = s
        if rand.random() > self.rar:
        	action = np.argmax(self.Q, axis = 1)[s]
    	else:
    		action = rand.randint(0, self.num_actions-1)
        if self.verbose: print "s =", s,"a =",action
        return action

    def query(self,s_prime,r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The ne state
        @returns: The selected action
        """
        
        if rand.random() > self.rar:
        	action = np.argmax(self.Q, axis = 1)[s_prime]
    	else:
    		action = rand.randint(0, self.num_actions-1)
    	self.rar = self.rar * self.radr
        self.Q[self.s, self.a] = (1 - self.alpha) * self.Q[self.s, self.a] + self.alpha * (r + self.gamma * self.Q[s_prime, action])
        if self.verbose: print "s =", s_prime,"a =",action,"r =",r
        if self.dyna > 0:
        	# time.sleep(1)
	        self.T[self.s, self.a, s_prime] += 1
	        self.R[self.s, self.a] = (1 - self.alpha) * self.R[self.s, self.a] + self.alpha * r
	        dyna_s = np.random.randint(0, self.num_states, size=self.dyna)
	        dyna_a = np.random.randint(0, self.num_actions, size=self.dyna)
	        dyna_r = self.R[dyna_s, dyna_a]
	        dyna_s_prime = np.argmax(self.T[dyna_s, dyna_a, :], axis=1)
	        self.Q[dyna_s, dyna_a] = (1 - self.alpha) * self.Q[dyna_s, dyna_a] + self.alpha * (dyna_r + self.gamma * self.Q[dyna_s_prime, dyna_a])
        self.s = s_prime
        self.a = action
        return action

    def author(self):
		return 'xhuang343' # replace tb34 with your Georgia Tech username.

if __name__=="__main__":
    print "Remember Q from Star Trek? Well, this isn't him"
