import numpy as np
import sys
from gym.envs.toy_text import discrete

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

GOAL = 4 # upper right corner
START = 20 # lower left corner
SNAKE1 = 7
SNAKE2 = 17

eps = 0.25

t = 0.01 #set your own parameter here for value iteration

class Robot_vs_snakes_world(discrete.DiscreteEnv):
    def __init__(self):
        self.shape = [5, 5]
        
        nS = np.prod(self.shape) # total states
        nA = 4 # total actions
        
        MAX_X, MAX_Y = self.shape
        
        P = {}
        grid = np.arange(nS).reshape(self.shape)
        it = np.nditer(grid, flags=['multi_index'])
        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index
            
            P[s] = {a: [] for a in range(nA)}
            """{0: [], 1: [], 2: [], 3: []}"""
            is_done = lambda s: s == GOAL # location
            
            if is_done(s):
                reward = 0.0
            elif s in [SNAKE1, SNAKE2]:
                reward = -15.0
            else:
                reward = -1.0
                
            if is_done(s):
                P[s][UP]=[(1.0,s,reward,True)]
                P[s][RIGHT]=[(1.0,s,reward,True)]
                P[s][DOWN]=[(1.0,s,reward,True)]
                P[s][LEFT]=[(1.0,s,reward,True)]
            else:
                ns_up = s if y==0 else s-MAX_X
                ns_right = s if x==(MAX_X-1) else s+1
                ns_down = s if y==(MAX_Y-1) else s+MAX_X
                ns_left = s if x==0 else s-1
                P[s][UP] = [(1-(2*eps),ns_up,reward,is_done(ns_up)),
                            (eps,ns_right,reward,is_done(ns_right)),
                            (eps,ns_left,reward,is_done(ns_left))]
                
                P[s][RIGHT]=[(1-(2*eps),ns_right,reward,is_done(ns_right)),  
                             (eps,ns_up,reward,is_done(ns_up)),
                             (eps,ns_down,reward,is_done(ns_down))]
                
                P[s][DOWN] = [(1-(2*eps),ns_down,reward,is_done(ns_down)),
                              (eps,ns_right,reward,is_done(ns_right)),
                              (eps,ns_left,reward,is_done(ns_left))]
                
                P[s][LEFT] = [(1-(2*eps) , ns_left , reward , is_done(ns_left)) ,
                              ( eps , ns_up , reward , is_done(ns_up)) ,
                              ( eps , ns_down , reward , is_done(ns_down))]
                
            it.iternext()
            
        isd = np.zeros(nS)
        isd[START] = 1.0
        self.P = P
        
        super(Robot_vs_snakes_world, self).__init__(nS , nA , P , isd)
                
    def _render(self):
        grid = np.arange(self.nS).reshape(self.shape)
        it = np.nditer(grid, flags=['multi_index'])
        
        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index
            
            if self.s == s:
                output = " R "
            elif s == GOAL:
                output = " G "
            elif s in [SNAKE1, SNAKE2]:
                output =  " S "
            else:
                output = " o "
                
            if x == 0:
                output = output.lstrip()
            if x == self.shape[1] - 1:
                output = output.rstrip()
                
            sys.stdout.write(output)
            
            if x == self.shape[1] - 1:
                sys.stdout.write("\n")
                
            it.iternext()
            
        sys.stdout.write("\n")

env = Robot_vs_snakes_world()

def step(s,V):
  action_count = env.nA
  v = np.zeros(action_count)
  for i in range (action_count):
    for j,k,l,u in env.P[s][i]:
      inc = l + V[k]
      v[i] = v[i] + j*inc
  return v

def value_iteration():
  S = env.nS # state count
  A = env.nA # action count
  policy = np.zeros ([S,A])
  V = np.zeros (S)
  
  while True:
    d = 0
    for i in range(S):
      n = step(i,V)
      n = np.max(n)
      V[i] = n
      diff = np.abs(n-V[i])
      d = max(diff,d)
    if d < t: # threshold value defined above
      break

  for i in range(S):
    n = step(i,V)
    n = np.argmax(n)
    policy[i,n] = 1.0 #discount factor is 1

  return policy,V

policy,V = value_iteration()
while True:
  env._render()
  direction = np.argmax(policy[env.s])
  env.step(direction)
  if env.s == GOAL:
    env._render()
    break

print('Policy')
for i,v in enumerate(policy):
  print(i)
  if v[0] == 1:
    print('up')
  if v[1] == 1:
    print('right')
  if v[2] == 1:
    print('down')
  if v[3] == 1:
    print('left')  
print('Value') 
print(V)