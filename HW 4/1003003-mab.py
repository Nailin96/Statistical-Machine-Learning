import numpy as np
import matplotlib.pyplot as plt

T = 1000

def e_greedy(Q,N,t,e):
  r = np.random.rand()
  r2 = np.random.randint(len(Q)-1)
  if r > e:
    return np.argmax(Q)
  elif r2 < np.argmax(Q):
        return r2
  return r2+1

def UCB(Q,N,t,c) :
  foo = np.ones(10)*t + 1
  foo = np.log(foo)
  foo = foo/(N+1)
  foo = np.sqrt(foo)
  foo = foo*c + Q
  foo = np.argmax(foo)
  return foo

def test_run (policy,param):
  true_means = np.random.normal (0,1,10)
  reward = np.zeros(T+1)
  Q = np.zeros(10)

  N = np.zeros(10)
  
  for i in range (T):
    a = policy(Q,N,i,param)
    r = np.random.normal (true_means [a], 1 )
    N[a] = N[a] + 1
    step = 1/N[a]
    Q[a] = Q[a] + ((r - Q[a]) * step)
    reward[i+1] = r
  return reward

def main(p):
  ave_g = np.zeros(T+1)
  ave_eg = np.zeros(T+1)
  ave_ucb = np.zeros(T+1)
  
  for i in range (2000):
    g = test_run(e_greedy,0.0)
    eg = test_run(e_greedy,p)
    ucb = test_run(UCB,p)
    
    ave_g += (g-ave_g)/(i+1)
    ave_eg += (eg-ave_eg)/(i+1)
    ave_ucb += (ucb-ave_ucb )/(i+1)

  t = np.arange(T+1)
  plt.plot(t,ave_g,'b-',t,ave_eg,'r-',t,ave_ucb,'g-')
  plt.show()
  print('parameter used = ' + str(p))
  
array = [0.1,0.2,0.3,0.4,0.5]
for i in array:
  main(i)