import gym
import random
import numpy as np
import pylab as pl
import matplotlib
#%matplotlib inline  

#This code synchronously updates the the action values of all action using both Q-learning and 
# double Q-learning over 10000 trials 
# Running the script will produce the figure (similar) from the document
#Learning rate is exponential (suggested in paper)

env = gym.make('Roulette-v0')

#Initialize table with all zeros
Q = np.zeros([env.observation_space.n,env.action_space.n])

#Initialize tables for double Q
Qa = np.zeros([env.observation_space.n,env.action_space.n])
Qb = np.zeros([env.observation_space.n,env.action_space.n])
updates=['updateA','updateB']

#lists to store the average action values
Q_episode=[]
Q_episode1=[]
# Set learning parameters
y = .98
num_episodes = 10000
#create lists to contain total rewards and steps per episode

for i in range(num_episodes):
    #Reset environment and get first new observation
    q_all=[]
    
    Qsum=0.0
    Qsum1=0.0
    
    if i==0:
        lr=1
    else:
        lr = 1/np.power(i,0.7)
    lr = float(lr)

    #env.render()
    s = env.reset()
    rAll = 0

    d = False
    j = 0
    #The Q-Table learning algorithm
    for a in np.arange(env.action_space.n):
        
        #Choose an action by greedily (with noise) picking from Q table
        #a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
        
        #Get new state and reward from environment
        s1,r,d,_ = env.step(a)
        #Update Q-Table with new knowledge
        Q[s,a] = Q[s,a] + lr*(r + y*np.max(Q[s1,:]) - Q[s,a])
        #q_all.append(Q[s,a])
        Qsum+=Q[s,a]
        
        choice = random.sample(updates,1)
        choice = choice[0]
        if choice=='updateA':
            a_star = np.argmax(Qa[s1,:])
            Qa[s,a] = Qa[s,a] + lr*(r + y*Qb[s1,a_star] - Qa[s,a])
            Qsum1+=Qa[s,a]
        if choice=='updateB':
            b_star = np.argmax(Qb[s1,:])
            Qb[s,a] = Qb[s,a] + lr*(r + y*(Qa[s1,b_star] - Qb[s,a]))
            Qsum1+=Qb[s,a]
        
        rAll += r
        s = s1
        #if d == True:
           # break
    
    #qval = Qsum/float(env.action_space.n)
    #qm = np.asarray(q_all,dtype='float32')
    qm = Qsum/float(env.action_space.n)
    qm1 = Qsum1/float(env.action_space.n)
    Q_episode.append(qm)
    Q_episode1.append(qm1)
    #jList.append(j)
    #rList.append(rAll)

#RR = np.asarray(RR,dtype='float32')
eps = np.arange(num_episodes)
Q_episode = np.asarray(Q_episode,dtype='float32')
Q_episode1 = np.asarray(Q_episode1,dtype='float32')
pl.plot(eps,Q_episode,label='Q-learning')
pl.plot(eps,Q_episode1,label='Double Q')
legend = pl.legend(loc='upper center', shadow=True)

pl.show()
