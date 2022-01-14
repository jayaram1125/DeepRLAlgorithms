import numpy as np
from gym.spaces import Discrete,Box
import torch
import torch.nn as nn
from torch.optim import Adam
import gym
from collections import deque
from collections import namedtuple
import random
import matplotlib.pyplot as plt


#Function implementing Multi layer perceptron
def MLP(sizes,activation = nn.Tanh,output_activation = nn.Identity):
    nnlayers =[]
    for j in range(len(sizes)-1):
        act = activation if j<len(sizes)-2 else output_activation 
        nnlayers += [nn.Linear(sizes[j],sizes[j+1]),act()]
    return nn.Sequential(*nnlayers)

'''
Class Implementing predicton Q-Network and Target Q-Network
'''
class QNetwork(nn.Module):
    def __init__ (self,observation_space,action_space,hidden_sizes):
        super().__init__()
        obs_dim = observation_space.shape[0]
        act_dim = action_space.n
        self.predict_q_logits_net = MLP([obs_dim]+hidden_sizes+[act_dim])
        self.target_q_logits_net =  MLP([obs_dim]+hidden_sizes+[act_dim])

class ReplayMemory:
    def __init__(self,capacity):
        self.capacity = capacity
        self.dq = deque([])
        self.fields = ['state', 'act', 'rew','nextstate','done'] 

    def store(self,timestep):
        if(len(self.dq) >= self.capacity):
             self.dq.popleft()
        self.dq.append(timestep)      

    def sampleandget(self,batch_size):
        if len(self.dq) < batch_size:
            batch_size = len(self.dq)
        batch = random.sample(self.dq, batch_size) 
        data = {field_name: np.array([getattr(batch[i], field_name) for i in range(batch_size)]) for field_name in self.fields}
        return {k:torch.as_tensor(v,dtype =torch.float32) for k,v in data.items()}        


def train():
    gamma = 0.95
    
    replaymemorycapacity = 256
    batch_size = 16
     
    hidden_sizes = [64]
    learningrate = 1e-3

    epsilon = 1.0
    target_nw_update_frequency = 5
    n_episodes = 5000

    seed = 1423
    torch.manual_seed(seed)
    env = gym.make('CartPole-v0')
    q = QNetwork(env.observation_space,env.action_space,hidden_sizes)
    mem = ReplayMemory(replaymemorycapacity)

    prediction_qnet_optimizer = Adam(q.predict_q_logits_net.parameters(),lr = learningrate)

    TimeStep = namedtuple('Timestep', ['state', 'act', 'rew','nextstate','done'])
        
    def getaction(state, action_space_len, epsilon):
        #print('state.dtype=',state.dtype)
        with torch.no_grad():
            #predict_q_logits = q.predict_q_logits_net(torch.as_tensor(state,dtype =torch.float32))
            predict_q_logits = q.predict_q_logits_net(torch.from_numpy(state).float())
        Q, A = torch.max(predict_q_logits, axis=0)
        A = A if torch.rand(1, ).item() > epsilon else torch.randint(0, action_space_len, (1,))
        return A
   
    def maxqtargetnetvalue_nextstate(sampled_nextstates_buf):
        with torch.no_grad():
            target_q_logits = q.target_q_logits_net(sampled_nextstates_buf)
            return torch.max(target_q_logits,dim=1)[0]     

    def predictedQvalue(sampled_states_buf):                     
        predict_q_logits = q.predict_q_logits_net(sampled_states_buf)
        return torch.max(predict_q_logits,dim=1)[0]     
              

    def computeloss(episode_len):
        if episode_len%target_nw_update_frequency:
            update_target_q_net()     
        data = mem.sampleandget(batch_size)
        sampled_states_buf =  data['state']
        sampled_act_buf = data['act']
        sampled_rew_buf = data['rew']
        sampled_nextstates_buf = data['nextstate']
        sampled_done_buf = data['done']
        yj = sampled_rew_buf + (1-sampled_done_buf)*gamma*maxqtargetnetvalue_nextstate(sampled_nextstates_buf)
        #print(yj)
        predictedQ = predictedQvalue(sampled_states_buf)  
        #print(predictedQ)   
        lossfn = nn.MSELoss()       
        loss =  lossfn(predictedQ,yj)
        return loss 

    def update_target_q_net():
        q.target_q_logits_net.load_state_dict(q.predict_q_logits_net.state_dict())        

    exp_replay_size = 256
    index = 0
    for i in range(exp_replay_size):
        obs = env.reset()
        done = False
        while(done != True):
            a =  getaction(obs, env.action_space.n, epsilon) 
            next_obs,r,done,_ = env.step(a.item())
            mem.store(TimeStep(obs,a,r,next_obs,done))
            obs = next_obs
            index += 1
        if(index > exp_replay_size):
            break  
    
    index = 128

    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        episode_ret = 0
        episode_len = 0
        losses = 0
        while not done:
            a =  getaction(obs, env.action_space.n, epsilon)    
            next_obs,r,done,_ = env.step(a.item())
            mem.store(TimeStep(obs,a,r,next_obs,done))
            
            episode_ret += r
            episode_len += 1
            obs = next_obs
            index += 1
            if(index > 128):
                index = 0
                for j in range(4):               
                    loss = computeloss(episode_len)
                    prediction_qnet_optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    prediction_qnet_optimizer.step()
        if epsilon > 0.05 :
            epsilon -= (1 / 5000)
        total_episode_return.append(episode_ret)
        episode_list.append(episode)    
        print('Episode:%d\t Episode Return :%.3f \t Episode len :%.3f \t'%(episode,episode_ret,episode_len))
    env.env.close()    	    


total_episode_return = [] 
episode_list = []
train()
plt.plot(episode_list,total_episode_return)
plt.xlabel('Episode number')
plt.ylabel('Episode return')
plt.title('DQN training')
plt.show()







