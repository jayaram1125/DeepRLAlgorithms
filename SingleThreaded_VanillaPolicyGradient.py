'''Implementation for continuous state(Box) and discrete action spaces'''
import numpy as np
import scipy.signal
from gym.spaces import Discrete,Box
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import gym
import datetime
import argparse

#Util function which augments the dimension by values specified by shape 
def combined_shape(length,shape=None):
    if shape is None:
        return (length,)
    else:
        return (length,shape) if np.isscalar(shape) else (length,*shape)

#Function implementing Multi layer perceptron
def MLP(sizes,activation=nn.Tanh,output_activation = nn.Identity):
    nnlayers =[]
    for j in range(len(sizes)-1):
        act = activation if j<len(sizes)-2 else output_activation 
        nnlayers += [nn.Linear(sizes[j],sizes[j+1]),act()]
    return nn.Sequential(*nnlayers)

def discount_cumsum(x,discount):
    '''
    input: 
        vector x, 
        [x0, 
         x1, 
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    '''
    return scipy.signal.lfilter([1],[1,float(-discount)],x[::-1], axis =0)[::-1]

'''
Class Implementing 
MLPActor -for policy and MLPCritic - for value function  
'''
class MLPActorCritic(nn.Module):
    def __init__ (self,observation_space,action_space,hidden_sizes,activation = nn.Tanh):
        super().__init__()
        obs_dim = observation_space.shape[0]
        act_dim = action_space.n
        self.pi_logits_net = MLP([obs_dim]+hidden_sizes+[act_dim],activation)
        self.v_logits_net =  MLP([obs_dim]+hidden_sizes+[1],activation)
    
    def step(self,obs):
        with torch.no_grad():
            pi_logits = self.pi_logits_net(obs)
            pi = Categorical(logits = pi_logits)
            a = pi.sample()
            logp_a = pi.log_prob(a)
            v_logits = self.v_logits_net(obs)
            v = torch.squeeze(v_logits,-1)
        return a.numpy(),v.numpy(),logp_a.numpy()
   
'''
Class Implementing a Buffer to store the value at each state in trajectory.
Values are stored at each timestep in trajectory or episode 
After the epoch completes , these stored values are used for computaion of advantage value and state value function value
'''
class VPGBuffer:
    def __init__(self,obs_dim,act_dim,size,gamma=0.99,lam=0.95):
        self.obs_buf = np.zeros(combined_shape(size,obs_dim),dtype =np.float32)
        self.act_buf = np.zeros(combined_shape(size,act_dim),dtype =np.float32)
        self.adv_buf = np.zeros(size,dtype = np.float32)
        self.rew_buf = np.zeros(size,dtype = np.float32)
        self.ret_buf = np.zeros(size,dtype = np.float32)
        self.val_buf = np.zeros(size,dtype = np.float32)
        self.logp_buf = np.zeros(size,dtype = np.float32)
        self.gamma = gamma  
        self.lam  = lam
        self.ptr ,self.path_start_index,self.max_size = 0,0,size

    def store(self,obs,act,rew,val,logp):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr +=1

    def finish_path(self,last_val=0):
        path_index_range = slice(self.path_start_index,self.ptr)
        rews = np.append(self.rew_buf[path_index_range],last_val)
        vals = np.append(self.val_buf[path_index_range],last_val)
        ''' Calulation of deltas
           deltaVt = rt +gamma*V(st+1) -V(st)
           deltaVt+1 = rt+1 +gamma*V(st+2) -V(st+1)
           .
           . 
           '''
        deltas = rews[:-1]+self.gamma*vals[1:]-vals[:-1]
        self.adv_buf[path_index_range] = discount_cumsum(deltas,self.gamma*self.lam)
        self.ret_buf[path_index_range] = discount_cumsum(rews,self.gamma)[:-1]
        self.path_start_index = self.ptr
    def get(self):
        assert self.ptr == self.max_size
        self.ptr,self.path_start_index = 0,0 
        adv_mean = np.mean(self.adv_buf)
        adv_std = np.std(self.adv_buf)                    
        self.adv_buf = (self.adv_buf - adv_mean) /adv_std
        data = dict(obs = self.obs_buf,act = self.act_buf,ret = self.ret_buf,adv = self.adv_buf,logp =self.logp_buf)
        return {k:torch.as_tensor(v,dtype =torch.float32) for k,v in data.items()}
    

def vpg(env_fn,actor_critic = MLPActorCritic,seed = 0,steps_per_epoch = 4000,epochs = 50,
        gamma = 0.99,pi_lr= 3e-4,vf_lr = 1e-3,train_v_iters = 80,lam = 0.97,max_ep_len = 1000):

    torch.manual_seed(seed)
    np.random.seed(seed)
    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape
    actor_critic_hidden_sizes = [64,64]
    ac = actor_critic(env.observation_space,env.action_space,actor_critic_hidden_sizes)
    buf = VPGBuffer(obs_dim,act_dim,steps_per_epoch,gamma,lam)

    def compute_loss_pi(data):
        obs,act,adv = data['obs'],data['act'],data['adv']
        pi = Categorical(logits = ac.pi_logits_net(obs))
        logp = pi.log_prob(act)
        loss_pi = -(logp*adv).mean()
        return loss_pi
    def compute_loss_v(data):
        obs,ret = data['obs'],data['ret']
        v_logits = ac.v_logits_net(obs)
        v = torch.squeeze(v_logits,-1)
        return ((v-ret)**2).mean()

    pi_optimizer = Adam(ac.pi_logits_net.parameters(),lr = pi_lr)
    vf_optimizer = Adam(ac.v_logits_net.parameters(),lr = vf_lr)

    def train_single_epoch():
        data = buf.get()
        #Single step of graident for training the Policy 
        pi_optimizer.zero_grad()
        loss_pi = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()   

        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            vf_optimizer.step()
        return loss_pi,loss_v            
                      
    
    o,ep_ret,ep_len = env.reset(),0,0 

    for epoch in range(epochs):
        batch_return = []
        batch_episode_lengths = []
        max_episode_return = 0
        #completed_rendering_current_epoch = False
        for step in range(steps_per_epoch):
            #if(not completed_rendering_current_epoch):
            #   env.render()
            a,v,logp =  ac.step(torch.as_tensor(o,dtype = torch.float32))
            print(a)
            next_o,r,done,_ = env.step(a)
            ep_ret += r
            ep_len += 1
            buf.store(o,a,r,v,logp)
            o = next_o
            
            timeout = ep_len == max_ep_len
            terminal = done or timeout
            epoch_ended = step == (steps_per_epoch-1)

            
            if terminal or epoch_ended:
                if epoch_ended and not(terminal):
                    print('Trajectory cut off by epoch at %d steps'%ep_len)
                if timeout or epoch_ended:
                    _,v,_ = ac.step(torch.as_tensor(o,dtype = torch.float32))
                else:
                    v = 0 
                buf.finish_path(v)
                batch_return.append(ep_ret)
                if ep_ret>max_episode_return :
                    max_episode_return = ep_ret        
                batch_episode_lengths.append(ep_len)
                o,ep_ret,ep_len = env.reset(),0,0 
        pi_batchloss,v_batchloss=train_single_epoch()
        print('Epoch:%d \t BatchLoss_Pi :%f \t BatchLoss_V :%f \t AverageBatchReturn:%.3f \t AverageBatchEpisodeLen:%.3f \t Max_episode_return: %.3f'
            %(epoch,pi_batchloss,v_batchloss,np.mean(batch_return),np.mean(batch_episode_lengths),max_episode_return))
    env.env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env',type=str,default='MountainCar-v0')
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--steps_per_epoch',type=int,default =4000)
    parser.add_argument('--epochs',type=int,default=50)
    args = parser.parse_args()

vpg(lambda:gym.make(args.env),actor_critic=MLPActorCritic,seed=args.seed,steps_per_epoch=args.steps_per_epoch,epochs=args.epochs)



