import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gym
from gym.spaces import Discrete,Box
import argparse
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs


#Function to implement MultiLayer Perceptron (Feed forward Neural network)
def MLP(sizes,activation=nn.Tanh,output_activation=nn.Identity):
	nnlayers = []
	for i in range(len(sizes)-1):
		act = activation if i <len(sizes)-2 else output_activation
		nnlayers += [nn.Linear(sizes[i],sizes[i+1]),act()]
	return nn.Sequential(*nnlayers)

#Function for training 
def train(env_name = 'CartPole-v0',hiddensizes=[32],lr = 1e-2,epochs = 50,
	batchsize = 5000, render = False):
    
    setup_pytorch_for_mpi()

    seed = 0
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    local_batch_size =  int(batchsize/num_procs())

    env = gym.make(env_name)
    observations_dim = env.observation_space.shape[0]
    number_of_actions = env.action_space.n
    logits_nw = MLP(sizes = [observations_dim]+hiddensizes+[number_of_actions])

    sync_params(logits_nw)

    
    def get_policy(obs):
	    logits = logits_nw(obs)
	    return Categorical(logits = logits)

    #Function returning a scalar value sampled from the Catgorical distribution 
    def get_action(obs):
    	return get_policy(obs).sample().item()

    # weights = R(Tau) i.e. total reward for single episode replicated 
    # for number of times that is equal to episode length
    def compute_loss(obs,action,weights):
    	logprob = get_policy(obs).log_prob(action)
    	# mean is returned since total value is averaged
    	# total number of timesteps in the batch
    	# each episode might have different timesteps .Hence ,two one dimensional vectors are
    	# multiplied .Summed and averaged by total timesteps in all trajectories 
    	# Here, sum is divided by DT  but in expression it is divided by D ( T = num of time steps in trajectory , D = num of trajectories)
    	return -(logprob*weights).mean()

    optimizer = Adam(logits_nw.parameters(),lr=lr)

    def train_single_epoch():
        batch_observations=[]
        batch_actions=[]
        batch_weights=[]
        batch_total_of_rewards_in_episode =[]
        batch_episode_lengths =[]
    	

        obs = env.reset()
        episode_completed = False
        episode_rewards = []
        completed_rendering_current_epoch = False

        while True:
            #if(not completed_rendering_current_epoch) and render:
            #    env.render()
            batch_observations.append(obs.copy())
            action = get_action(torch.as_tensor(obs,dtype=torch.float32))
            obs,rew,episode_completed,_ = env.step(action)
            batch_actions.append(action)
            episode_rewards.append(rew)

            if episode_completed:
                episode_rewards_total = sum(episode_rewards)
                episode_len = len(episode_rewards)

                batch_total_of_rewards_in_episode.append(episode_rewards_total)
                batch_episode_lengths.append(episode_len)

                batch_weights += [episode_rewards_total]*episode_len

                #Resetting the Observations, episode rewards and the episde completed flag
                #Getting ready for a new episode
                obs,episode_rewards,episode_completed = env.reset(),[],False

                # the size of batch is enough
                if(len(batch_observations))>local_batch_size:
                    break

        optimizer.zero_grad()
        batch_loss = compute_loss(obs = torch.as_tensor(batch_observations,dtype=torch.float32),action = torch.as_tensor(batch_actions,dtype=torch.int32),weights = torch.as_tensor(batch_weights,dtype=torch.int32))
        batch_loss.backward()
        mpi_avg_grads(logits_nw) 
        optimizer.step()
        return batch_loss,batch_total_of_rewards_in_episode,batch_episode_lengths

    for i in range(epochs):
        batch_loss,batch_total_of_rewards_in_episode,batch_episode_lengths = train_single_epoch()
        average_of_batch_total_of_rewards_in_episode = np.mean(batch_total_of_rewards_in_episode)
        average_of_batch_episode_lengths = np.mean(batch_episode_lengths)
        if proc_id() == 0:     
            print('epoch:%3d\t loss:%.3f \t return:%.3f\t episode_length:%.3f'%(i,batch_loss,average_of_batch_total_of_rewards_in_episode,average_of_batch_episode_lengths))
    env.env.close()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name',type=str,default='CartPole-v0')
    parser.add_argument('--render',default='store_true')
    parser.add_argument('--lr',type=float,default=1e-2)
    args=parser.parse_args()
    mpi_fork(4)
    train(env_name = args.env_name,render=args.render,lr=args.lr)
