import numpy as np
import gym
import safety_gym
import os, sys
from arguments import get_args
from mpi4py import MPI
from rl_modules.gac_agent import gac_agent
import random
import torch

"""
train the agent, the MPI part code is copy from openai baselines(https://github.com/openai/baselines/blob/master/baselines/her)

"""
def get_env_params(env):
    obs = env.reset()
    # close the environment
    params = {'obs': obs.shape[0],
            'action': env.action_space.shape[0],
            'action_max': env.action_space.high[0],
            }
    # params['max_timesteps'] = env._max_episode_steps
    params['max_timesteps'] = 1000
    return params

def launch(args):
    env = gym.make(args.env_name)
    test_env = gym.make(args.env_name)
    # set random seeds for reproduce
    env.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    np.random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    if args.cuda:
        torch.cuda.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    # get the environment parameters
    env_params = get_env_params(env)
    
    # create the ddpg_agent
    if args.alg == 'gac':
        print("Start GAC...")
        gac_trainer = gac_agent(args, env, test_env, env_params)
        gac_trainer.learn()
    elif args.alg == 'sac':
        print("Missing SAC.")
    elif args.alg == 'td3':
        print('Missing TD3.')
    else:
        print("Missing DDPG.")

if __name__ == '__main__':
    # take the configuration for the HER
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'
    # get the params
    args = get_args()
    launch(args)
