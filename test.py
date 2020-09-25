import torch
import os
import gym
from gym import wrappers
import numpy as np
import joblib

from logx import EpochLogger
from train import get_env_params
from arguments import get_args

from rl_modules.models import actor
from rl_modules.sac_models import actor as actor_sac
from rl_modules.gac_models import actor as actor_gac

import time

"""
    args needs:
        --alg
        --env-name
        --load-fold
        --cuda
        --n-test-rollouts
    
    For remote server:
        xvfb-run -s "-screen 0 1400x900x24" bash
"""
class Test:
    def __init__(self, args):
        self.args = args
        self.env = gym.make(args.env_name)
        self.env_params = get_env_params(self.env)

        self.video_file = 'data_test/test_video'
        self.output_dir = 'data_test'
        self.exp_name = 'test'
        self.logger = EpochLogger(output_dir=self.output_dir, exp_name=self.exp_name)
        # self.env = wrappers.Monitor(self.env, self.video_file, force=True)

        device = 'cuda' if args.cuda else 'cpu'
        self.device = torch.device(device)

        # load
        data_file = os.path.join(args.load_fold, 'vars.pkl')
        data = joblib.load(data_file)

        ## load obs_mean obs_std g_mean g_std
        self.obs_mean = data['observation_mean']
        self.obs_std  = data['observation_std']

        ## load policy model
        model = {
            'ddpg' : actor,
            'td3'  : actor,
            'sac'  : actor_sac,
            'gac'  : actor_gac
        }
        self.actor_network = model[args.alg](self.env_params).to(self.device)
        model_file = os.path.join(args.load_fold, 'pyt_save', 'model.pt')
        self.actor_network.load_state_dict(torch.load(model_file))

    def run(self):
        self._eval_agent()
        self.logger.log_tabular('EpReward')
        self.logger.log_tabular('EpCost')
        self.logger.dump_tabular()

    def _preproc_inputs(self, obs):
        obs_norm = np.clip((obs-self.obs_mean)/self.obs_std, 
                            -self.args.clip_range, self.args.clip_range)
        # concatenate the stuffs
        inputs = torch.tensor(obs_norm, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda(self.device)
        return inputs

    def _eval_agent(self):
        for _ in range(self.args.n_test_rollouts):
            obs, ep_reward, ep_cost = self.env.reset(), 0, 0
            for _ in range(self.env_params['max_timesteps']):
                if self.args.render:
                    self.env.render()
                    time.sleep(1e-3)

                with torch.no_grad():
                    input_tensor = self._preproc_inputs(obs)
                    if self.args.alg == 'gac':
                        pi = self.actor_network(input_tensor, std=0.5)
                    elif self.args.alg == 'sac':
                        pi, _ = self.actor_network(input_tensor)
                    else:
                        pi = self.actor_network(input_tensor)
                    # convert the actions
                    actions = pi.detach().cpu().numpy().squeeze()
                obs, reward, cost, info = self.env.step(actions)
                ep_reward += reward
                ep_cost += cost
                self.logger.store(EpReward=ep_reward, EpCost=ep_cost)

if __name__ == '__main__':
#    from pyvirtualdisplay import Display
#    virtual_display = Display(visible=0, size=(1400, 900))
#    virtual_display.start()

    args = get_args()
    test = Test(args)
    test.run()
    test.env.close()
