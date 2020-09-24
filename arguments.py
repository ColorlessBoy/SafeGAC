import argparse

"""
Here are the param for the training

"""

def get_args():
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--alg', type=str, default='gac', help='the algorithm name')
    parser.add_argument('--env-name', type=str, default='Safexp-PointGoal0-v0', help='the environment name')
    parser.add_argument('--n-epochs', type=int, default=50, help='the number of epochs to train the agent')
    parser.add_argument('--n-train-rollouts', type=int, default=4, help='the number of train')
    parser.add_argument('--n-batches', type=int, default=50, help='the times to update the network')
    parser.add_argument('--save-interval', type=int, default=5, help='the interval that save the trajectory')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--num-workers', type=int, default=1, help='the number of cpus to collect samples')
    parser.add_argument('--replay-strategy', type=str, default='future', help='the HER strategy')
    parser.add_argument('--save-dir', type=str, default='data/', help='the path to save the data')
    parser.add_argument('--noise-eps', type=float, default=0.2, help='noise eps')
    parser.add_argument('--buffer-size', type=int, default=int(1e6), help='the size of the buffer')
    parser.add_argument('--batch-size', type=int, default=256, help='the sample batch size')
    parser.add_argument('--gamma', type=float, default=0.99, help='the discount factor')
    parser.add_argument('--action-l2', type=float, default=1, help='l2 reg')
    parser.add_argument('--lr-actor', type=float, default=0.001, help='the learning rate of the actor')
    parser.add_argument('--lr-critic', type=float, default=0.001, help='the learning rate of the critic')
    parser.add_argument('--polyak', type=float, default=0.995, help='the average coefficient')
    parser.add_argument('--n-test-rollouts', type=int, default=10, help='the number of tests')
    parser.add_argument('--clip-range', type=float, default=5, help='the clip range')
    parser.add_argument('--demo-length', type=int, default=20, help='the demo length')
    parser.add_argument('--cuda', action='store_true', help='if use gpu do the acceleration')
    parser.add_argument('--mmd', action='store_true', help='if use mmd entropy in gac')
    parser.add_argument('--expand-batch', type=int, default=100, help='batch of actions for mmd')
    parser.add_argument('--beta-mmd', type=float, default=0.1, help='hyper_parameter of mmd_entropy')
    parser.add_argument('--reward-scale', type=float, default=1.0, help='true-reward = original-reward * reward-scale')
    parser.add_argument('--alpha', type=float, default=0.02, help='hyperparameter of entropy in sac')
    parser.add_argument('--load-fold', type=str, default='tmp', help='load data and model from this fold')
    parser.add_argument('--render', action='store_true', help='enable env.render()')
    parser.add_argument('--warmup_steps', type=int, default=10000, help='warm up for replay buffer')

    args = parser.parse_args()

    return args
