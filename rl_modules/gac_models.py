import torch
import torch.nn as nn
import torch.nn.functional as F


# define the generator network
class actor(nn.Module):
    def __init__(self, env_params):
        super(actor, self).__init__()
        self.epsilon_dim = env_params['action']
        self.fc1 = nn.Linear(env_params['obs'] + self.epsilon_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.action_out = nn.Linear(300, env_params['action'])

    def forward(self, x, std=1.0, epsilon_limit=5.0):
        epsilon = (std * torch.randn(x.shape[0], self.epsilon_dim, 
                    device=x.device)).clamp(-epsilon_limit, epsilon_limit)
        x = torch.cat([x, epsilon], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        actions = torch.tanh(self.action_out(x))

        return actions

class critic(nn.Module):
    def __init__(self, env_params):
        super(critic, self).__init__()
        self.fc1 = nn.Linear(env_params['obs'] + env_params['action'], 400)
        self.fc2 = nn.Linear(400, 300)
        self.q_out = nn.Linear(300, 1)

    def forward(self, x, actions):
        x = torch.cat([x, actions], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.q_out(x)

        return q_value

# Maximum Mean Discrepancy
# geomloss: https://github.com/jeanfeydy/geomloss

class Sqrt0(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        result = input.sqrt()
        result[input < 0] = 0
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_tensors
        grad_input = grad_output / (2*result)
        grad_input[result == 0] = 0
        return grad_input

def sqrt_0(x):
    return Sqrt0.apply(x)

def squared_distances(x, y):
    if x.dim() == 2:
        D_xx = (x*x).sum(-1).unsqueeze(1)  # (N,1)
        D_xy = torch.matmul( x, y.permute(1,0) )  # (N,D) @ (D,M) = (N,M)
        D_yy = (y*y).sum(-1).unsqueeze(0)  # (1,M)
    elif x.dim() == 3:  # Batch computation
        D_xx = (x*x).sum(-1).unsqueeze(2)  # (B,N,1)
        D_xy = torch.matmul( x, y.permute(0,2,1) )  # (B,N,D) @ (B,D,M) = (B,N,M)
        D_yy = (y*y).sum(-1).unsqueeze(1)  # (B,1,M)
    else:
        print("x.shape : ", x.shape)
        raise ValueError("Incorrect number of dimensions")

    return D_xx - 2*D_xy + D_yy

def gaussian_kernel(x, y, blur=1.0):
    C2 = squared_distances(x / blur, y / blur)
    return (- .5 * C2 ).exp()

def energy_kernel(x, y, blur=None):
    return -squared_distances(x, y)

kernel_routines = {
    "gaussian" : gaussian_kernel,
    "energy"   : energy_kernel,
}

def mmd(x, y, kernel='energy'):
    b = x.shape[0]
    m = x.shape[1]
    n = y.shape[1]

    if kernel in kernel_routines:
        kernel = kernel_routines[kernel]

    K_xx = kernel(x, x).mean()
    K_xy = kernel(x, y).mean()
    K_yy = kernel(y, y).mean()

    return sqrt_0(K_xx + K_yy - 2*K_xy)