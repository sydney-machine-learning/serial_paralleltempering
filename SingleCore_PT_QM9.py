import os
import os.path as osp
import math
import random
import numpy as np
import torch
import copy
import argparse
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU

from torch_geometric.datasets import QM9
from torch_geometric.nn import NNConv, Set2Set
from torch_geometric.data import DataLoader
from torch_geometric.utils import remove_self_loops

target = 0
dim = 32
batch_size=128

class MyTransform(object):
    def __call__(self, data):
        data.y = data.y[:, target]
        return data


class Complete(object):
    def __call__(self, data):
        device = data.edge_index.device

        row = torch.arange(data.num_nodes, dtype=torch.long, device=device)
        col = torch.arange(data.num_nodes, dtype=torch.long, device=device)

        row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
        col = col.repeat(data.num_nodes)
        edge_index = torch.stack([row, col], dim=0)

        edge_attr = None
        if data.edge_attr is not None:
            idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
            size = list(data.edge_attr.size())
            size[0] = data.num_nodes * data.num_nodes
            edge_attr = data.edge_attr.new_zeros(size)
            edge_attr[idx] = data.edge_attr

        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        data.edge_attr = edge_attr
        data.edge_index = edge_index

        return data

path = osp.join(os.path.dirname(os.path.abspath("__file__")), 'data', 'QM9')
transform = T.Compose([MyTransform(), Complete(), T.Distance(norm=False)])
dataset = QM9(path, transform=transform).shuffle()

mean = dataset.data.y.mean(dim=0, keepdim=True)
std = dataset.data.y.std(dim=0, keepdim=True)
dataset.data.y = (dataset.data.y - mean) / std
mean, std = mean[:, target].item(), std[:, target].item()

test_dataset = dataset[513:769]
train_dataset = dataset[:512]

# Initialise and parse command-line inputs

parser = argparse.ArgumentParser(description='PT MCMC GNN')
parser.add_argument('-s', '--samples', help='Number of samples', default=800, dest="samples", type=int)
parser.add_argument('-r', '--replicas', help='Number of chains/replicas, best to have one per availble core/cpu',
                    default=4, dest="num_chains", type=int)
parser.add_argument('-lr', '--learning_rate', help='Learning Rate for Model', dest="learning_rate",
                    default=0.001, type=float)
parser.add_argument('-swap', '--swap', help='Swap Ratio', dest="swap_ratio", default=0.02, type=float)
parser.add_argument('-b', '--burn', help='How many samples to discard before determing posteriors', dest="burn_in",
                    default=0.50, type=float)
parser.add_argument('-pt', '--ptsamples', help='Ratio of PT vs straight MCMC samples to run', dest="pt_samples",
                    default=0.60, type=float)
parser.add_argument('-step', '--step_size', help='Step size for proposals (0.02, 0.05, 0.1 etc)', dest="step_size",
                   default=0.005, type=float)
parser.add_argument('-t', '--temperature', help='Demoninator to determine Max Temperature of chains (MT=no.chains*t) ',
                    default=2, dest="mt_val", type=int)  # Junk
args = parser.parse_args()


def f(): raise Exception("Found exit()")

class Net(torch.nn.Module):
    def __init__(self, lrate):
        super(Net, self).__init__()
        self.lin0 = torch.nn.Linear(dataset.num_features, dim)

        nn = Sequential(Linear(5, 128), ReLU(), Linear(128, dim * dim))
        self.conv = NNConv(dim, dim, nn, aggr='mean')
        self.gru = GRU(dim, dim)

        self.set2set = Set2Set(dim, processing_steps=3)
        self.lin1 = torch.nn.Linear(2 * dim, dim)
        self.lin2 = torch.nn.Linear(dim, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lrate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min',
                                                               factor=0.7, patience=5,
                                                               min_lr=0.00001)

    def forward(self,x,edge_index,edge_attr,batch):
        out = F.relu(self.lin0(x))
        h = out.unsqueeze(0)

        for i in range(3):
            m = F.relu(self.conv(out, edge_index, edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = self.set2set(out, batch)
        out = F.relu(self.lin1(out))
        out = self.lin2(out)
        return out.view(-1)

    def evaluate_proposal(self, data, w=None):
        if w is not None:
            self.loadparameters(w)
        y_pred = torch.zeros((len(data), batch_size))
        for i, sample in enumerate(data, 0):
            a = copy.deepcopy(self.forward(sample.x,sample.edge_index,sample.edge_attr,sample.batch).detach())
            y_pred[i] = a
        return y_pred

    def langevin_gradient(self, data, w=None):
        if w is not None:
            self.loadparameters(w)
        self.los = 0
        for i, sample in enumerate(data, 0):
            labels = sample.y
            self.optimizer.zero_grad()
            predicted = self.forward(sample.x,sample.edge_index,sample.edge_attr,sample.batch)
            loss = F.mse_loss(predicted, labels)
            loss.backward()
            self.los += loss.item() * sample.num_graphs
            self.optimizer.step()
        return copy.deepcopy(self.state_dict())

    def getparameters(self, w=None):
        l = np.array([1, 2])
        dic = {}
        if w is None:
            dic = self.state_dict()
        else:
            dic = copy.deepcopy(w)
        for name in sorted(dic.keys()):
            l = np.concatenate((l, np.array(copy.deepcopy(dic[name])).reshape(-1)), axis=None)
        l = l[2:]
        return l

    def dictfromlist(self, param):
        dic = {}
        i = 0
        for name in sorted(self.state_dict().keys()):
            dic[name] = torch.FloatTensor(param[i:i + (self.state_dict()[name]).view(-1).shape[0]]).view(
                self.state_dict()[name].shape)
            i += (self.state_dict()[name]).view(-1).shape[0]
        # self.loadparameters(dic)
        return dic

    def loadparameters(self, param):
        self.load_state_dict(param)

    def addnoiseandcopy(self, mea, std_dev):
        dic = {}
        w = self.state_dict()
        for name in (w.keys()):
            dic[name] = copy.deepcopy(w[name]) + torch.zeros(w[name].size()).normal_(mean=mea, std=std_dev)
        self.loadparameters(dic)
        return dic

class MCMC():
    def __init__(self, use_langevin_gradients, learn_rate, samples, burn_in, swap_interval, path, batch_size, step_size, num_replica, max_temp, pt_samples):
        self.gnn = Net(learn_rate)
        self.batch_size = batch_size
        self.adapttemp = 1
        self.maxtemp = max_temp
        self.num_chains = num_replica
        self.swap_interval = swap_interval
        self.path = path
        self.burn_in = burn_in
        self.geometric = True
        self.samples = samples
        self.traindata = DataLoader(train_dataset, batch_size=128, shuffle=True)
        self.testdata = DataLoader(test_dataset, batch_size=128, shuffle=False)
        self.use_langevin_gradients = use_langevin_gradients
        self.sgd_depth = 1  # Keep as 1
        self.learn_rate = learn_rate
        self.l_prob = 0.7  # Ratio of langevin based proposals, higher value leads to more computation time, evaluate for different problems
        self.step_size = step_size
        self.tau_sq = 0.5
        self.pt_samples = pt_samples
        self.temperatures = []

    def error(self, loader):
        error=0
        for data in loader:
            error += (self.gnn(data.x,data.edge_index,data.edge_attr,data.batch) * std - data.y * std).abs().sum().item()  # MAE
        return error/ len(loader.dataset)

    def likelihood_func(self, gnn, loader, tau_pro, w=None):
        y = torch.zeros((len(loader), self.batch_size))
        for i, dat in enumerate(loader, 0):
            labels = dat.y
            y[i]=labels
        if w is not None:
            fx = self.gnn.evaluate_proposal(loader, w)
        else:
            fx = self.gnn.evaluate_proposal(loader)
        error = self.error(loader)
        y=y.numpy()
        fx=fx.numpy()
        y=y.ravel()
        fx=fx.ravel()
        loss = np.sum(-0.5 * np.log(2 * math.pi * tau_pro) - 0.5 * np.square(y - fx) / tau_pro)
        lhood=np.sum(loss)
        return [lhood/self.adapttemp, fx, error]

    def prior_likelihood(self, sigma_squared, w_list):
        part1 = -1 * ((len(w_list)) / 2) * np.log(sigma_squared)
        part2 = 1 / (2 * sigma_squared) * (sum(np.square(w_list)))
        log_loss = part1 - part2
        return log_loss


    def default_beta_ladder(self, ndim, ntemps,
                            Tmax):  # https://github.com/konqr/ptemcee/blob/master/ptemcee/sampler.py
        """
        Returns a ladder of :math:`\beta \equiv 1/T` under a geometric spacing that is determined by the
        arguments ``ntemps`` and ``Tmax``.  The temperature selection algorithm works as follows:
        Ideally, ``Tmax`` should be specified such that the tempered posterior looks like the prior at
        this temperature.  If using adaptive parallel tempering, per `arXiv:1501.05823
        <http://arxiv.org/abs/1501.05823>`_, choosing ``Tmax = inf`` is a safe bet, so long as
        ``ntemps`` is also specified.
        """
        if type(ndim) != int or ndim < 1:
            raise ValueError('Invalid number of dimensions specified.')
        if ntemps is None and Tmax is None:
            raise ValueError('Must specify one of ``ntemps`` and ``Tmax``.')
        if Tmax is not None and Tmax <= 1:
            raise ValueError('``Tmax`` must be greater than 1.')
        if ntemps is not None and (type(ntemps) != int or ntemps < 1):
            raise ValueError('Invalid number of temperatures specified.')

        tstep = np.array([25.2741, 7., 4.47502, 3.5236, 3.0232,
                          2.71225, 2.49879, 2.34226, 2.22198, 2.12628,
                          2.04807, 1.98276, 1.92728, 1.87946, 1.83774,
                          1.80096, 1.76826, 1.73895, 1.7125, 1.68849,
                          1.66657, 1.64647, 1.62795, 1.61083, 1.59494,
                          1.58014, 1.56632, 1.55338, 1.54123, 1.5298,
                          1.51901, 1.50881, 1.49916, 1.49, 1.4813,
                          1.47302, 1.46512, 1.45759, 1.45039, 1.4435,
                          1.4369, 1.43056, 1.42448, 1.41864, 1.41302,
                          1.40761, 1.40239, 1.39736, 1.3925, 1.38781,
                          1.38327, 1.37888, 1.37463, 1.37051, 1.36652,
                          1.36265, 1.35889, 1.35524, 1.3517, 1.34825,
                          1.3449, 1.34164, 1.33847, 1.33538, 1.33236,
                          1.32943, 1.32656, 1.32377, 1.32104, 1.31838,
                          1.31578, 1.31325, 1.31076, 1.30834, 1.30596,
                          1.30364, 1.30137, 1.29915, 1.29697, 1.29484,
                          1.29275, 1.29071, 1.2887, 1.28673, 1.2848,
                          1.28291, 1.28106, 1.27923, 1.27745, 1.27569,
                          1.27397, 1.27227, 1.27061, 1.26898, 1.26737,
                          1.26579, 1.26424, 1.26271, 1.26121,
                          1.25973])

        if ndim > tstep.shape[0]:
            # An approximation to the temperature step at large
            # dimension
            tstep = 1.0 + 2.0 * np.sqrt(np.log(4.0)) / np.sqrt(ndim)
        else:
            tstep = tstep[ndim - 1]

        appendInf = False
        if Tmax == np.inf:
            appendInf = True
            Tmax = None
            ntemps = ntemps - 1

        if ntemps is not None:
            if Tmax is None:
                # Determine Tmax from ntemps.
                Tmax = tstep ** (ntemps - 1)
        else:
            if Tmax is None:
                raise ValueError('Must specify at least one of ``ntemps'' and '
                                 'finite ``Tmax``.')

            # Determine ntemps from Tmax.
            ntemps = int(np.log(Tmax) / np.log(tstep) + 2)

        betas = np.logspace(0, -np.log10(Tmax), ntemps)
        if appendInf:
            # Use a geometric spacing, but replace the top-most temperature with
            # infinity.
            betas = np.concatenate((betas, [0]))

        return betas

    def assign_temperatures(self):
        if self.geometric == True:
            betas = self.default_beta_ladder(2, ntemps=self.num_chains, Tmax=self.maxtemp)
            for i in range(0, self.num_chains):
                self.temperatures.append(np.inf if betas[i] == 0 else 1.0 / betas[i])
            return self.temperatures
        else:

            tmpr_rate = (self.maxtemp / self.num_chains)
            temp = 1
            for i in range(0, self.num_chains):
                self.temperatures.append(temp)
                temp += tmpr_rate
            return self.temperatures

    def sampler(self):

        total_samples = self.samples
        gnn = self.gnn

        # Random Initialisation of weights
        w = gnn.state_dict()
        w_size = len(gnn.getparameters(w))
        step_w = self.step_size
        sigma_squared = 25

        nreplicas = self.num_chains
        samples = total_samples / nreplicas

        temp_ladder = self.assign_temperatures()
        
        swap_proposed = 0
        successful_swap = 0

        burnin = int(self.burn_in * samples)
        # pt_stage = int(0.99 * samples) # paralel tempering is used only for exploration, it does not form the posterior, later mcmc in parallel is used with swaps
        pt_samples = int(
            self.pt_samples * samples)  # paralel tempering is used only for exploration, it does not form the posterior, later mcmc in parallel is used with swaps

        swap_interval = 1  # when to check to swap

        rmse_train = np.zeros((nreplicas, int(samples)))
        rmse_test = np.zeros((nreplicas, int(samples)))

        weight_array = np.zeros((nreplicas, int(samples)))
        weight_array1 = np.zeros((nreplicas, int(samples)))
        weight_array2 = np.zeros((nreplicas, int(samples)))
        weight_array3 = np.zeros((nreplicas, int(samples)))
        weight_array4 = np.zeros((nreplicas, int(samples)))
        sum_value_array = np.zeros((nreplicas, int(samples)))

        tau_pro = np.zeros(nreplicas)

        likelihood = np.zeros(nreplicas)
        prior_current = np.zeros(nreplicas)

        w_proposal = np.zeros(w_size)# proposal for each replica
        old_w_list = [ ] #list to save state dict of all nreplicas (chains)


        for r in range(nreplicas): #initializing all chains

            pred_train = gnn.evaluate_proposal(self.traindata)
            pred_train = pred_train.numpy()
            pred_train = pred_train.ravel()

            y_train = torch.zeros((len(self.traindata), self.batch_size))
            for i, dat in enumerate(self.traindata, 0):
                y_train[i] = dat.y

            y_train = y_train.numpy()
            y_train = y_train.ravel()

            eta = np.log(np.var(pred_train - y_train))
            tau_pro[r] = np.exp(eta)

            w_proposal = np.random.randn(w_size)
            #w_proposal[r,:] = gnn.dictfromlist(w_proposal[r,:])

            prior_current[r] = self.prior_likelihood(sigma_squared, gnn.getparameters(w))

            [likelihood[r], pred_train, rmsetrain] = self.likelihood_func(gnn, self.traindata, tau_pro[r])
            [_, pred_test, rmsetest] = self.likelihood_func(gnn, self.testdata, tau_pro[r])

            rmse_train[r, 0] = rmsetrain
            rmse_test[r, 0] = rmsetest
            likelihood[r] = likelihood[r] * (1.0 / temp_ladder[r])
            old_w = gnn.state_dict()
            old_w_list.append(old_w)

        print(' begin sampling ....')

        old_w=np.array(old_w_list)
        num_accepted = np.zeros(nreplicas)
        langevin_count = np.zeros(nreplicas)

        init_count = 0

        for i in range(int(samples - 1)):

            for r in range(nreplicas):
                self.adapttemp = temp_ladder[r]

                if i < pt_samples:
                    self.adapttemp = temp_ladder[r]

                else:
                    self.adapttemp = 1

                if i == pt_samples and init_count == 0:
                    [likelihood[r], pred_train, rmsetrain] = self.likelihood_func(gnn, self.train, tau_pro, w)
                    [_, pred_test, rmsetest] = self.likelihood_func(gnn, self.test, tau_pro, w)
                    init_count = 1

                lx = np.random.uniform(0, 1, 1)

                if (self.use_langevin_gradients is True) and (lx < self.l_prob):
                    w_gd = gnn.langevin_gradient(self.traindata)
                    w_proposal = gnn.addnoiseandcopy(0, step_w)
                    w_prop_gd = gnn.langevin_gradient(self.traindata)
                    wc_delta = (gnn.getparameters(w) - gnn.getparameters(w_prop_gd))
                    wp_delta = (gnn.getparameters(w_proposal) - gnn.getparameters(w_gd))
                    sigma_sq = step_w
                    first = -0.5 * np.sum(wc_delta * wc_delta) / sigma_sq
                    second = -0.5 * np.sum(wp_delta * wp_delta) / sigma_sq
                    diff_prop = first - second
                    langevin_count = langevin_count + 1

                else:
                    diff_prop = 0
                    w_proposal = gnn.addnoiseandcopy(0, step_w)

                [likelihood_proposal, pred_train, rmsetrain] = self.likelihood_func(gnn, self.traindata, tau_pro[r])
                [likelihood_ignore, pred_test, rmsetest] = self.likelihood_func(gnn, self.testdata, tau_pro[r])

                prior_prop = self.prior_likelihood(sigma_squared, gnn.getparameters(w_proposal))
                diff_likelihood = likelihood_proposal - likelihood[r]
                diff_prior = prior_prop - prior_current

                sum_value = diff_likelihood + diff_prior + diff_prop
                sum_value_array[r, i] = sum_value[0]
                u = np.log(random.uniform(0, 1))

                if u < sum_value[0]:
                    num_accepted[r] = num_accepted[r] + 1
                    likelihood[r] = likelihood_proposal
                    prior_current[r] = prior_prop
                    w = copy.deepcopy(w_proposal) # rnn.getparameters(w_proposal)
                    old_w[r]= w.copy()
                    gnn.loadparameters(old_w[(r+1)%nreplicas]) #loading parameters of next chain
                    print('Chain '+ str(r), i, rmsetrain, rmsetest, 'Accepted')
                    rmse_train[r,i] = rmsetrain
                    rmse_test[r,i] = rmsetest

                else:
                    gnn.loadparameters(old_w[(r + 1) % nreplicas]) #loading parameters of next chain
                    print('Chain ' + str(r), i, rmsetrain, rmsetest, 'Rejected')
                    rmse_train[r,i] = rmse_train[r,i-1]
                    rmse_test[r,i] = rmse_test[r,i-1]

                ll = gnn.getparameters()
                weight_array[r,i] = ll[0]
                weight_array1[r,i] = ll[100]
                weight_array2[r,i] = ll[5000]
                weight_array3[r,i] = ll[10000]

            print
            ' time to check swap --------------------------------------- *'

            for s in range(1, nreplicas):

                swap_proposed=swap_proposed+1
                lhood1 = likelihood[s - 1]
                lhood2 = likelihood[s]

                swap_proposal = min(1, 0.5*math.exp(lhood2 - lhood1))

                u = np.random.uniform(0, 1)

                if u < swap_proposal:
                    temp = old_w[s - 1]
                    old_w[s - 1] = old_w[s].copy()
                    old_w[s] = temp.copy()
                    successful_swap=successful_swap+1

            gnn.loadparameters(old_w[0]) #in case of swap, need to make sure that the first chain's parameters are loaded.

        accept_percentage = np.sum(num_accepted) / (self.samples * 1.0) * 100

        print(accept_percentage, '% was accepted')
        swap_percentage = successful_swap / (swap_proposed * 1.0) * 100

        print(swap_percentage, '% was swapped')
        #return (rep_diffscore, accept_ratio, posterior, predcore_list, x_data, y_data, data_vec, rep_acceptlist,
                #rep_likelihoodlist, diffscore, total_time / 3600)

def make_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def main():

    samples = args.samples
    num_chains = args.num_chains
    swap_ratio = args.swap_ratio
    burn_in = args.burn_in
    learning_rate = args.learning_rate
    step_size = args.step_size
    max_temp = 2
    use_langevin_gradients = True  # False leaves it as Random-walk proposals. Note that Langevin gradients will take a bit more time computationally
    swap_interval = int(swap_ratio * samples / num_chains) # int(swap_ratio * (NumSample/num_chains)) #how ofen you swap neighbours. note if swap is more than Num_samples, its off
    # learn_rate = 0.01  # in case langevin gradients are used. Can select other values, we found small value is ok.

    path = 'Graph_torch/GNN' # change this to your directory for results output - produces large datasets
    pt_samples = 0.6

    mcmc = MCMC(use_langevin_gradients, learning_rate, samples, burn_in, swap_interval, path, batch_size, step_size,
                num_chains, max_temp, pt_samples)

    mcmc.sampler()

if __name__ == "__main__": main()
