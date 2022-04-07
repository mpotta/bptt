import numpy as np
import scipy.signal as signal
import cmath
import sys
import csv

# Set Parameters
# filter_order = N = 5
# hidden_size = 10

seq_len = 150
wash_out = 50
batch_size = 100
split_ratio = 0.8

import torch
from torch import nn

class RNNLayer(nn.Module):
    """Linear RNN.

    Parameters:
        input_size: Number of input neurons
        hidden_size: Number of hidden neurons

    Inputs:
        input: tensor of shape (seq_len, batch, input_size)
        hidden: tensor of shape (batch, hidden_size), initial hidden activity
                if None, hidden is initialized through self.init_hidden()
        
    Outputs:
        output: tensor of shape (seq_len, batch, hidden_size)
        hidden: tensor of shape (batch, hidden_size), final hidden activity
    """

    def __init__(self, input_size, hidden_size, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size, bias=False)

    def init_hidden(self, input_shape):
        batch_size = input_shape[1]
        return torch.zeros(batch_size, self.hidden_size)

    def recurrence(self, input, hidden):
        """Run network for one time step.
        
        Inputs:
            input: tensor of shape (batch, input_size)
            hidden: tensor of shape (batch, hidden_size)
        
        Outputs:
            h_new: tensor of shape (batch, hidden_size),
                network activity at the next time step
        """
        h_new = torch.tanh(self.i2h(input) + self.h2h(hidden))
        return h_new

    def forward(self, input, hidden=None):
        """Propogate input through the network."""
        
        # If hidden activity is not provided, initialize it
        if hidden is None:
            hidden = self.init_hidden(input.shape).to(input.device)

        # Loop through time
        output = []
        steps = range(input.size(0))
        for i in steps:
            hidden = self.recurrence(input[i], hidden)
            output.append(hidden)

        # Stack together output from all time steps
        output = torch.stack(output, dim=0)
        return output, hidden


class RNNNet(nn.Module):
    """Recurrent network model.

    Parameters:
        input_size: int, input size
        hidden_size: int, hidden size
        output_size: int, output size
    
    Inputs:
        x: tensor of shape (Seq Len, Batch, Input size)

    Outputs:
        out: tensor of shape (Seq Len, Batch, Output size)
        rnn_output: tensor of shape (Seq Len, Batch, Hidden size)
    """
    def __init__(self, input_size, hidden_size, output_size, **kwargs):
        super().__init__()

        self.rnn = RNNLayer(input_size, hidden_size, **kwargs)
        self.fc = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, x):
        rnn_output, _ = self.rnn(x)
        out = self.fc(rnn_output)
        return out, rnn_output

def sample_poles(filter_order):
    """
    Systems represented by stable right-sided sequences have all poles located within the unit circle, which suggests all eigenvalues of the system have magnitude less than one and thus state variables decay over time.
    """
    centerX = centerY = 0
    R = 1
    z = [0] * filter_order
    p = []
    for _ in range(filter_order // 2):
        r = R * np.sqrt(np.random.rand())
        theta = np.random.rand() * 2 * np.pi
        x = centerX + r * np.cos(theta)
        y = centerY + r * np.sin(theta)
        t = x + y * 1j

        p.append(t)
        p.append(t.conjugate())

    if filter_order%2 == 1:
        p.append(np.random.rand())
    return z, p

def spectral_normalization(W, threshold=0.9):
    eigen_values = torch.linalg.eigvals(W)
    spectral_radius = max(abs(eigen_values))
    if spectral_radius > threshold:
        W = W/max(abs(eigen_values))
    return W

def jacobian(network, x, rnn_state):
    """Compute the Jacobian of an RNN state vector h(t+1) with respect to h(t) for each t."""
    jac_stacked = []
    hidden = rnn_state[-1]
    feed_seq = x.shape[0]
    
    for t in range(feed_seq):
        hidden = network.rnn.recurrence(x[t], hidden)
        W_hh = network.rnn.h2h.weight
        W_diag = torch.diag(1-torch.tanh(torch.pow(torch.flatten(hidden),2)))

        jac_t = torch.matmul(W_hh, W_diag)  # [N, N]
        jac_stacked.append(jac_t)

    # Stack together output from all time steps
    return torch.stack(jac_stacked)

def extract_eigen_values(M):
    eW = np.linalg.eigvals(M)
    eW = np.flip(np.sort(eW, axis=1), axis=1)
    rea = [n.real for n in eW]
    imag = [n.imag for n in eW]
    return rea, imag

def log(x):
	return np.array([cmath.log(xx) for xx in x])

def findClosestPair(real_pole, imag_pole, W_bptt_set):
    min_euclidean_dist = sys.maxsize
    a = np.array((real_pole ,imag_pole))
    for pair in W_bptt_set:
        b = np.array((pair[0], pair[1]))
        euclidean_dist = np.linalg.norm(a-b)
        if euclidean_dist < min_euclidean_dist:
            min_euclidean_dist = euclidean_dist
            closest_pair = pair
    return min_euclidean_dist, closest_pair

def compute_similarity(threshold, real_poles, imag_poles, realW_bptt, imagW_bptt):
    count = 0
    W_bptt_set = [[realW_bptt[i], imagW_bptt[i]] for i in range(len(realW_bptt))]
    for real_pole, imag_pole in zip(real_poles, imag_poles):
        min_euclidean_dist, closest_pair = findClosestPair(real_pole, imag_pole, W_bptt_set)
        if min_euclidean_dist <= threshold:
            count += 1
        W_bptt_set.remove(closest_pair)
    return count

def compute_similarity_t(W_filter_stacked, W_bptt_stacked, seq_len, N, threshold=0.05):
    count_stacked = []
    euclidean_dist_stacked = []
    largest_exp_dist_stacked = []
    for i in range(seq_len):
        flag = 1
        count = 0
        euclidean_dist = 0
        W_bptt_r = W_bptt_stacked[0][i]
        W_bptt_i = W_bptt_stacked[1][i]
        W_bptt_set = [[W_bptt_r[j], W_bptt_i[j]] for j in range(len(W_bptt_r))]
        for real_pole, imag_pole in zip(W_filter_stacked[0][i], W_filter_stacked[1][i]):
            min_euclidean_dist, closest_pair = findClosestPair(real_pole, imag_pole, W_bptt_set)
            if flag == 1:
                largest_exp_dist = min_euclidean_dist
                flag = 0
            euclidean_dist += min_euclidean_dist
            if min_euclidean_dist <= threshold:
                count += 1
            W_bptt_set.remove(closest_pair)

        count_stacked.append(count)
        euclidean_dist_stacked.append(euclidean_dist/N)
        largest_exp_dist_stacked.append(largest_exp_dist)

    return count_stacked, euclidean_dist_stacked, largest_exp_dist_stacked

def run_network(N_current, hidden_dim):
    N = N_current

    # Generate Data
    z, p = sample_poles(N)
    b, a = signal.zpk2tf(z, p, 1)

    # Instantiate Model
    net = RNNNet(1, N, 1)

    # Vector with a "1" for input x
    i2h = torch.zeros((N,1))
    i2h[0,0] = torch.tensor(1.0)
    net.rnn.i2h.weight = nn.Parameter(i2h)

    # Shifted diagonal matrix and IIR filter coefficients
    a_copy = -a
    a_copy[0] = 1
    h2h = torch.diag(torch.ones(N-1), diagonal=-1)
    h2h[0,:] = torch.from_numpy(a_copy[1:])
    net.rnn.h2h.weight = nn.Parameter(h2h)

    # Weight for linear output layer to select h[0]
    fc = torch.zeros((1,N))
    fc[0][0] = torch.tensor(1.0)
    net.fc.weight = nn.Parameter(fc) 

    # Create input and output of shape (batch_size x seq_len x input_dims) 
    def create_dataset(batch_size, seq_len):
        X = np.random.normal(0, 1, size=(seq_len, batch_size, 1))
        X = torch.from_numpy(X).type(torch.float)
        Y, _ = net(X)
        return X, Y

    X, Y = create_dataset(batch_size, seq_len)

    split_index = int(split_ratio*batch_size)
    train_X, train_Y = X[:,:split_index,:], Y[:,:split_index,:]
    test_X, test_Y = X[:,split_index:,:], Y[:,split_index:,:]

    from torch.autograd import Variable

    net_trained = RNNNet(1, hidden_dim, 1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net_trained.parameters(), lr=0.01)

    loss_train = []
    epochs = 5000
    for epoch in range(epochs):
        # Zero the gradient buffer
        optimizer.zero_grad()
        
        # Forward pass
        train_inp = Variable(train_X)
        train_out = Variable(train_Y)
        pred, _ = net_trained(train_inp)

        # Measure the loss
        loss = criterion(pred[:,wash_out:,:], train_out[:,wash_out:,:])
        loss_train.append(loss.data.item())

        # Print the loss
        if epoch%50==0:
            print(epoch, loss.data.item())

        # Backward pass
        loss.backward()
        optimizer.step()

    # Predict
    pred, rnn_output = net_trained(test_X)

    # Compute Loss
    loss_test = criterion(pred[wash_out:,:,:], test_Y[wash_out:,:,:]).detach().numpy()
    print(loss_test)

    feed_seq = 100
    sample_id = 0

    # Long-term Jacobian for the filter coefficients
    _, rnn_state = net(test_X[0:wash_out,sample_id:sample_id+1,:])
    jac_stacked_filter = jacobian(net, test_X[wash_out:wash_out+feed_seq,sample_id:sample_id+1,:], rnn_state)

    # Long-term Jacobian for the bptt-rnn
    _, rnn_state = net_trained(test_X[0:wash_out,sample_id:sample_id+1,:])
    jac_stacked_bptt = jacobian(net_trained, test_X[wash_out:wash_out+feed_seq,sample_id:sample_id+1,:], rnn_state)

    W_filter_stacked = extract_eigen_values(jac_stacked_filter.detach().numpy())
    W_bptt_stacked = extract_eigen_values(jac_stacked_bptt.detach().numpy())
    count_stacked, euclidean_dist_stacked, largest_exp_dist_stacked = compute_similarity_t(W_filter_stacked, W_bptt_stacked, feed_seq, N)

    real_poles, imag_poles = [x.real for x in p], [x.imag for x in p]

    return {
        "N": N,
        "hidden_dims": hidden_dim,
        "poles": str([real_poles, imag_poles]),
        "count_stacked": str(count_stacked),
        "euclidean_dist_stacked": str(euclidean_dist_stacked),
        "largest_exp_dist_stacked": str(largest_exp_dist_stacked),
    }

def main():
    trials = 20
    N_max = 15

    f = open("bulk_run_"+str(trials)+".csv", "w")
    fieldnames = ['N', 'trial_number', 'hidden_dim', 'count_stacked', 'euclidean_dist_stacked', 'largest_exp_dist_stacked']
    writer = csv.DictWriter(f, fieldnames=fieldnames)

    for N_current in range(3, N_max+1, 2):
        for i in range(trials):
            for hidden_dim in range(N_current,N_current*2,N_current*5):
                result = run_network(N_current, hidden_dim)
                writer.writerow({
                    'N':result['N'],
                    'trial_number':i,
                    'hidden_dim':result['hidden_dims'],
                    'count_stacked':result['count_stacked'],
                    'euclidean_dist_stacked':result['euclidean_dist_stacked'],
                    'largest_exp_dist_stacked':result['largest_exp_dist_stacked'],
                    })
    f.close()

if __name__ == "__main__":
  main()





