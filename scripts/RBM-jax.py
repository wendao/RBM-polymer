# -*- coding: utf-8 -*-

import sys, os, argparse
from math import sqrt, log
import jax
import jax.numpy as jnp
import numpy as np
from jax import random, grad, jit, vmap
from functools import partial

def int2xy(conf):
    L = len(conf)
    x = np.zeros([L+2, 2])
    x[0,0] = 0
    x[0,1] = 0
    x[1,0] = 1
    x[1,1] = 0
    for i in range(L):
        if conf[i] == 1:
            #forward
            x[i+2,0] = 2*x[i+1,0]-x[i,0]
            x[i+2,1] = 2*x[i+1,1]-x[i,1]
        elif conf[i] == 3:
            #back
            x[i+2,0] = x[i,0]
            x[i+2,1] = x[i,1]
        else:
            dx1 = x[i+1,0] - x[i,0]
            dy1 = x[i+1,1] - x[i,1]
            if conf[i] == 2:
                #right
                if dx1 == 0:
                    dy = 0
                    if dy1>0: dx = -1
                    else: dx = 1
                elif dy1 == 0:
                    dx = 0
                    if dx1>0: dy = 1
                    else: dy = -1
                else:
                    print("error!")
            elif conf[i] == 0:
                #left
                if dx1 == 0:
                    dy = 0
                    if dy1>0: dx = 1
                    else: dx = -1
                elif dy1 == 0:
                    dx = 0
                    if dx1>0: dy = -1
                    else: dy = 1
                else:
                    print("error!")
            x[i+2,0] = x[i+1,0] + dx
            x[i+2,1] = x[i+1,1] + dy
    return x

def cal_rg2(xy):
    cm = np.sum(xy, axis=0) / xy.shape[0]
    return np.mean(np.sum((xy-cm)**2, axis=1))

def get_data(fn, n_vis, n_val):
    """
    Load or generate data for the RBM, process it on CPU, and then transfer to device.
    This is more efficient than element-wise operations on GPU.
    """
    
    if fn.isdigit():
        # Generate random data on CPU using numpy
        num_data = int(fn)
        sys.stderr.write(f"Generating random data: nv= {n_vis}, nd= {num_data}\n")
        
        # Generate random categories directly
        np_rng = np.random.RandomState(42)  # Fixed seed for reproducibility
        dat_lst = np_rng.randint(0, n_val, size=(num_data, n_vis))
    else:
        # Read from file
        with open(fn, 'r') as fp:
            lines = fp.readlines()
            es = lines[0].split()
            nl = int(es[0])  # number of lines (samples)
            nv = int(es[1])  # number of visible units per sample
            sys.stderr.write(f"Loading data: nv= {nv}, nd= {nl}\n")
            
            # Pre-allocate array for data
            dat_lst = np.zeros((nl, n_vis), dtype=np.int32)
            
            # Parse file data
            for l in range(nl):
                line = lines[l+1]  # +1 to skip header
                for i in range(min(n_vis, len(line))):
                    dat_lst[l, i] = int(line[i])
    
    # Create one-hot encoding efficiently using numpy operations
    # This is much faster than doing it sample by sample
    
    # Create the output array directly with the right shape
    one_hot_data = np.zeros((dat_lst.shape[0], n_vis * n_val), dtype=np.float32)
    
    # Fill in the one-hot values
    for i in range(n_vis):
        vals = dat_lst[:, i]
        # For each position, set the corresponding one-hot value to 1
        for j in range(len(vals)):
            val = vals[j]
            if val < n_val:  # Safety check
                one_hot_data[j, i * n_val + val] = 1.0
    
    # Return numpy array - we'll convert to JAX array at the point of use
    return one_hot_data

class RBM:
    def __init__(self, n_vis=2, n_hid=1, n_val=2, seed=0):
        self.n_vis = n_vis  # num of units in visible (input) layer
        self.n_hid = n_hid  # num of units in hidden layer
        self.n_val = n_val  # num of values for each node
        self.n_node = n_vis * n_val # each visnode has n_val nodes
        
        self.key = random.PRNGKey(seed)
        key1, key2, self.key = random.split(self.key, 3)
        
        # Initialize parameters
        a = sqrt(6. / (self.n_vis * n_val + n_hid))
        self.W = random.uniform(key1, shape=(self.n_node, n_hid), minval=-a, maxval=a)
        self.hb = jnp.zeros(n_hid)
        self.vb = jnp.zeros(self.n_node)
        
        # For momentum
        self.dW = jnp.zeros((self.n_node, n_hid))
        self.dh = jnp.zeros(n_hid)
        self.dv = jnp.zeros(self.n_node)
        
        # For KL
        self.enum_states = None
        self.prob_states = None
        self.prob_RGs = None
        
        self.L_coeff = 0.0
        self.M_coeff = 0.0
        
        self.no_softmax = False
        
        # Device info
        self.device = jax.devices()[0]  # Default to first available device
    
    def set_device(self, device):
        """Set the device to use for computation"""
        self.device = device
        # Move parameters to device
        self.W = jax.device_put(self.W, device)
        self.hb = jax.device_put(self.hb, device)
        self.vb = jax.device_put(self.vb, device)
        self.dW = jax.device_put(self.dW, device)
        self.dh = jax.device_put(self.dh, device)
        self.dv = jax.device_put(self.dv, device)
    
    @partial(jit, static_argnums=(0,))
    def sigmoid(self, x):
        return 1.0 / (1.0 + jnp.exp(-x))
    
    @partial(jit, static_argnums=(0,))
    def propup(self, v):
        pre_sigmoid_activation = jnp.dot(v, self.W) + self.hb
        return self.sigmoid(pre_sigmoid_activation)
    
    @partial(jit, static_argnums=(0,))
    def propdown(self, h):
        return jnp.dot(h, self.W.T) + self.vb
    
    def sample_h_given_v(self, v0, key):
        h1_prob = self.propup(v0)
        key, subkey = random.split(key)
        h1 = h1_prob > random.uniform(subkey, shape=h1_prob.shape)
        return h1_prob, h1, key
    
    def sample_v_given_h(self, h0, key):
        v1_logits = self.propdown(h0).reshape(-1, self.n_val)
        key, subkey = random.split(key)
        
        if self.no_softmax:
            v1_prob = self.sigmoid(v1_logits.reshape(-1, self.n_node))
            v1 = v1_prob > random.uniform(subkey, shape=v1_prob.shape)
            return v1_prob, v1, key
        else:
            # Apply softmax and sample
            v1_prob = jax.nn.softmax(v1_logits, axis=1)
            v1_sample = random.categorical(subkey, v1_logits)
            v1 = jax.nn.one_hot(v1_sample, self.n_val)
            return v1_prob.reshape(-1, self.n_node), v1.reshape(-1, self.n_node), key
    
    def gibbs_hvh(self, h0, key):
        v1_prob, v1, key = self.sample_v_given_h(h0, key)
        h1_prob, h1, key = self.sample_h_given_v(v1, key)
        return v1_prob, v1, h1_prob, h1, key
    
    def contrastive_divergence(self, input_data, lr=0.1, cdk=1, batch_size=None, shuffle=False):
        # If input is numpy, convert to JAX array on device
        if isinstance(input_data, np.ndarray):
            input_data = jax.device_put(input_data, self.device)
            
        n_sample = input_data.shape[0]
        if batch_size is None or batch_size == 0:
            batch_size = n_sample
        
        # Create batches
        if shuffle:
            key_shuffle, self.key = random.split(self.key)
            indices = random.permutation(key_shuffle, jnp.arange(n_sample))
            input_data = input_data[indices]
        
        n_batches = n_sample // batch_size
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            batch_data = input_data[start_idx:end_idx]
            
            # Positive phase
            ph_prob, ph_sample, self.key = self.sample_h_given_v(batch_data, self.key)
            
            # Negative phase (CD-k)
            chain_h = ph_sample
            for step in range(cdk):
                nv_prob, nv_sample, nh_prob, chain_h, self.key = self.gibbs_hvh(chain_h, self.key)
            
            # Compute gradients
            if self.M_coeff > 0:
                # Apply momentum
                self.dW = self.M_coeff * self.dW + (jnp.dot(batch_data.T, ph_prob) - jnp.dot(nv_sample.T, nh_prob)) * lr / batch_size
                self.dv = self.M_coeff * self.dv + jnp.mean(batch_data - nv_sample, axis=0) * lr
                self.dh = self.M_coeff * self.dh + jnp.mean(ph_prob - nh_prob, axis=0) * lr
            else:
                self.dW = (jnp.dot(batch_data.T, ph_prob) - jnp.dot(nv_sample.T, nh_prob)) * lr / batch_size
                self.dv = jnp.mean(batch_data - nv_sample, axis=0) * lr
                self.dh = jnp.mean(ph_prob - nh_prob, axis=0) * lr
            
            # Update parameters
            self.W += self.dW
            self.vb += self.dv
            self.hb += self.dh
            
            # Apply weight decay
            self.W_decay(lr)
    
    def W_decay(self, lr):
        if self.L_coeff > 0:  # L2 regularization
            self.W -= self.L_coeff * lr * self.W
        elif self.L_coeff < 0:  # L1 regularization
            self.W += self.L_coeff * lr * jnp.sign(self.W)
        else:
            # Clip weights
            self.W = jnp.clip(self.W, -10.0, 10.0)
    
    @partial(jit, static_argnums=(0,))
    def get_free_energy(self, v):
        x = jnp.dot(v, self.W) + self.hb
        vt = jnp.dot(v, self.vb)
        ht = jnp.sum(jnp.log(1.0 + jnp.exp(x)), axis=1)
        fe = -ht - vt
        return jnp.mean(fe)
    
    def check_status(self, input_data, epoch):
        # Ensure data is on device
        if isinstance(input_data, np.ndarray):
            input_data = jax.device_put(input_data, self.device)
            
        # Reconstruct and compute metrics
        ph_prob, ph_sample, key = self.sample_h_given_v(input_data, self.key)
        nv_prob, nv_sample, nh_prob, nh_sample, key = self.gibbs_hvh(ph_sample, key)
        
        # Reconstruction error
        error = jnp.sum((input_data - nv_sample) ** 2) / input_data.shape[0]
        
        # Cross entropy
        cross = -jnp.mean(jnp.sum(input_data * jnp.log(jnp.clip(nv_prob, 1e-10, 1.0)), axis=1))
        
        # Free energy
        free_energy = self.get_free_energy(input_data)
        
        sys.stdout.write("Training: ")
        sys.stdout.write(f"epoch= {epoch} ")
        sys.stdout.write(f"cross= {float(cross)} ")
        sys.stdout.write(f"error= {float(error)} ")
        sys.stdout.write(f"freeE= {float(free_energy)} ")
        
        if self.enum_states is not None:
            sys.stdout.write(f"KL= {self.check_KL()} ")
        if self.prob_RGs is not None:
            sys.stdout.write(f"rgKL= {self.check_rgKL(nv_sample)} ")
        
        sys.stdout.write("\n")
        return
    
    def check_KL(self):
        if self.enum_states is None or self.prob_states is None:
            return 0.0
        
        # Ensure states are on device
        enum_states = jax.device_put(self.enum_states, self.device)
        prob_states = jax.device_put(self.prob_states, self.device)
        
        ph_act = jnp.dot(enum_states, self.W) + self.hb
        vt = jnp.dot(enum_states, self.vb)
        ht = jnp.sum(-jnp.log(self.sigmoid(-ph_act)), axis=1)
        p_th = jax.nn.softmax(vt + ht)
        KL = jnp.sum(prob_states * jnp.log(prob_states / p_th))
        
        return float(KL)
    
    def check_rgKL(self, v):
        if self.prob_RGs is None:
            return 0.0
            
        # Convert to numpy for processing
        v_np = np.array(v)
        v_reshape = v_np.reshape(-1, self.n_val)
        ndx = np.argmax(v_reshape, axis=1)
        ndx_np = ndx.reshape(-1, self.n_vis)
        
        pv = np.ones(shape=np.array(self.prob_RGs).shape)
        for intcoor in ndx_np:
            rg2 = cal_rg2(int2xy(intcoor))
            i = int(rg2 * 2)
            if i >= len(pv): 
                i = len(pv) - 1
            pv[i] += 1
        
        pv /= np.sum(pv)
        prg = np.array(self.prob_RGs)
        KL = np.sum(prg * np.log(prg / pv))
        
        return KL
    
    def load_enum_states(self, fn):
        lines = open(fn, 'r').readlines()
        n_states = int(lines[0])
        dat_lst = []
        
        prob_states_list = []
        
        for i in range(1, n_states+1):
            es = lines[i].strip().split()
            for v in range(self.n_vis):
                dat_lst.append(int(es[0][v]))
            prob = float(es[1])
            if prob < 1e-10:
                prob = 1e-10
            prob_states_list.append(prob)
        
        # Create numpy arrays first
        self.prob_states = np.array(prob_states_list)
        dat_arr = np.array(dat_lst)
        
        # Create one-hot encoding using numpy
        enum_states_list = []
        for i in range(0, len(dat_arr), self.n_vis):
            one_hot_state = np.zeros(self.n_node)
            for j in range(min(self.n_vis, len(dat_arr) - i)):
                val = dat_arr[i + j]
                if val < self.n_val:  # Safety check
                    one_hot_state[j * self.n_val + val] = 1.0
            enum_states_list.append(one_hot_state)
        
        self.enum_states = np.array(enum_states_list)
        
        # Transfer to device only when needed in check_KL
        sys.stderr.write("Exact states info loaded!\n")
        return
    
    def load_enum_RGs(self, fn):
        lines = open(fn, 'r').readlines()
        n_states = int(lines[0].split()[0])
        
        prob_RGs_list = []
        
        for i in range(1, n_states+1):
            es = lines[i].strip().split()
            prob = float(es[1]) + 1e-10  # log(1/20000)
            assert(int(float(es[0])*2) == i-1)
            prob_RGs_list.append(prob)
            
        self.prob_RGs = np.array(prob_RGs_list)  # Keep as numpy array
        sys.stderr.write("Rg info from obs data loaded!\n")
        return
    
    def load(self, fn):
        lines = open(fn, 'r').readlines()
        elems = lines[-1].split()
        last_epoch = int(elems[0])
        pos = 1
        
        # Extract weights using numpy first for efficiency
        w_values = np.zeros((self.n_node, self.n_hid))
        for i in range(self.n_node):
            for j in range(self.n_hid):
                w_values[i, j] = float(elems[pos])
                pos += 1
        
        # Extract visible biases
        vb_values = np.zeros(self.n_node)
        for i in range(self.n_node):
            vb_values[i] = float(elems[pos])
            pos += 1
        
        # Extract hidden biases
        hb_values = np.zeros(self.n_hid)
        for j in range(self.n_hid):
            hb_values[j] = float(elems[pos])
            pos += 1
        
        # Transfer to JAX arrays on device
        self.W = jax.device_put(w_values, self.device)
        self.vb = jax.device_put(vb_values, self.device)
        self.hb = jax.device_put(hb_values, self.device)
        
        sys.stderr.write(f"Loading weights and restart from epoch={last_epoch}\n")
        return last_epoch
    
    def save(self, fn, epoch=0):
        # Convert to numpy for saving
        W_np = np.array(self.W)
        vb_np = np.array(self.vb)
        hb_np = np.array(self.hb)
        
        with open(fn, 'a') as fp:
            fp.write(f"{epoch} ")
            for i in range(self.n_node):
                for j in range(self.n_hid):
                    fp.write(f"{W_np[i,j]:.4f} ")
            for i in range(self.n_node):
                fp.write(f"{vb_np[i]:.4f} ")
            for j in range(self.n_hid):
                fp.write(f"{hb_np[j]:.4f} ")
            fp.write("\n")
        return
    
    def daydream(self, n, k=1):
        sys.stderr.write("Day dreaming ...\n")
        key_h, self.key = random.split(self.key)
        h0 = random.uniform(key_h, shape=(1, self.n_hid))
        h0 = jax.device_put(h0, self.device)
        
        # Burn-in period
        for i in range(100):
            v_prob, v, key = self.sample_v_given_h(h0, self.key)
            h_prob, h0, key = self.sample_h_given_v(v, key)
            self.key = key
        
        # Generate samples
        for i in range(n):
            for cdk in range(k):
                v_prob, v, key = self.sample_v_given_h(h0, self.key)
                h_prob, h0, key = self.sample_h_given_v(v, key)
                self.key = key
            
            # Get the most probable value for each visible node
            v_reshape = v.reshape(-1, self.n_val)
            ndx = jnp.argmax(v_reshape, axis=1)
            ndx_np = np.array(ndx)  # Convert to numpy for printing
            for n in ndx_np:
                sys.stdout.write(str(int(n)))
            sys.stdout.write("\n")
        return
    
    def yeardream(self, v, n, k=1):
        sys.stderr.write("Year dreaming ...\n")
        # Ensure data is on device
        if isinstance(v, np.ndarray):
            v = jax.device_put(v, self.device)
            
        # Burn-in period
        for i in range(20):
            h_prob, h0, key = self.sample_h_given_v(v, self.key)
            v_prob, v, key = self.sample_v_given_h(h0, key)
            self.key = key
        
        # Generate samples
        for i in range(n):
            for cdk in range(k):
                h_prob, h0, key = self.sample_h_given_v(v, self.key)
                v_prob, v, key = self.sample_v_given_h(h0, key)
                self.key = key
            
            # Get the most probable value for each visible node
            v_reshape = v.reshape(-1, self.n_val)
            ndx = jnp.argmax(v_reshape, axis=1)
            ndx_np = np.array(ndx)  # Convert to numpy for easy handling
            
            for j, n in enumerate(ndx_np):
                sys.stdout.write(str(int(n)))
                if j % self.n_vis == (self.n_vis-1):
                    sys.stdout.write("\n")
        return

def run_rbm():
    # Setup options
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", '--learning_rate', type=float, default=0.1, help="learning rate")
    parser.add_argument("-k", '--cdk', type=int, default=1, help="CD-k steps")
    parser.add_argument("-d", '--data', type=str, default="0", help="number/file of data")
    parser.add_argument("-t", '--train', type=int, default=0, help="training steps")
    parser.add_argument("-g", '--generate', type=int, default=0, help="generate steps")
    parser.add_argument("-b", '--batch_size', type=int, default=0, help="batch size")
    parser.add_argument("-v", '--visible', type=int, default=1, help="visible node")
    parser.add_argument("-n", '--hidden', type=int, default=1, help="hidden node")
    parser.add_argument("-l", '--level', type=int, default=2, help="level of vis node")
    parser.add_argument("-w", '--weight', type=str, help="R/W weights parameter")
    parser.add_argument("-i", '--interval', type=int, default=10, help="interval for output")
    parser.add_argument("-s", '--rescaleL', type=float, default=0.0, help="L1(<0)/L2(>0) type weights decay")
    parser.add_argument("-m", '--momentum', type=float, default=0.0, help="momentum coefficient")
    parser.add_argument("-x", '--gpu', type=int, default=-1, help="using GPU x (>=0) or CPU (-1)")
    parser.add_argument("-kl", '--KL', type=str, help="check KL")
    parser.add_argument("-rg", '--rgKL', type=str, help="check KL of RG")
    parser.add_argument("-no", '--nosoftmax', action="store_true", default=False, help="training with independent P_state")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    
    args = parser.parse_args()
    learning_rate = args.learning_rate
    training_epochs = args.train
    gen_steps = args.generate
    bs = args.batch_size
    k = args.cdk
    num_vis = args.visible
    num_hid = args.hidden
    num_val = args.level
    
    # Determine device to use
    if args.gpu >= 0:
        try:
            # Try to get the specified GPU
            devices = jax.devices('gpu')
            if args.gpu < len(devices):
                device = devices[args.gpu]
                sys.stderr.write(f"Using GPU {args.gpu}: {device}\n")
            else:
                # If specified GPU not available, use first GPU
                device = devices[0]
                sys.stderr.write(f"GPU {args.gpu} not found. Using GPU 0: {device}\n")
        except:
            # If GPU not available, fall back to CPU
            device = jax.devices('cpu')[0]
            sys.stderr.write("GPU requested but not available. Using CPU instead.\n")
    else:
        # CPU requested
        device = jax.devices('cpu')[0]
        sys.stderr.write("Using CPU\n")
    
    # Load or generate data on CPU
    data_np = None
    if args.data != "0":
        data_np = get_data(args.data, num_vis, num_val)
    
    # Construct RBM
    sys.stderr.write(f"Creating RBM: nv= {num_vis}, nl= {num_val}, nh= {num_hid}\n")
    rbm = RBM(n_vis=num_vis, n_hid=num_hid, n_val=num_val, seed=args.seed)
    rbm.set_device(device)  # Set device for computation
    rbm.L_coeff = args.rescaleL
    rbm.M_coeff = args.momentum
    rbm.no_softmax = args.nosoftmax
    
    if rbm.no_softmax:
        sys.stderr.write("No softmax\n")
    else:
        sys.stderr.write("Do softmax\n")
    
    # Load weights
    epoch_start = 0
    if args.weight is not None:
        if os.path.isfile(args.weight):
            epoch_start = rbm.load(args.weight)
    
    # Load KL reference
    if args.KL is not None: 
        rbm.load_enum_states(args.KL)
    if args.rgKL is not None: 
        rbm.load_enum_RGs(args.rgKL)
    
    # Train
    if training_epochs > 0 and data_np is not None:
        if epoch_start == 0: 
            rbm.check_status(data_np, 0)
        for epoch in range(epoch_start, training_epochs):
            rbm.contrastive_divergence(
                input_data=data_np, 
                lr=learning_rate, 
                cdk=k,
                batch_size=bs, 
                shuffle=(epoch > 0)
            )
            if (epoch+1) % args.interval == 0:
                rbm.check_status(data_np, epoch+1)
                sys.stdout.flush()
                sys.stderr.flush()
                # Save weights (append), in one line
                if args.weight is not None: 
                    rbm.save(args.weight, epoch+1)
    
    if gen_steps > 0:
        if data_np is not None:
            rbm.yeardream(data_np, gen_steps, k)
        else:
            rbm.daydream(gen_steps, k)

if __name__ == "__main__":
    run_rbm()

