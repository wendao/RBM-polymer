# -*- coding: utf-8 -*-
#
# PyTorch port of RBM.py (originally written for mxnet 1.0.0 / Python 2.7).
# Behaviour is kept identical to the mxnet version: same CD-k algorithm,
# same Gibbs sampling, same free-energy / KL diagnostics, same weight
# save/load text format and the same command-line interface.
#
# Notes on the mxnet -> torch mapping (see comments inline as well):
#   nd.dot(a, b)                 -> a @ b            (2D matmul)
#   nd.random_uniform(lo,hi,sh)  -> torch.empty(sh).uniform_(lo,hi)
#   nd.sigmoid / log / exp       -> torch.sigmoid / log / exp
#   nd.softmax(x)                -> torch.softmax(x, dim=-1)   (mxnet default axis=-1)
#   nd.sample_multinomial(p)     -> torch.multinomial(p, 1) (one draw per row)
#   nd.one_hot(idx, depth)       -> F.one_hot(idx, depth).float()
#   (a > b) returns 1.0/0.0      -> (a > b).float()  (mxnet returns floats)
#   nd.clip(x, lo, hi)           -> torch.clamp(x, lo, hi)
#   x.asnumpy()                  -> x.detach().cpu().numpy()
#   x.asnumpy()[0]  (scalar)     -> x.item()
#
# Requirements: Python>=3.7, torch (any recent version, CPU is fine on macOS).

import sys, os, argparse, time
from math import sqrt, log
import torch
import torch.nn.functional as F
import numpy as np


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

class RBM(object):
    def __init__(self, n_vis=2, n_hid=1, n_val=2, ctx=torch.device("cpu")):
        self.n_vis = n_vis  # num of units in visible (input) layer
        self.n_hid = n_hid  # num of units in hidden layer
        self.n_val = n_val  # num of values for each node
        self.n_node = n_vis * n_val # each visnode has n_val nodes

        self.ctx = ctx

        a = sqrt(6. / ( self.n_vis * n_val + n_hid ))

        self.W = torch.empty((self.n_node, n_hid), device=self.ctx).uniform_(-a, a)
        self.hb = torch.zeros(n_hid, device=self.ctx)        # initialize h bias 0
        self.vb = torch.zeros(self.n_node, device=self.ctx)  # initialize v bias 0
        self.dW = torch.zeros((self.n_node, n_hid), device=self.ctx)
        self.dh = torch.zeros(n_hid, device=self.ctx)
        self.dv = torch.zeros(self.n_node, device=self.ctx)

        #for KL
        self.enum_states = None
        self.prob_states = None
        self.prob_RGs = None

        self.L_coeff = 0.0
        self.M_coeff = 0.0

    def contrastive_divergence(self, input, lr=0.1, cdk=1, batch_size=None, shuffle=False):
        n_sample = input.shape[0]
        if batch_size == 0: batch_size = n_sample

        # Replicates mx.io.NDArrayIter(..., last_batch_handle='discard'):
        # iterate in fixed-size batches, optionally shuffled, drop the last
        # incomplete batch.
        # Build the index tensor on CPU (randperm is not supported on the MPS
        # backend in some torch versions) and move it to the data device.
        if shuffle:
            order = torch.randperm(n_sample).to(input.device)
        else:
            order = torch.arange(n_sample).to(input.device)
        n_batch = n_sample // batch_size

        for bi in range(n_batch):
            idx = order[bi*batch_size:(bi+1)*batch_size]
            sub = input[idx]

            ph_prob, ph_sample = self.sample_h_given_v(sub)
            chain_start = ph_sample

            for step in range(cdk):
                if step == 0:
                    nv_prob, nv_sample, nh_prob, nh_sample = self.gibbs_hvh(chain_start)
                else:
                    nv_prob, nv_sample, nh_prob, nh_sample = self.gibbs_hvh(nh_sample)

            if self.M_coeff > 0:
                self.dW *= self.M_coeff
                self.dv *= self.M_coeff
                self.dh *= self.M_coeff
                self.dW += (torch.matmul(sub.t(), ph_prob) - torch.matmul(nv_sample.t(), nh_prob)) * lr / batch_size
                self.dv += torch.mean(sub - nv_sample, dim=0) * lr
                self.dh += torch.mean(ph_prob - nh_prob, dim=0) * lr
            else:
                self.dW = (torch.matmul(sub.t(), ph_prob) - torch.matmul(nv_sample.t(), nh_prob)) * lr / batch_size
                self.dv = torch.mean(sub - nv_sample, dim=0) * lr
                self.dh = torch.mean(ph_prob - nh_prob, dim=0) * lr

            self.W  = self.W + self.dW
            self.vb = self.vb + self.dv
            self.hb = self.hb + self.dh

            self.W_decay(lr)
        return

    def W_decay(self, lr):
        #go through weights
        if self.L_coeff>0: #L2
            self.W -= self.L_coeff * lr * self.W
        elif self.L_coeff<0: #L1
            self.W += self.L_coeff * lr * torch.sign(self.W)
        else:
            #fix boundary
            self.W = torch.clamp(self.W, -10.0, 10.0)
        return

    def check_status(self, input, epoch, extra=""):
        n_sample = input.shape[0]

        ph_prob, ph_sample = self.sample_h_given_v(input)
        nv_prob, nv_sample, nh_prob, nh_sample = self.gibbs_hvh(ph_sample)
        error = torch.sum((input - nv_sample) ** 2) / n_sample
        # log_softmax avoids nan when softmax underflows to exactly 0 in fp32
        log_nv_prob = F.log_softmax(
            self.propdown(ph_sample).reshape([-1, self.n_val]), dim=-1
        ).reshape([-1, self.n_node])
        cross = -torch.mean(torch.sum(input * log_nv_prob, dim=1))
        freeE = self.get_free_energy(input)

        sys.stdout.write( "Training: " )
        sys.stdout.write( "epoch= %d " % epoch )
        sys.stdout.write( "cross= %f " % cross.item() )
        sys.stdout.write( "error= %f " % error.item() )
        sys.stdout.write( "freeE= %f " % freeE.item() )

        if self.enum_states is not None:
            sys.stdout.write( "KL= %f " % self.check_KL() )
        if self.prob_RGs is not None:
            sys.stdout.write( "rgKL= %f " % self.check_rgKL(nv_sample) )

        if extra:
            sys.stdout.write(extra)

        sys.stdout.write("\n")
        return

    def load_enum_states(self, fn):
        lines = open(fn, 'r').readlines()
        n_states = int(lines[0])
        dat_lst = []

        self.prob_states = torch.zeros(n_states, device=self.ctx)

        for i in range(1, n_states+1):
            es = lines[i].strip().split()
            for v in range(self.n_vis):
                dat_lst.append(int(es[0][v]))
            self.prob_states[i-1] = float(es[1])
            if self.prob_states[i-1] < 1e-10:
                self.prob_states[i-1] = 1e-10

        dat_lst = torch.tensor(dat_lst, dtype=torch.int64)

        self.enum_states = F.one_hot(dat_lst, self.n_val).reshape([-1, self.n_vis * self.n_val]).float().to(self.ctx)
        sys.stderr.write("Exact states info loaded!\n")
        return

    def load_enum_RGs(self, fn):
        #use gsl-histogram result [0, 0.5)
        lines = open(fn, 'r').readlines()
        n_states = int(lines[0].split()[0])

        self.prob_RGs = torch.zeros(n_states, device=self.ctx)

        for i in range(1, n_states+1):
            es = lines[i].strip().split()
            self.prob_RGs[i-1] = float(es[1]) + 1e-10 #log(1/20000)
            assert( int(float(es[0])*2) == i-1 )

        sys.stderr.write("Rg info from obs data loaded!\n")
        return

    def check_KL(self):
        ph_act = torch.matmul(self.enum_states, self.W) + self.hb
        vt = torch.matmul(self.enum_states, self.vb)
        ht = torch.sum(-torch.log(torch.sigmoid(-ph_act)), dim=1)
        p_th = torch.softmax(vt+ht, dim=0)
        KL = torch.sum(self.prob_states * torch.log(self.prob_states/p_th))
        return KL.item()

    def check_rgKL(self, v):
        ndx = torch.argmax(v.reshape([-1,self.n_val]), dim=1)
        pv = np.ones(shape=self.prob_RGs.shape)
        for intcoor in ndx.detach().cpu().numpy().reshape([-1,self.n_vis]):
            rg2 = cal_rg2(int2xy(intcoor))
            i = int(rg2*2)
            if i>=len(pv): i=len(pv)-1
            pv[i] += 1
        pv /= np.sum(pv)
        prg = self.prob_RGs.detach().cpu().numpy()
        KL = np.sum(prg * np.log(prg/pv))
        return KL

    def sample_h_given_v(self, v0):
        h1_prob = self.propup(v0)
        # mxnet comparison returns 1.0/0.0 floats; keep that dtype with .float()
        h1 = (h1_prob > torch.rand(h1_prob.shape, device=self.ctx)).float()
        return [h1_prob, h1]

    def sample_v_given_h(self, h0):
        v1_prob = self.propdown(h0).reshape([-1, self.n_val])
        v1_prob = torch.softmax(v1_prob, dim=-1)
        v1_args = torch.multinomial(v1_prob, 1).reshape(-1)  # one category per row
        v1 = F.one_hot(v1_args, self.n_val).float()
        return [v1_prob.reshape([-1,self.n_node]), v1.reshape([-1,self.n_node])]

    def propup(self, v):
        pre_sigmoid_activation = torch.matmul(v, self.W) + self.hb
        return torch.sigmoid(pre_sigmoid_activation)

    def propdown(self, h):
        return torch.matmul(h, self.W.t()) + self.vb

    def gibbs_hvh(self, h0):
        v1_prob, v1 = self.sample_v_given_h(h0)
        h1_prob, h1 = self.sample_h_given_v(v1)
        return [v1_prob, v1, h1_prob, h1]

    def get_free_energy(self, v):
        x = torch.matmul( v, self.W ) + self.hb
        vt = torch.matmul( v, self.vb )
        ht = torch.sum( torch.log( 1.0 + torch.exp(x) ), dim=1 )
        fe = -ht-vt #free energy, how to prevent scale
        return torch.mean( fe )

    def reconstruct(self, v):
        h = torch.sigmoid( torch.matmul( v, self.W ) + self.hb )
        reconstructed_v_prob = torch.sigmoid( torch.matmul( h, self.W.t() ) + self.vb )
        return reconstructed_v_prob

    def daydream(self, n, k=1):
        sys.stderr.write("Day dreaming ...\n")
        h0 = torch.empty((1, self.n_hid), device=self.ctx).uniform_(0, 1)

        for i in range(100):
            v_prob, v = self.sample_v_given_h(h0)
            h_prob, h0 = self.sample_h_given_v(v)
        for i in range(n):
            for cdk in range(k):
                v_prob, v = self.sample_v_given_h(h0)
                h_prob, h0 = self.sample_h_given_v(v)
            ndx = torch.argmax(v.reshape([-1,self.n_val]), dim=1)
            for n in ndx.detach().cpu().numpy(): sys.stdout.write(str(int(n)))
            sys.stdout.write("\n")
        return

    def yeardream(self, v, n, k=1):
        sys.stderr.write("Year dreaming ...\n")
        n_sample = v.shape[0]
        for i in range(100):
            h_prob, h0 = self.sample_h_given_v(v)
            v_prob, v = self.sample_v_given_h(h0)
        for i in range(n):
            for cdk in range(k):
                h_prob, h0 = self.sample_h_given_v(v)
                v_prob, v = self.sample_v_given_h(h0)
            ndx = torch.argmax(v.reshape([-1,self.n_val]), dim=1)
            for j, n in enumerate(ndx.detach().cpu().numpy()):
                sys.stdout.write(str(int(n)))
                if j % self.n_vis == (self.n_vis-1):
                    sys.stdout.write("\n")
        return

    def load(self, fn):
        lines = open(fn, 'r').readlines()
        elems = lines[-1].split()
        last_epoch = int(elems[0])
        pos = 1
        for i in range(self.n_node):
            for j in range(self.n_hid):
                self.W[i,j] = float(elems[pos])
                pos += 1
        for i in range(self.n_node):
            self.vb[i] = float(elems[pos])
            pos += 1
        for j in range(self.n_hid):
            self.hb[j] = float(elems[pos])
            pos += 1

        sys.stderr.write("Loading weights and restart from epoch=%d\n" % last_epoch)
        return last_epoch

    def save(self, fn, epoch=0):
        W = self.W.detach().cpu().numpy()
        vb = self.vb.detach().cpu().numpy()
        hb = self.hb.detach().cpu().numpy()
        with open(fn, 'a') as fp:
            fp.write("%d " % epoch)
            for i in range(self.n_node):
                for j in range(self.n_hid):
                    fp.write("%6.4f " % W[i,j])
            for i in range(self.n_node):
                fp.write("%6.4f " % vb[i])
            for j in range(self.n_hid):
                fp.write("%6.4f " % hb[j])
            fp.write("\n")
        return

def get_data( fn, n_vis, n_val, ctx=torch.device("cpu") ):
    if fn.isdigit():
        #random
        num_data = int(fn)
        prob = torch.ones((num_data * n_vis, n_val)) / n_val
        dat_lst = torch.multinomial(prob, 1).reshape(-1)
        sys.stderr.write("Generating random data: nv= %d, nd= %d\n" % (n_vis, num_data))
    else:
        #read from file
        with open(fn, 'r') as fp:
            lines = fp.readlines()
            es = lines[0].split()
            nl = int(es[0])
            nv = int(es[1])
            dat_lst = []
            sys.stderr.write("Loading data: nv= %d, nd= %d\n" % (nv, nl))
            for l in range(1, nl+1):
                for i in range(nv):
                    dat_lst.append(int(lines[l][i]))
            dat_lst = torch.tensor(dat_lst, dtype=torch.int64)
    data = F.one_hot(dat_lst.to(torch.int64), n_val).reshape([-1, n_vis * n_val]).float()
    return data.to(ctx)

def run_rbm():
    #setup options
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
    parser.add_argument("-x", '--gpu', type=int, default=-1, help="using GPU x or CPU(-1)")
    parser.add_argument("-kl", '--KL', type=str, help="check KL")
    parser.add_argument("-rg", '--rgKL', type=str, help="check KL of RG")

    args = parser.parse_args()
    learning_rate = args.learning_rate
    training_epochs = args.train
    gen_steps = args.generate
    bs = args.batch_size
    k = args.cdk
    num_vis = args.visible
    num_hid = args.hidden
    num_val = args.level

    if args.gpu<0:
        ctx = torch.device("cpu")
        sys.stderr.write("Using CPUs\n")
    else:
        # -x >= 0 requests an accelerator. Preference: CUDA (Nvidia) first,
        # then MPS (Apple Silicon GPU), otherwise fall back to CPU so the same
        # command line keeps working on any machine.
        if torch.cuda.is_available():
            ctx = torch.device("cuda:%d" % args.gpu)
            sys.stderr.write("Using GPU %d\n" % args.gpu)
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            ctx = torch.device("mps")
            sys.stderr.write("Using Apple MPS (Metal GPU)\n")
        else:
            ctx = torch.device("cpu")
            sys.stderr.write("GPU %d requested but no CUDA/MPS; using CPUs\n" % args.gpu)

    if args.data != "0":
        data = get_data(args.data, num_vis, num_val, ctx)
    else:
        data = None

    # construct RBM
    sys.stderr.write("Creating RBM: nv= %d, nl= %d, nh= %d\n" % (num_vis, num_val, num_hid))
    rbm = RBM(n_vis=num_vis, n_hid=num_hid, n_val=num_val, ctx=ctx)
    rbm.L_coeff = args.rescaleL
    rbm.M_coeff = args.momentum

    # load weights
    epoch_start = 0
    if args.weight is not None:
        if os.path.isfile(args.weight):
            epoch_start = rbm.load(args.weight)

    # load KL reference
    if args.KL is not None: rbm.load_enum_states(args.KL)
    if args.rgKL is not None: rbm.load_enum_RGs(args.rgKL)

    # train
    if training_epochs > 0 and data is not None:
        if epoch_start == 0: rbm.check_status(data, 0)
        t_win = time.perf_counter()  # wall-clock timer for the current window
        for epoch in range(epoch_start, training_epochs):
            rbm.contrastive_divergence(input=data, lr=learning_rate, cdk=k,
                batch_size=bs, shuffle=(epoch>0))
            if (epoch+1) % args.interval == 0:
                # Force pending accelerator (CUDA/MPS) work to finish so the
                # timing reflects real compute, not just async dispatch.
                if ctx.type == "cuda":
                    torch.cuda.synchronize()
                elif ctx.type == "mps":
                    torch.mps.synchronize()
                dt = time.perf_counter() - t_win
                timing = "（ %.2f ms/epoch on %s ）" % (1000.0*dt/args.interval, ctx.type)
                rbm.check_status(data, epoch+1, extra=timing)
                sys.stdout.flush()
                sys.stderr.flush()
                #save weights (append), in one line
                if args.weight is not None: rbm.save(args.weight, epoch+1)
                t_win = time.perf_counter()  # reset for next window

    if gen_steps > 0:
        if data is not None:
            rbm.yeardream(data, gen_steps, k)
        else:
            rbm.daydream(gen_steps, k)

if __name__ == "__main__":
    run_rbm()
