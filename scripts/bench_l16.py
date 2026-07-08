# -*- coding: utf-8 -*-
# Benchmark harness: compares the current RBM-tor.py training step against an
# optimized variant on the L16 dataset (or any data file).
#
# What the optimized variant changes (algorithm-preserving):
#   1. Categorical visible sampling via inverse-CDF (rand + cumsum + compare)
#      instead of torch.multinomial. Same distribution, but stays fully on the
#      accelerator and avoids the host<->device sync / slow kernel that
#      torch.multinomial triggers (notably on MPS).
#   2. Free energy uses F.softplus instead of log(1+exp(x)) -> no fp32 overflow.
#   3. Optional torch.compile on the per-batch CD-k step to cut per-op dispatch
#      overhead (the real bottleneck when you run ~1e6 tiny epochs).
#
# Usage:
#   python bench_l16.py -d ../data/L16/conf1.txt -v 14 -l 3 -n 512 -b 10000 \
#          -e 60 --device cpu            # or: --device mps / --device cuda
#
# It prints ms/epoch for: baseline, fast (vectorized sampler), and
# fast+compile, plus a sanity check that the training loss tracks.

import importlib.util, sys, os, time, argparse
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))

def load_rbm_module():
    spec = importlib.util.spec_from_file_location(
        "rbmtor", os.path.join(HERE, "RBM-tor.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

rbmtor = load_rbm_module()
BaseRBM = rbmtor.RBM
get_data = rbmtor.get_data


class FastRBM(BaseRBM):
    """Same math as BaseRBM, faster visible sampling + stable free energy."""

    def sample_v_given_h(self, h0):
        pre = self.propdown(h0).reshape([-1, self.n_val])
        v1_prob = torch.softmax(pre, dim=-1)
        # inverse-CDF categorical draw: idx = #{cdf < u}
        cdf = torch.cumsum(v1_prob, dim=-1)
        u = torch.rand(v1_prob.shape[0], 1, device=v1_prob.device)
        idx = (u > cdf).sum(dim=1).clamp_(max=self.n_val - 1)
        v1 = F.one_hot(idx, self.n_val).to(v1_prob.dtype)
        return [v1_prob.reshape([-1, self.n_node]),
                v1.reshape([-1, self.n_node])]

    def get_free_energy(self, v):
        x = torch.matmul(v, self.W) + self.hb
        vt = torch.matmul(v, self.vb)
        ht = torch.sum(F.softplus(x), dim=1)
        return torch.mean(-ht - vt)


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()


def clone_state(src, dst):
    dst.W = src.W.clone()
    dst.vb = src.vb.clone()
    dst.hb = src.hb.clone()
    dst.dW = src.dW.clone()
    dst.dv = src.dv.clone()
    dst.dh = src.dh.clone()


def time_training(rbm, data, lr, k, bs, dev, epochs, warmup=5, label=""):
    # warmup (build any compile cache, warm allocator)
    for _ in range(warmup):
        rbm.contrastive_divergence(data, lr=lr, cdk=k, batch_size=bs, shuffle=True)
    sync(dev)
    t0 = time.perf_counter()
    for _ in range(epochs):
        rbm.contrastive_divergence(data, lr=lr, cdk=k, batch_size=bs, shuffle=True)
    sync(dev)
    dt = time.perf_counter() - t0
    ms = 1000.0 * dt / epochs
    err = rbm.check_status_error(data) if hasattr(rbm, "check_status_error") else None
    print("  %-20s %8.2f ms/epoch" % (label, ms))
    return ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data", required=True)
    ap.add_argument("-v", "--visible", type=int, default=14)
    ap.add_argument("-l", "--level", type=int, default=3)
    ap.add_argument("-n", "--hidden", type=int, default=512)
    ap.add_argument("-b", "--batch_size", type=int, default=10000)
    ap.add_argument("-k", "--cdk", type=int, default=1)
    ap.add_argument("-r", "--lr", type=float, default=0.1)
    ap.add_argument("-m", "--momentum", type=float, default=0.5)
    ap.add_argument("-e", "--epochs", type=int, default=60)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    dev = torch.device(args.device)
    print("device=%s  nv=%d nl=%d nh=%d batch=%d cdk=%d epochs=%d" % (
        args.device, args.visible, args.level, args.hidden,
        args.batch_size, args.cdk, args.epochs))

    data = get_data(args.data, args.visible, args.level, dev)
    print("data shape:", tuple(data.shape))

    def make(cls):
        torch.manual_seed(0)
        r = cls(n_vis=args.visible, n_hid=args.hidden, n_val=args.level, ctx=dev)
        r.M_coeff = args.momentum
        return r

    print("\n--- ms/epoch (lower is better) ---")
    base = make(BaseRBM)
    time_training(base, data, args.lr, args.cdk, args.batch_size, dev,
                  args.epochs, label="baseline")

    fast = make(FastRBM)
    time_training(fast, data, args.lr, args.cdk, args.batch_size, dev,
                  args.epochs, label="fast (vec sampler)")

    # fast + torch.compile on the CD step
    try:
        comp = make(FastRBM)
        comp.contrastive_divergence = torch.compile(
            comp.contrastive_divergence, dynamic=False)
        time_training(comp, data, args.lr, args.cdk, args.batch_size, dev,
                      args.epochs, warmup=8, label="fast + compile")
    except Exception as ex:
        print("  fast + compile        skipped (%s)" % type(ex).__name__)

    # correctness: loss after identical #epochs from identical init
    print("\n--- training-loss sanity (from seed 0, %d epochs) ---" % args.epochs)
    for name, cls in [("baseline", BaseRBM), ("fast", FastRBM)]:
        r = make(cls)
        for _ in range(args.epochs):
            r.contrastive_divergence(data, lr=args.lr, cdk=args.cdk,
                                     batch_size=args.batch_size, shuffle=True)
        ph_prob, ph_sample = r.sample_h_given_v(data)
        nv_prob, nv_sample, nh_prob, nh_sample = r.gibbs_hvh(ph_sample)
        err = (torch.sum((data - nv_sample) ** 2) / data.shape[0]).item()
        fe = r.get_free_energy(data).item()
        print("  %-10s recon_err=%.4f  freeE=%.3f" % (name, err, fe))


if __name__ == "__main__":
    main()
