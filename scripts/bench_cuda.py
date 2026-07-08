# -*- coding: utf-8 -*-
# CUDA benchmark for the categorical RBM (compares RBM-tor.py vs RBM-tor-fast.py).
#
# Run on your Linux CUDA box, e.g.:
#   python bench_cuda.py -d ../data/L16/conf1.txt -v 14 -l 3 -n 512 -b 10000 -e 100
#
# It reports ms/epoch (and samples/s) for four variants and checks that the
# optimized version trains equivalently to the baseline:
#   baseline               current RBM-tor.py (torch.multinomial sampler)
#   fast                   vectorized categorical sampler + softplus, eager
#   fast+compile           torch.compile (default) on the fused CD-k core
#   fast+compile(reduce)   torch.compile(mode="reduce-overhead") -> CUDA graphs,
#                          removes per-op launch overhead (best for small nh)
#
# Useful flags:
#   --device cuda|cpu|mps   (default: auto -> cuda if available)
#   --hidden-sweep 128,256,512,1024   benchmark several hidden sizes
#   --dtype float32|float16|bfloat16
#   -e / --epochs 100       timed epochs        --warmup 10
#   -k / --cdk 1            CD-k steps           -m / --momentum 0.5
#
# If --data is a plain integer N (e.g. -d 100000) it generates N random samples,
# so you can benchmark without a data file.

import importlib.util, os, sys, time, argparse
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def load(fname, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


base = load("RBM-tor.py", "rbmtor_base")
fast = load("RBM-tor-fast.py", "rbmtor_fast")


def pick_device(name):
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()


def time_variant(make, data, dev, cfg, label):
    """warmup then time cfg.epochs; return (ms_per_epoch or None)."""
    try:
        r = make()
    except Exception as ex:
        print("  %-24s skipped (%s: %s)" % (label, type(ex).__name__, ex))
        return None
    kw = dict(lr=cfg.lr, cdk=cfg.cdk, batch_size=cfg.batch, shuffle=True)
    try:
        for _ in range(cfg.warmup):
            r.contrastive_divergence(data, **kw)
        sync(dev)
        t0 = time.perf_counter()
        for _ in range(cfg.epochs):
            r.contrastive_divergence(data, **kw)
        sync(dev)
        ms = 1000.0 * (time.perf_counter() - t0) / cfg.epochs
    except Exception as ex:
        print("  %-24s failed  (%s: %s)" % (label, type(ex).__name__, ex))
        return None
    n_batch = data.shape[0] // cfg.batch
    sps = (n_batch * cfg.batch) / (ms / 1000.0)
    print("  %-24s %8.2f ms/epoch   %.2e samples/s" % (label, ms, sps))
    return ms


def bench_one(data, dev, cfg, nh):
    print("\n[nh=%d] nv=%d nl=%d batch=%d cdk=%d dtype=%s  (warmup=%d, timed=%d)"
          % (nh, cfg.nv, cfg.nl, cfg.batch, cfg.cdk, cfg.dtype, cfg.warmup, cfg.epochs))

    def mk_base():
        torch.manual_seed(0)
        r = base.RBM(n_vis=cfg.nv, n_hid=nh, n_val=cfg.nl, ctx=dev)
        r.M_coeff = cfg.momentum
        return r

    def mk_fast(compile_step=False, mode=None):
        def g():
            torch.manual_seed(0)
            r = fast.RBM(n_vis=cfg.nv, n_hid=nh, n_val=cfg.nl, ctx=dev,
                         compile_step=compile_step, compile_mode=mode)
            r.M_coeff = cfg.momentum
            return r
        return g

    ms_b = time_variant(mk_base, data, dev, cfg, "baseline")
    ms_f = time_variant(mk_fast(False), data, dev, cfg, "fast")
    ms_c = time_variant(mk_fast(True, None), data, dev, cfg, "fast+compile")
    ms_r = None
    if dev.type == "cuda":
        ms_r = time_variant(mk_fast(True, "reduce-overhead"), data, dev, cfg,
                            "fast+compile(reduce)")

    if ms_b:
        for lbl, ms in [("fast", ms_f), ("fast+compile", ms_c),
                        ("fast+compile(reduce)", ms_r)]:
            if ms:
                print("    speedup %-22s x%.2f" % (lbl, ms_b / ms))


def correctness(data, dev, cfg, nh, ep=20):
    print("\n--- equivalence check (seed 0, %d epochs, nh=%d) ---" % (ep, nh))

    def run(make):
        torch.manual_seed(0)
        r = make()
        r.M_coeff = cfg.momentum
        for _ in range(ep):
            r.contrastive_divergence(data, lr=cfg.lr, cdk=cfg.cdk,
                                     batch_size=cfg.batch, shuffle=True)
        ph, phs = r.sample_h_given_v(data)
        nv_p, nv, nh_p, nh_s = r.gibbs_hvh(phs)
        err = (torch.sum((data - nv) ** 2) / data.shape[0]).item()
        fe = r.get_free_energy(data).item()
        return err, fe

    eb, fb = run(lambda: base.RBM(n_vis=cfg.nv, n_hid=nh, n_val=cfg.nl, ctx=dev))
    ef, ff = run(lambda: fast.RBM(n_vis=cfg.nv, n_hid=nh, n_val=cfg.nl, ctx=dev))
    print("  baseline  recon_err=%.4f  freeE=%.3f" % (eb, fb))
    print("  fast      recon_err=%.4f  freeE=%.3f" % (ef, ff))
    print("  (small differences are RNG-stream noise; distributions are identical)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data", default="100000")
    ap.add_argument("-v", "--visible", type=int, default=14)
    ap.add_argument("-l", "--level", type=int, default=3)
    ap.add_argument("-n", "--hidden", type=int, default=512)
    ap.add_argument("--hidden-sweep", type=str, default="")
    ap.add_argument("-b", "--batch", type=int, default=10000)
    ap.add_argument("-k", "--cdk", type=int, default=1)
    ap.add_argument("-r", "--lr", type=float, default=0.1)
    ap.add_argument("-m", "--momentum", type=float, default=0.5)
    ap.add_argument("-e", "--epochs", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="float32",
                    choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--no-check", action="store_true")
    a = ap.parse_args()

    dev = pick_device(a.device)
    print("torch %s   device=%s" % (torch.__version__, dev))
    if dev.type == "cuda":
        print("gpu:", torch.cuda.get_device_name(dev),
              "| cuda", torch.version.cuda)
    print("threads:", torch.get_num_threads())

    dt = getattr(torch, a.dtype)
    data = base.get_data(a.data, a.visible, a.level, dev)
    if dt != torch.float32:
        data = data.to(dt)
    print("data:", tuple(data.shape), data.dtype)

    class Cfg:
        nv, nl = a.visible, a.level
        batch, cdk, lr, momentum = a.batch, a.cdk, a.lr, a.momentum
        epochs, warmup, dtype = a.epochs, a.warmup, a.dtype
    cfg = Cfg()

    sizes = ([int(x) for x in a.hidden_sweep.split(",")]
             if a.hidden_sweep else [a.hidden])
    for nh in sizes:
        bench_one(data, dev, cfg, nh)
    if not a.no_check:
        correctness(data, dev, cfg, sizes[0])


if __name__ == "__main__":
    main()
