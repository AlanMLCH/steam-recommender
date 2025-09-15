import json, argparse


def psi(a_hist, b_hist):
    s = 0.0
    for a, b in zip(a_hist, b_hist):
        a = max(a, 1e-6); b = max(b, 1e-6)
        s += (a - b) * (math.log(a) - math.log(b))
    return s


def main():
    import os, math
    p = argparse.ArgumentParser()
    p.add_argument('--baseline', required=True)
    p.add_argument('--current', required=True)
    p.add_argument('--threshold', type=float, default=0.25)
    args = p.parse_args()
    try:
        base = json.load(open(args.baseline))
        cur = json.load(open(args.current))
    except FileNotFoundError:
        return
    b = base.get('user_norm_mean', 1.0)
    c = cur.get('user_norm_mean', 1.0)
    # toy PSI using 2-bin histogram
    a_hist = [0.5, 0.5] if b==1.0 else [0.6, 0.4]
    b_hist = [0.5, 0.5] if c==1.0 else [0.4, 0.6]
    score = psi(a_hist, b_hist)
    out = {'psi': score}
    print(json.dumps(out))
    if score > args.threshold:
        open(os.path.join(os.path.dirname(args.baseline),'retrain_requested'), 'w').close()


if __name__ == '__main__':
    main()