import argparse, yaml
from pathlib import Path
p = argparse.ArgumentParser(); p.add_argument('--weight', type=int, default=10)
args = p.parse_args()
cfg = {
    'http':{
    'routers':{'rerank':{'rule':'PathPrefix(`/rerank`)','service':'rerank-wrr'}},
    'services':{
    'rerank-wrr':{'weighted':{'services':[{'name':'champion','weight':100-args.weight},{'name':'challenger','weight':args.weight}]}},
    'champion':{'loadBalancer':{'servers':[{'url':'http://reranker_champion:8080'}]}},
    'challenger':{'loadBalancer':{'servers':[{'url':'http://reranker_challenger:8080'}]}}
}}}
Path('traefik/dynamic.yml').write_text(yaml.safe_dump(cfg))
print(f"Set challenger canary weight={args.weight}%")