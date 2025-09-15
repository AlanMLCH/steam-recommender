import os, time, requests, yaml


DYNAMIC_PATH = os.getenv('DYNAMIC_PATH','/etc/traefik/dynamic.yml')
SLA_MS = int(os.getenv('SLA_P95_MS','150'))
KPI_MIN = float(os.getenv('KPI_MIN','0.0'))
METRICS = os.getenv('GATEWAY_METRICS','http://gateway:8000/metrics_json')
MODE = os.getenv('MODE','canary')
CANARY_WEIGHT = int(os.getenv('CANARY_WEIGHT','10'))


CANARY_TMPL = {
'http': {
'routers': {'rerank': {'rule': 'PathPrefix(`/rerank`)', 'service': 'rerank-wrr'}},
'services': {
'rerank-wrr': {'weighted': {'services': [ {'name':'champion','weight': 100-CANARY_WEIGHT}, {'name':'challenger','weight': CANARY_WEIGHT} ]}},
'champion': {'loadBalancer': {'servers':[{'url':'http://reranker_champion:8080'}]}},
'challenger': {'loadBalancer': {'servers':[{'url':'http://reranker_challenger:8080'}]}}
}
}
}


BLUE_TMPL = {
'http': {
'routers': {'rerank': {'rule': 'PathPrefix(`/rerank`)', 'service': 'champion'}},
'services': {
'champion': {'loadBalancer': {'servers':[{'url':'http://reranker_champion:8080'}]}},
'challenger': {'loadBalancer': {'servers':[{'url':'http://reranker_challenger:8080'}]}}
}
}
}


def write_cfg(obj):
with open(DYNAMIC_PATH,'w') as f:
yaml.safe_dump(obj, f)


def set_canary(weight:int):
CANARY_TMPL['http']['services']['rerank-wrr']['weighted']['services'][0]['weight'] = 100-weight
CANARY_TMPL['http']['services']['rerank-wrr']['weighted']['services'][1]['weight'] = weight
write_cfg(CANARY_TMPL)


if MODE == 'canary':
set_canary(CANARY_WEIGHT)
else:
write_cfg(BLUE_TMPL)


bad_ticks = 0
while True:
try:
r = requests.get(METRICS, timeout=3).json()
p95 = r.get('p95_ms',0)
kpi = r.get('kpi',0)
if p95 > SLA_MS or kpi < KPI_MIN:
bad_ticks += 1
else:
bad_ticks = 0
# auto-rollback if 3 consecutive bad intervals
if bad_ticks >= 3:
write_cfg(BLUE_TMPL) # 100% champion
bad_ticks = 0
except Exception:
pass
time.sleep(10)