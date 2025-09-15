COMPOSE = docker compose


.PHONY: up down logs train canary shadow-on shadow-off blue green weights


up:
$(COMPOSE) up --build -d traefik retrieval reranker_champion reranker_challenger gateway watchdog cron


down:
$(COMPOSE) down -v


logs:
$(COMPOSE) logs -f --tail=200


train:
$(COMPOSE) run --rm trainer


canary:
python scripts/set_weights.py --weight $${W:-10}


shadow-on:
bash scripts/shadow_on.sh


shadow-off:
bash scripts/shadow_off.sh


blue:
bash scripts/blue_green.sh champion


green:
bash scripts/blue_green.sh challenger


weights:
python scripts/set_weights.py --weight 0 # 0% challenger (100% champion)