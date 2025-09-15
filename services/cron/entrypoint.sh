#!/bin/sh
printenv | sed 's/\(.\+\)/export \1/g' > /etc/profile.d/env.sh
cron && tail -f /var/log/cron.log