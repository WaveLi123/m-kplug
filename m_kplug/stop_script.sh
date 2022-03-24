#!/bin/bash

#pid=$(ps -aux | grep -E 'hmc_tags_server_estimator|hmc_tags_client_estimator|tensorflow_model_server' | grep python | grep -v 'screen' | grep -v 'SCREEN' | grep -v 'grep' | gawk '{print $2}')
pid=$(ps -aux | grep $1 | grep -v 'screen' | grep -v 'SCREEN' | grep -v 'grep' | awk '{print $2}')

echo ${pid}

for p in ${pid}; do
  cmd="kill -9 ${p}"
  echo ${cmd}
  $(${cmd})
done
