#!/bin/bash

#341
# rsync -avz -e 'ssh -p 10929' ./* root@region-42.seetacloud.com:/root/autodl-tmp/code/torchcuda/.
rsync -avz -e 'ssh -p 10929' root@region-42.seetacloud.com:/root/autodl-tmp/code/torchcuda/* ./.
# uTXHQIDvCQcF