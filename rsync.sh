#!/bin/bash

# 357
# rsync -avz -e 'ssh -p 42302' root@region-42.seetacloud.com:/root/autodl-tmp/code/torchcuda/* ./.
rsync -avz -e 'ssh -p 42302' ./* root@region-42.seetacloud.com:/root/autodl-tmp/code/torchcuda/.
# NxEc5hi3lwXk

# rsync -avz -e 'ssh -p 54060' ./* root@region-41.seetacloud.com:/root/autodl-tmp/code/torchcuda/.
# aKABDaULSmfp