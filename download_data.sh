#!/bin/bash

set -euo pipefail

mkdir -p data

curl -C - -L -o data/tiny-imagenet-200.zip https://cs231n.stanford.edu/tiny-imagenet-200.zip

unzip data/tiny-imagenet-200.zip -d data
