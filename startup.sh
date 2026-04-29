#!/bin/bash
unalias python
unset LD_LIBRARY_PATH
unset PYTHONPATH

echo "$LD_LIBRARY_PATH"

source ~/gw_data/gw_env/bin/activate

which python

