#!/bin/bash

###
# Usage: script.sh <experiment number>, e.g. deleteExperiment.sh 1
#        deletes checkpoints, tensorboard and logs
###
exp_no=$1

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

root_dir=$(dirname $(readlink -fm $DIR))

#delete checkpoints
to_delete=$root_dir/checkpoints/checkpoints$exp_no
if [ -d "$to_delete" ]; then
    rm -r $to_delete
    echo "deleted $to_delete"
fi

#delete tensorboard
to_delete=$root_dir/tensorboard/tensorboard$exp_no
if [ -d "$to_delete" ]; then
    rm -r $to_delete
    echo "deleted $to_delete"
fi


#delete logs
to_delete=$root_dir/logs/log$exp_no*
if [ -f "$to_delete" ]; then
    rm $to_delete
    echo "deleted $to_delete"
fi

