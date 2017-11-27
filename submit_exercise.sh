#!/bin/bash

REL_SCRIPT_DIR=$(dirname "$0")
INITIAL_DIR=$(pwd)
cd $REL_SCRIPT_DIR
ABS_SCRIPT_DIR=$(pwd)

#First argument is mandatory username
if [ $# -lt 2 ]; then
    echo 1>&2 "Usage: $0 <exercise_num> <username>"
    echo 1>&2 "e.g. $0 1 s111"
    exit 1
fi

cd exercise_$1
chmod -R a+r dl4cv
chmod a+x dl4cv dl4cv/classifiers
echo "Enter the password for user $2 to upload your model files and dl4cv directory:"

rsync --delete-before -rlv -e 'ssh -x -p 58022' --exclude '*.pyc' --exclude 'output.*' --exclude "__pycache__/" models/ dl4cv $2@filecremers1.informatik.tu-muenchen.de:submit/EX$1/

cd $INITIAL_DIR
