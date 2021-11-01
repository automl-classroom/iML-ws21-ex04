#!/bin/bash

MODE=$1
PW=$2

if [ $MODE = "pack" ]; then
    zip -P $PW -re intern.zip solutions others
elif [ $MODE = "unpack" ]; then
    unzip -P $PW intern.zip
fi