#!/bin/sh
python sac.py --exp_name youbot_all_img_final --ENV youBot_all --ARM --BASE --RPA 4 --HIDDEN_SIZE 256 --BATCH_SIZE 256 -lr 0.001 -steps 550000 --TEST_INTERVAL 5500 --DEMONSTRATIONS 'youbot_all_final' --UPDATE_START 0 --PRIORITIZE_REPLAY --BEHAVIOR_CLONING
