#!/bin/bash
sudo docker stop foxy_controller || true
sudo docker rm foxy_controller || true
cd ~/a1_gym/a1_gym_deploy/docker/
sudo make autostart