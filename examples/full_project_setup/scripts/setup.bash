#!/usr/bin/env bash
# replace localhost with the port you see on the smartphone
export ROS_MASTER_URI="http://172.20.10.2:11311"

# You want your local IP, usually starting with 192.168, following RFC1918
# Windows powershell:
#    (Get-NetIPAddress | Where-Object { $_.AddressState -eq "Preferred" -and $_.ValidLifetime -lt "24:00:00" }).IPAddress
# linux:
#    hostname -I | awk '{print $1}'
# macOS:
#    ipconfig getifaddr en1
#    ifconfig en0 | awk '/inet / {print $2}'
export COPPELIA_SIM_IP="145.108.66.189"
