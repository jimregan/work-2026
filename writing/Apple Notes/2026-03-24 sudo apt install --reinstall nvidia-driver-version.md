sudo apt install --reinstall nvidia-driver-<version>


apt list --installed | grep nvidia-driver


## ubuntu-drivers devices


sudo apt install --reinstall nvidia-driver-550

6.8.0-1008-nvidia

dpkg -l | grep linux-headers


find /lib/modules/$(uname -r) -name '*nvidia*.ko'


nvidiafb.ko

nvidia/550.163.01: added