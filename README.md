# BFMC - Brain Project

Logs:
- 12/1/2023: Haven't debug StreamerProcess and build DetectionProcess yet.
- 12/1/2023: Can run detection demo with the following:
```
bash get_pi_requirements.sh
cd src/utils/detection
python3 detect.py
```

### ROS + Docker
After successfully installing docker, we will pull ROS image from the Docker hub \

**Ubuntu** requires root admission to install
```
sudo docker pull osrf/ros-noetic-desktop-full
```
Others:
```
docker pull osrf/ros-noetic-desktop-fulll 
```

Run the container to verify the installation of ROS (don't forget the root admission if u use Ubuntu)
```
docker run <repository>:TAG
```
If everything is working well, we can see the root@<Tag>. If the docker container
does not run properly, please try

```
docker run -i -t <IMAGE-ID>
```




