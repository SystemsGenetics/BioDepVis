# BioDepVis

A visualizer for gene-coexpression networks (GCNs) and alignments between them.

Watch our video demo showcasing some of the features at https://clemson.app.box.com/v/BioDepVis

## Dependencies

BioDepVis requires CUDA, gcc, and Qt.

### Ubuntu
```
# CUDA drivers can be downloaded and installed from NVIDIA's website

sudo apt-get install gcc qt5-default
```

### Palmetto
```
module load cuda-toolkit/9.2 gcc/5.4.0 Qt/5.9.2
```

## Usage

To build and run the executable:
```
make CUDADIR=$CUDA_ROOT

./BioDepVis --config [config-file] --ont [ont-file]
```

The default values for these input files are `config/test_M-R.json` and `go-basic.obo`, which are provided in this repo. This configuration will load two gene networks, Maize and Rice, and their alignment, and the ontology database provided by `go-basic.obo`. The `config` folder contains several other example graph/alignment configurations. Use these example files to create your own configurations.

## Stream from a remote machine

Install [VirtualGL](https://virtualgl.org/) and [TurboVNC](https://turbovnc.org/) on the remote machine. You will also need to install TurboVNC or an equivalent VNC client on your local machine.

Login to the remote machine through SSH and start a VNC server:
```
LANG=C /opt/TurboVNC/bin/vncserver
```

Start a VNC client on your local machine and connect to the remote VNC server at `<hostname>:1`. For the TurboVNC client:
```
/opt/TurboVNC/bin/vncviewer
```

Start BioDep-Vis on the remote machine:
```
vglrun ./BioDepVis [...]
```

To stop the VNC server:
```
/opt/TurboVNC/bin/vncserver -kill :1
```

## Stream from Palmetto

Palmetto should already have VirtualGL and TurboVNC installed. However, because compute nodes cannot be accessed directly via SSH but only through the login node, you must tunnel the VNC server through the login node on a second SSH session. As before, you will need to install a VNC client on your local machine. Binary packages for TurboVNC can be found [here](https://sourceforge.net/projects/turbovnc/files/2.2).

Login to a GPU node on Palmetto and start a VNC server:
```
ssh -X <username>@login.palmetto.clemson.edu
qsub -I -l select=1:ngpus=2:ncpus=4:mem=32gb,walltime=02:00:00

LANG=C vncserver
```

Look for `TurboVNC: <node>:<port>` in the output. For example: `node0263:1`.

Login to another Palmetto session with tunnelling:
```
ssh -L 10000:<node>.palmetto.clemson.edu:<5900 + port> <username>@login.palmetto.clemson.edu

# example:
ssh -L 10000:node0263.palmetto.clemson.edu:5901 ksapra@login.palmetto.clemson.edu
```

Start a VNC client on your local machine and connect to `localhost:10000`. A remote desktop will appear, from which point you can launch BioDepVis as before with `vglrun`.

## Record VNC Session

Install `vnc2flv`:
```
wget https://pypi.python.org/packages/1e/8e/40c71faa24e19dab555eeb25d6c07efbc503e98b0344f0b4c3131f59947f/vnc2flv-20100207.tar.gz
tar xvf vnc2flv-20100207.tar.gz

cd vnc2flv-20100207
python setup.py install --user
```

Start the VNC server at <port> (e.g. 5901), then start recording:
```
cd tools
python flvrec.py localhost <port>
```

To stop recording, enter `Ctrl-C` and an FLV file will be saved.

## Launch from CCT

The Complexity Connector Tool (CCT) is a graphical interface for creating config files for BioDepVis, instead of writing the JSON files yourself. With CCT you can create or load a configuration and then launch BioDepVis right from the interface! You can even have multiple instances of BioDepVis running at once! Refer to the CCT docs for installation and usage instructions.

## Run in Docker container

This feature is still incomplete:
```
sudo docker run --runtime=nvidia --rm -it \
  -e DISPLAY=$DISPLAY -u docker -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
  systemsgenetics/biodepvis
```
