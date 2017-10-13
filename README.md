# BioDep-Vis

Watch our video demo showcasing some of the features at https://clemson.app.box.com/v/BioDepVis

## Dependencies

You should have the following packages installed on your system:
- CUDA 5.0 or greater
- gcc 4.8 or greater

On Palmetto, these packages are available as modules:
```
module load cuda-toolkit/7.5.18 gcc/4.8.1
```

Run `install-libglui.sh` to download and extract GLUI.

## Usage

To build and run the executable:
```
make

./biodep-vis --ont_file go-basic.obo --json_file input.json
```

The program reads a json file and ontology file as input. We have provided both a sample json and ontology file for testing.

The json file must consist of two data types, graphs and alignments. The graph component requires tab seperated files as input and an assigned index for each graph. The alignment component requires a graph id as the input along with tab seperated alignment graph as output. When providing the graphs, the file must also contain coordinates for the graphs in 3d space.

Optionally, you can provide a cluster file and an ontology file. See `data/M.tab.cluster` and `data/Maize_info.txt` for examples.

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
vglrun -c proxy ./biodep-vis --ont_file FILE --json_file FILE
```

To stop the VNC server:
```
/opt/TurboVNC/bin/vncserver -kill :1
```

## Stream from Palmetto

Palmetto should already have VirtualGL and TurboVNC installed. However, because compute nodes cannot be accessed directly via SSH but only through the login node, you must tunnel the VNC server through the login node on a second SSH session.

Login to a GPU node on Palmetto and start a VNC server:
```
ssh -X <username>@login.palmetto.clemson.edu
qsub -I -l select=1:ngpus=1:ncpus=16:mem=32gb,walltime=02:00:00

LANG=C /opt/TurboVNC/bin/vncserver
```

Look for `TurboVNC: <node>:<port>` in the output. For example: `node0263:1`.

Login to another Palmetto session with tunnelling:
```
ssh -L 10000:<node>.palmetto.clemson.edu:<5900 + port> <username>@login.palmetto.clemson.edu

# example:
ssh -L 10000:node0263.palmetto.clemson.edu:5901 ksapra@login.palmetto.clemson.edu
```

Start a VNC client on your local machine and connect to `localhost:10000`.

Start BioDep-Vis on the GPU node:
```
vglrun -c proxy ./biodep-vis --ont_file FILE --json_file FILE
```

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
