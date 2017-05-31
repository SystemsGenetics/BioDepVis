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

vglrun ./G3NAV --ont_file FILE --json_file FILE
```

The program reads a json file and ontology file as input. We have provided both a sample json and ontology file for testing.

The json file must consist of two data types, graphs and alignments. The graph component requires tab seperated files as input and an assigned index for each graph. The alignment component requires a graph id as the input along with tab seperated alignment graph as output. When providing the graphs, the file must also contain coordinates for the graphs in 3d space.

Optionally, you can provide a cluster file that is tab seperated with (nodename)(tab)(clusterid). You can also provide an ontology file (see `data/Maize_info.txt` for sample). These files must be processed by the configuraton generator (provided in the BioDep-Vis exacutable folder).

Graph Id Example

```
"graph1": {
    "id": 1,
    "name": "Maize",
    "fileLocation": "./data/M.tab",
    "clusterLocation": "./data/M.tab.cluster",
    "Ontology" : "./data/Maize_info.txt",
    "x": -100,
    "y": 0,
    "z": 0,
    "w": 200,
    "h": 200
},
"graph2": {
    "id": 2,
    "name": "Rice",
    "fileLocation": "./data/R.tab",
    "clusterLocation": "./data/R.tab.cluster",
    "Ontology" : "./data/Rice_info.txt",
    "x": 100,
    "y": 0,
    "z": 0,
    "w": 200,
    "h": 200
}
```

Alignment Id Example

```
"alignment": {
    "alignment1": {
        "graphID1": 1,
        "graphID2": 2,
        "filelocation": "./data/alignment.output"
    }
}
```

## Stream from Palmetto using VNC

Login to a GPU node on Palmetto and launch a VNC server:
```
ssh -X <username>@user.palmetto.clemson.edu
qsub -I -l select=1:ngpus=1:ncpus=16:mem=32gb,walltime=02:00:00

/opt/TurboVNC/bin/vncserver -geometry 1920x1080
```

When you launch a VNC server for the first time you may be asked to set a password, which can be whatever you want. Look for `TurboVNC: <node>:<port>`. For example: `node0263:1`.

Login to another Palmetto session with tunnelling:
```
ssh -L 10000:<node>.palmetto.clemson.edu:<5900 + port> <username>@user.palmetto.clemson.edu

# example:
ssh -L 10000:node0263.palmetto.clemson.edu:5901 ksapra@user.palmetto.clemson.edu
```

Install a VNC client on your machine, such as `Vinagre` on Ubuntu. Connect to `localhost:10000`.

To run the visualizer on the GPU node:
```
vglrun ./G3NAV --ont_file FILE --json_file FILE
```

To disconnect:
```
/opt/TurboVNC/bin/vncserver -kill :<port>

# example:
/opt/TurboVNC/bin/vncserver -kill :1
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

## Control Keys for G3NAV

Space - Force Directed Layout

Q - Zoom Out

W - Rotate on y axis

E - Zoom In

R - Reset View

U,A,J - Pan Down View

O - Pan Down View (slow)

I - Pan Left View

S,K - Pan Right View

D - Rotate on X axis and zoom out

L - Pan Up View

X - Show only selected nodes

V - change Edge Design (curved to 2D)

, - Show node type
