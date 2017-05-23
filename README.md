# BioDep-Vis
Watch our video demo showcasing some of the features at 
https://clemson.app.box.com/v/BioDepVis

## Requirements
- CUDA 5.0 or greater
- gcc 4.8 or greater

## Palmetto Usage

```
module load gcc/4.8.1 cuda-toolkit/7.5.18
```

## First Time Usage
In order to install glui, please run `./firsttime.sh` after downloading the source code. This installs GLUI-2.35.

## Compilation
To Compile the code go in to the folder and type `make`. To compile and create execuatable.

## Running
In order to run the code, type `vglrun ./G3NAV`. This reads an input.json file. We have provided a sample input.json.

The input.json consists of two components, graphs and alignments. Graph components require tab seperated file as input and an index assigned to these graphs. While the Alignment component requires graph id as the input along with tab seperated alignment graph as output. When providing the graphs please provide coordinates of these graphs in 3d space.

Optionally you can provide a cluster file tab seperated with (nodename)(tab)(clusterid). You can also provide an ontology file (see `data/Maize_info.txt` for sample). These files must be processed by the Configuraton Generator (provided in the BioDep-Vis exacutable folder).

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
vglrun ./G3NAV
```

To disconnect:
```
/opt/TurboVNC/bin/vncserver -kill :<port>

# example:
/opt/TurboVNC/bin/vncserver -kill :1
```

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
