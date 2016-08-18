# BioDep-Vis

Requirement: 
CUDA:5.0 or greater
gcc 4.8 or greater

#Palmetto Usage
module load gcc/4.8.1
module load cuda-toolkit/7.5.18

#First Time Usage
In order to install glui, please run ./firsttime.sh after downloading the source code. This installs GLUI-2.35.

#Compilation
To Compile the code go in to the folder and type make. To compile and create execuatable

#Running

In order to run the code, type ./G3NAV.exe. This reads an input.json file. We have provided a sample input.json. 
The input.json consist of two components, graphs and alignments. Graph components require tab seperated file as input and an index assigned to these graphs. While the Alignment component requires graph id as the input along with tab seperated alignment graph as output. When providing the graphs please provide coordinates of these graphs in 3d space.

Optionally you can provide a cluster file tab seperated with (nodename)(tab)(clusterid). You can also provide an ontology file (see data/Maize_info.txt for sample).

Graph Id Example
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



Alignment Id Example

    "alignment": {
        "alignment1": {
            "graphID1": 1,
            "graphID2": 2,
            "filelocation": "./data/alignment.output"
        }


#Visualizing Using VNC

1. Download Putty  and TurboVNC
Putty : http://the.earth.li/~sgtatham/putty/latest/x86/putty.exe
TigerVNC: http://sourceforge.net/projects/turbovnc/?source=typ_redirect

2. Create a connection to palmetto -> (user.palmetto.clemson.edu)

3. open interactive session to node with GPU -> qsub -I -l select=1:ngpus=1:ncpus=16:mem=32gb,walltime=02:00:00

4. Launch VNC on the node -> /opt/TurboVNC/bin/vncserver
4.a If you launching for firstime you may have to set a vnc password, which you provide as anything you want

5. Look for "TurboVNC: node<nodenumber:portno>"  [(node0263:1)]

6. Launch Another Session of Putty->Go to SSH->Tunnelling

7. Add A source Port as any number > 10000

8. Add Destination node<nodenumber>.palmetto.clemson.edu<590<portno> [node0263.palmetto.clemson.edu:5901]

9. Click 'Add'

10. Go back to 'Logging' and Log into Palmetto to activate this forwarding using Step 2

11. Open TigerVNC

12: Connect to 'localhost:<source port>, with the soruce port you mentioned above'  [localhost:10000]

13. Done

14: To disconnect, please use '/opt/TurboVNC/bin/vncserver  -kill :<portno>' [/opt/TurboVNC/bin/vncserver  -kill :1]

//For Linux Please replace step 6 to step 10 with following
ssh -L <sourceport> node<nodenumber>.palmetto.clemson.edu:<590<portno>> username@user.palmetto.clemson.edu      [ssh -L 10000:node0263.palmetto.clemson.edu:5901  ksapra@user.palmetto.clemson.edu]


