# Configuration File Generator

## Run

Step 0 : Log into Palmetto with -X server
Windows: https://wiki.utdallas.edu/wiki/display/FAQ/X11+Forwarding+using+Xming+and+PuTTY
- install Xming
- Launch Xming: an icon will appear on your taskbar when it is running
- Customize putty session to enable X11 forwarding:  ssh --> enable X11 forwarding

Mac or Linux : ssh -X username@user.palmetto.clemson.edu

Step 1: qsub -X -I -l select=1:ncpus=8:mem=16gb:ngpus=1:gpu_model=k40,walltime=2:30:00
Step 2: module load anaconda/2.5.0
Step 3: cd ConfigGenerator
Step 4: To run use 'python server.py'

A large window should pop up outside the terminal. In the section labeled "Add New Network" the Taxon ID and organism name must be provided. Tab delimited Graph Id file must go in the blank area labeled "Network File", and the Ontology file provided can go in the slot labeled "Ontology File". For x, y, and z coordinates used recommended numbers (-100,0,0 for condition 1 and 100,0,0 for condition 2).


## Usage

A user interface is implemented for the ease of loading and visualizing.
The 'Add New Network' window allows the user to load a network (and ontology) file along with specifying the x, y, and z coordinate position for the network on the display window.
Multiple networks can be added and viewed in the central window. Networks can also be deleted when selected in the 'Networks Added' window and using 'Remove Selected' button.
In the 'Aligned Networks' window, the drop down selections can be used to select the networks in order to load and visualize their alignments. A bit score file for the purpose of calculating alignment can also be read into BioDep-Vis.
The Alignment folder provides the path to the location of the alignments networks, where as Json Folder points to the directory where the json file for the configuration is written.
