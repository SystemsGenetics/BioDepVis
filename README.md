# BioDep-Vis

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


