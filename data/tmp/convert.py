import sys
filename = sys.argv[1]

f = open(filename)
lines = f.readlines()
f.close()

clusterid=1
for line in lines:
	line = line.replace("\n","")
	nodes = line.split("\t")
	for node in nodes:
		print "%s\t%d" % (node,clusterid)
	clusterid = clusterid + 1
