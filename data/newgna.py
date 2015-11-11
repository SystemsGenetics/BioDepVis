import sys
inputfilename = sys.argv[1]
outputfilename = sys.argv[2]

f = open(inputfilename,'r')
lines = f.readlines()
f.close()

nodelist = []
for line in lines:
	line = line.replace("\n","")
	[a,b] = line.split()
	if a not in nodelist:
		nodelist.append(a)
	if b not in nodelist:
		nodelist.append(b)


f = open(outputfilename,'w')
for node in nodelist:
	f.write("%s\t%s\n" % (node,node))
f.close()

