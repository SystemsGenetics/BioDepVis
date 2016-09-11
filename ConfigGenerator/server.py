from PyQt4 import QtCore,QtGui # Import the PyQt4 module we'll need
from PyQt4.QtGui import QFileDialog
import sys # We need sys so that we can pass argv to QApplication

import design # This file holds our MainWindow and all design related things
              # it also keeps events etc that we defined in Qt Designer


import itertools
import json
import pdb

class G3NApp(QtGui.QMainWindow, design.Ui_MainWindow):
	networkNameList = {}
	alignmentNameList = {}
	jsonfilename = ""
	def __init__(self):
		# Explaining super is out of the scope of this article
		# So please google it if you're not familar with it
		# Simple reason why we use it here is that it allows us to
		# access variables, methods etc in the design.py file
		super(self.__class__, self).__init__()
		self.setupUi(self)  # This is defined in design.py file automatically                         # It sets up layout and widgets that are defined
		self.addNetwork.clicked.connect(self.addNewNetworkFunc)                           
		self.addAlignment.clicked.connect(self.addNewCustomAlignmentFunc)
		self.removeSelectedNetwork.clicked.connect(self.removeSelectedNetworkFunc)
		self.networkfileselection.clicked.connect(self.networkfileselectionFunc)
		self.ontologyfileselection.clicked.connect(self.ontologyfileselectionFunc)		
		self.finishbutton.clicked.connect(self.finishfunc)
		self.alignmentfolderbutton.clicked.connect(self.alignmentfolderbuttonFunc)
		self.jsonfolderbutton.clicked.connect(self.jsonfolderbuttonFunc)
		self.evalfilebutton.clicked.connect(self.evalfilebuttonFunc)
		




	def finishfunc(self):


		if self.alignmentfolder.text() == "" :
			errormsg = "Alignment Folder Name Not Provided"
			self.createErroMessageBox(errormsg)
			return


		if self.jsonfoldername.text() == "" :
			errormsg = "Json Folder Name Not Provided"
			self.createErroMessageBox(errormsg)
			return
		
		if self.networkNameList == {} :
			errormsg = "No Networks Added"
			self.createErroMessageBox(errormsg)
			return
		

		data = {}
		index = 1
		data["graph"] = {}
		data["alignment"] = {}
		graphdata = data["graph"]
		alignmentdata = data["alignment"]

		for item in self.networkNameList:
			
			graphdata["graph"+str(index)] = {}
			graphdata["graph"+str(index)]["id"] = index
			graphdata["graph"+str(index)]["name"]= item
			graphdata["graph"+str(index)]["fileLocation"] = str(self.networkNameList[item]['filename'])
			graphdata["graph"+str(index)]["Ontology"] = str(self.networkNameList[item]['fileloc'])
			graphdata["graph"+str(index)]["clusterLocation"] = str(self.networkNameList[item]['filename']) +'.cluster'
			graphdata["graph"+str(index)]["x"] = int(str(self.networkNameList[item]['xpos']))
			graphdata["graph"+str(index)]["y"] = int(str(self.networkNameList[item]['ypos']))
			graphdata["graph"+str(index)]["z"] = int(str(self.networkNameList[item]['zpos']))
			graphdata["graph"+str(index)]["w"] = 200
			graphdata["graph"+str(index)]["h"] = 200
			index = index + 1

		index = 1
		for item in self.alignmentNameList:
			name1,name2 = item.split("-")
			alignmentdata["alignment"+str(index)] = {}
			alignmentdata["alignment"+str(index)]["graphID1"] = self.networkNameList[name1]['id']
			alignmentdata["alignment"+str(index)]["graphID2"] = self.networkNameList[name2]['id']
			alignmentdata["alignment"+str(index)]["filelocation"] =  str(self.alignmentfolder.text()) + "/" + self.alignmentNameList[item]['alignment']
			alignmentdata["alignment"+str(index)]["evalfile"] = self.alignmentNameList[item]['evalfile']


		with open(str(self.jsonfoldername.text()) + "/" + self.jsonfilename[:-1]+".json", 'w') as txtfile:
			json.dump(data, txtfile, indent=4)
		exit(0)

	def networkfileselectionFunc(self):
		self.networkfilename.setText(QFileDialog.getOpenFileName())

	def ontologyfileselectionFunc(self):
		self.ontologyfilename.setText(QFileDialog.getOpenFileName())

	def jsonfolderbuttonFunc(self):
		self.jsonfoldername.setText(str(QFileDialog.getExistingDirectory(self, "Select Directory")))
		
	def alignmentfolderbuttonFunc(self):
		self.alignmentfolder.setText(str(QFileDialog.getExistingDirectory(self, "Select Directory")))		
	

	def evalfilebuttonFunc(self):
		self.evalfile.setText(QFileDialog.getOpenFileName())

	def createErroMessageBox(self,errormsg):
			msg = QtGui.QMessageBox()
			msg.setIcon(msg.Information)
			msg.setText(errormsg)
		 	msg.setStandardButtons(msg.Ok)
		 	retval = msg.exec_()		


	def addNewCustomAlignmentFunc(self):
		print 'Adding New Pair Given by User'
		'''
		if self.network1name.text() != "":
			self.createErroMessageBox("Network 1 Name Not Given")
			return ;
		if self.network2name.text() != "":
			self.createErroMessageBox("Network 2 Name Not Given")
			return ;
		'''
		n1name = str(self.network1name.currentText());
		n2name = str(self.network2name.currentText());

		print 'ALigning', n1name, 'and', n2name

		if n1name not in self.networkNameList:
			self.createErroMessageBox("Network 1 ("+ n1name+ ") Name not Present in Network List")
			return ;
		if n2name not in self.networkNameList:
			self.createErroMessageBox("Network 1 ("+ n2name+ ") Name not Present in Network List")
			return ;

		if n1name == n2name:
			self.createErroMessageBox("Cannot Align to self")
			return ;

		if self.alignmentname.text() == "":
			self.createErroMessageBox("Please Provide Alignment File Name")
			return 

		if self.evalfile.text() == "":
			self.createErroMessageBox("Please Provide Eval File")
			return 


		alignment = n1name+'-'+n2name 
		alignmentfilename = str(self.alignmentname.text())
		evalfilename = str(self.evalfile.text())
		if (alignment not in self.alignmentNameList):
			
			#item.setCheckState(1)

			#Add To List
			print 'Adding Alignment between', n1name, 'and', n2name
		
			self.alignmentNameList[alignment] = {}
			self.alignmentNameList[alignment]['alignment'] = alignmentfilename
			self.alignmentNameList[alignment]['evalfile'] = evalfilename

			self.alignmentTable.setRowCount(len(self.alignmentNameList))
			self.alignmentTable.setItem(len(self.alignmentNameList)-1,0,QtGui.QTableWidgetItem(n1name))
			self.alignmentTable.setItem(len(self.alignmentNameList)-1,1,QtGui.QTableWidgetItem(n2name))
			self.alignmentTable.setItem(len(self.alignmentNameList)-1,2,QtGui.QTableWidgetItem(alignmentfilename))
			self.alignmentTable.setItem(len(self.alignmentNameList)-1,3,QtGui.QTableWidgetItem(evalfilename))
			print 'Alignment add Compute'

		else:
			self.createErroMessageBox("ALignment Pair (" + alignment + ") already exsist")

	def removeSelectedNetworkFunc(self):
		print 'Removing Selected'
		errormsg = ""
		if self.removeRowId.text() == "":
			errormsg = "No Id Provided ( Please provided, Use Comma to seperate multiple)"
		if errormsg != "":
			self.createErroMessageBox(errormsg)
		else:
			listname = self.removeRowId.text().split(",")
			for item in listname:
				self.networkTable.removeRow(int(item)-1)	
				for itr in self.networkNameList:
					if self.networkNameList[itr]['id'] == int(item):
						
						del self.networkNameList[itr]
						print 'Removing', itr
						print self.network1name.findData(QtCore.QString(itr))
						self.network1name.removeItem(self.network1name.findData(QtCore.QString(itr)));
						self.network2name.removeItem(self.network2name.findData(QtCore.QString(itr)));
						break




	def addNewNetworkFunc(self):

		errormsg = ""
		

		if self.taxId.text() == "":
			 errormsg = "Name Field Not Provided for Adding Network"
		if self.name.text() == "":
			 errormsg = "Name Field Not Provided for Adding Network"
		if self.networkfilename.text() == "":
			 errormsg = "Filename  Field Not Provided for Adding Network"
		if self.ontologyfilename.text() == "":
			 errormsg = "File Location Field Not Provided for Adding Network"			 					 
		if self.xpos.text() == "":
			 errormsg = "Xpos Field Not Provided for Adding Network"
		if self.ypos.text() == "":
			 errormsg = "Ypos Field Not Provided for Adding Network"
		if self.zpos.text() == "":
			 errormsg = "Zpos Field Not Provided for Adding Network"			 
		if str(self.name.text()) in self.networkNameList:
			 errormsg = "Name Already in List, Please use a different name"



		if errormsg != "":
			self.createErroMessageBox(errormsg)
		else:
			name = str(self.name.text())
			#item =QtGui.QListWidgetItem(name)
			#item.setCheckState(1)

			#Add to Dict
			self.networkNameList[name] = {}
			self.networkNameList[name]['taxid'] = self.taxId.text()
			self.networkNameList[name]['id'] = len(self.networkNameList)
			self.networkNameList[name]['filename'] = self.networkfilename.text()
			self.networkNameList[name]['fileloc'] = self.ontologyfilename.text()
			self.networkNameList[name]['xpos'] = self.xpos.text()
			self.networkNameList[name]['ypos'] = self.ypos.text()
			self.networkNameList[name]['zpos'] = self.zpos.text()
			self.jsonfilename = self.jsonfilename + name + '-'
			print 'Add Complete'
			self.taxId.clear()
			self.name.clear()
			self.networkfilename.clear()
			self.ontologyfilename.clear()
			self.xpos.clear()
			self.ypos.clear()
			self.zpos.clear()

			self.networkTable.setRowCount(len(self.networkNameList))
			self.networkTable.setItem(len(self.networkNameList)-1,0,QtGui.QTableWidgetItem(self.networkNameList[name]['taxid']))
			self.networkTable.setItem(len(self.networkNameList)-1,1,QtGui.QTableWidgetItem(name))
			self.networkTable.setItem(len(self.networkNameList)-1,2,QtGui.QTableWidgetItem(self.networkNameList[name]['filename']))
			self.networkTable.setItem(len(self.networkNameList)-1,3,QtGui.QTableWidgetItem(self.networkNameList[name]['fileloc']))
			self.networkTable.setItem(len(self.networkNameList)-1,4,QtGui.QTableWidgetItem(self.networkNameList[name]['xpos']))
			self.networkTable.setItem(len(self.networkNameList)-1,5,QtGui.QTableWidgetItem(self.networkNameList[name]['ypos']))
			self.networkTable.setItem(len(self.networkNameList)-1,6,QtGui.QTableWidgetItem(self.networkNameList[name]['zpos']))

			self.network1name.addItem(name,QtCore.QString(name));
			self.network2name.addItem(name,QtCore.QString(name));


def main():
    app = QtGui.QApplication(sys.argv)  # A new instance of QApplication
    form = G3NApp()                 # We set the form to be our ExampleApp (design)
    form.show()                         # Show the form
    app.exec_()                         # and execute the app


if __name__ == '__main__':              # if we're running file directly and not importing it
    main()                              # run the main function