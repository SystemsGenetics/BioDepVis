# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'design.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(1242, 673)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.finishbutton = QtGui.QPushButton(self.centralwidget)
        self.finishbutton.setGeometry(QtCore.QRect(210, 600, 841, 32))
        self.finishbutton.setObjectName(_fromUtf8("finishbutton"))
        self.removeSelectedNetwork = QtGui.QPushButton(self.centralwidget)
        self.removeSelectedNetwork.setGeometry(QtCore.QRect(600, 390, 111, 51))
        self.removeSelectedNetwork.setObjectName(_fromUtf8("removeSelectedNetwork"))
        self.addAlignment = QtGui.QPushButton(self.centralwidget)
        self.addAlignment.setGeometry(QtCore.QRect(1130, 80, 110, 41))
        self.addAlignment.setObjectName(_fromUtf8("addAlignment"))
        self.label_7 = QtGui.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(830, 60, 111, 21))
        self.label_7.setObjectName(_fromUtf8("label_7"))
        self.label_8 = QtGui.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(830, 90, 111, 21))
        self.label_8.setObjectName(_fromUtf8("label_8"))
        self.addNetwork = QtGui.QPushButton(self.centralwidget)
        self.addNetwork.setGeometry(QtCore.QRect(80, 280, 110, 32))
        self.addNetwork.setObjectName(_fromUtf8("addNetwork"))
        self.line = QtGui.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(290, 20, 20, 471))
        self.line.setFrameShape(QtGui.QFrame.VLine)
        self.line.setFrameShadow(QtGui.QFrame.Sunken)
        self.line.setObjectName(_fromUtf8("line"))
        self.line_2 = QtGui.QFrame(self.centralwidget)
        self.line_2.setGeometry(QtCore.QRect(810, 40, 20, 291))
        self.line_2.setFrameShape(QtGui.QFrame.VLine)
        self.line_2.setFrameShadow(QtGui.QFrame.Sunken)
        self.line_2.setObjectName(_fromUtf8("line_2"))
        self.line_3 = QtGui.QFrame(self.centralwidget)
        self.line_3.setGeometry(QtCore.QRect(70, 510, 1201, 20))
        self.line_3.setFrameShape(QtGui.QFrame.HLine)
        self.line_3.setFrameShadow(QtGui.QFrame.Sunken)
        self.line_3.setObjectName(_fromUtf8("line_3"))
        self.label_4 = QtGui.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(90, 20, 111, 16))
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.label_9 = QtGui.QLabel(self.centralwidget)
        self.label_9.setGeometry(QtCore.QRect(490, 20, 131, 16))
        self.label_9.setObjectName(_fromUtf8("label_9"))
        self.label_10 = QtGui.QLabel(self.centralwidget)
        self.label_10.setGeometry(QtCore.QRect(1000, 20, 101, 16))
        self.label_10.setObjectName(_fromUtf8("label_10"))
        self.zpos = QtGui.QLineEdit(self.centralwidget)
        self.zpos.setGeometry(QtCore.QRect(140, 240, 127, 21))
        self.zpos.setObjectName(_fromUtf8("zpos"))
        self.label_11 = QtGui.QLabel(self.centralwidget)
        self.label_11.setGeometry(QtCore.QRect(20, 250, 35, 16))
        self.label_11.setObjectName(_fromUtf8("label_11"))
        self.networkfilename = QtGui.QLineEdit(self.centralwidget)
        self.networkfilename.setGeometry(QtCore.QRect(140, 120, 121, 21))
        self.networkfilename.setObjectName(_fromUtf8("networkfilename"))
        self.networkTable = QtGui.QTableWidget(self.centralwidget)
        self.networkTable.setGeometry(QtCore.QRect(310, 40, 491, 281))
        self.networkTable.setObjectName(_fromUtf8("networkTable"))
        self.networkTable.setColumnCount(7)
        self.networkTable.setRowCount(0)
        item = QtGui.QTableWidgetItem()
        self.networkTable.setHorizontalHeaderItem(0, item)
        item = QtGui.QTableWidgetItem()
        self.networkTable.setHorizontalHeaderItem(1, item)
        item = QtGui.QTableWidgetItem()
        self.networkTable.setHorizontalHeaderItem(2, item)
        item = QtGui.QTableWidgetItem()
        self.networkTable.setHorizontalHeaderItem(3, item)
        item = QtGui.QTableWidgetItem()
        self.networkTable.setHorizontalHeaderItem(4, item)
        item = QtGui.QTableWidgetItem()
        self.networkTable.setHorizontalHeaderItem(5, item)
        item = QtGui.QTableWidgetItem()
        self.networkTable.setHorizontalHeaderItem(6, item)
        self.label = QtGui.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(16, 89, 86, 16))
        self.label.setObjectName(_fromUtf8("label"))
        self.label_5 = QtGui.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(16, 183, 35, 16))
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.label_2 = QtGui.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(16, 120, 91, 16))
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.label_3 = QtGui.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(16, 152, 131, 16))
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.label_6 = QtGui.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(16, 215, 35, 16))
        self.label_6.setObjectName(_fromUtf8("label_6"))
        self.name = QtGui.QLineEdit(self.centralwidget)
        self.name.setGeometry(QtCore.QRect(140, 89, 127, 21))
        self.name.setObjectName(_fromUtf8("name"))
        self.ypos = QtGui.QLineEdit(self.centralwidget)
        self.ypos.setGeometry(QtCore.QRect(140, 213, 127, 21))
        self.ypos.setObjectName(_fromUtf8("ypos"))
        self.xpos = QtGui.QLineEdit(self.centralwidget)
        self.xpos.setGeometry(QtCore.QRect(140, 182, 127, 21))
        self.xpos.setObjectName(_fromUtf8("xpos"))
        self.ontologyfilename = QtGui.QLineEdit(self.centralwidget)
        self.ontologyfilename.setGeometry(QtCore.QRect(140, 150, 121, 21))
        self.ontologyfilename.setObjectName(_fromUtf8("ontologyfilename"))
        self.networkfileselection = QtGui.QPushButton(self.centralwidget)
        self.networkfileselection.setGeometry(QtCore.QRect(260, 120, 31, 21))
        self.networkfileselection.setObjectName(_fromUtf8("networkfileselection"))
        self.ontologyfileselection = QtGui.QPushButton(self.centralwidget)
        self.ontologyfileselection.setGeometry(QtCore.QRect(260, 150, 31, 21))
        self.ontologyfileselection.setObjectName(_fromUtf8("ontologyfileselection"))
        self.label_12 = QtGui.QLabel(self.centralwidget)
        self.label_12.setGeometry(QtCore.QRect(280, 560, 101, 16))
        self.label_12.setObjectName(_fromUtf8("label_12"))
        self.label_13 = QtGui.QLabel(self.centralwidget)
        self.label_13.setGeometry(QtCore.QRect(300, 580, 81, 16))
        self.label_13.setObjectName(_fromUtf8("label_13"))
        self.alignmentfolder = QtGui.QLineEdit(self.centralwidget)
        self.alignmentfolder.setGeometry(QtCore.QRect(380, 560, 581, 21))
        self.alignmentfolder.setObjectName(_fromUtf8("alignmentfolder"))
        self.jsonfoldername = QtGui.QLineEdit(self.centralwidget)
        self.jsonfoldername.setGeometry(QtCore.QRect(380, 580, 581, 21))
        self.jsonfoldername.setObjectName(_fromUtf8("jsonfoldername"))
        self.alignmentfolderbutton = QtGui.QPushButton(self.centralwidget)
        self.alignmentfolderbutton.setGeometry(QtCore.QRect(960, 560, 31, 21))
        self.alignmentfolderbutton.setObjectName(_fromUtf8("alignmentfolderbutton"))
        self.jsonfolderbutton = QtGui.QPushButton(self.centralwidget)
        self.jsonfolderbutton.setGeometry(QtCore.QRect(960, 580, 31, 21))
        self.jsonfolderbutton.setObjectName(_fromUtf8("jsonfolderbutton"))
        self.alignmentname = QtGui.QLineEdit(self.centralwidget)
        self.alignmentname.setGeometry(QtCore.QRect(950, 120, 181, 21))
        self.alignmentname.setText(_fromUtf8(""))
        self.alignmentname.setObjectName(_fromUtf8("alignmentname"))
        self.label_14 = QtGui.QLabel(self.centralwidget)
        self.label_14.setGeometry(QtCore.QRect(830, 120, 111, 21))
        self.label_14.setObjectName(_fromUtf8("label_14"))
        self.label_15 = QtGui.QLabel(self.centralwidget)
        self.label_15.setGeometry(QtCore.QRect(20, 60, 86, 16))
        self.label_15.setObjectName(_fromUtf8("label_15"))
        self.taxId = QtGui.QLineEdit(self.centralwidget)
        self.taxId.setGeometry(QtCore.QRect(140, 60, 127, 21))
        self.taxId.setObjectName(_fromUtf8("taxId"))
        self.removeRowId = QtGui.QLineEdit(self.centralwidget)
        self.removeRowId.setGeometry(QtCore.QRect(530, 400, 71, 21))
        self.removeRowId.setObjectName(_fromUtf8("removeRowId"))
        self.label_16 = QtGui.QLabel(self.centralwidget)
        self.label_16.setGeometry(QtCore.QRect(420, 400, 101, 21))
        self.label_16.setObjectName(_fromUtf8("label_16"))
        self.alignmentTable = QtGui.QTableWidget(self.centralwidget)
        self.alignmentTable.setGeometry(QtCore.QRect(830, 200, 401, 251))
        self.alignmentTable.setObjectName(_fromUtf8("alignmentTable"))
        self.alignmentTable.setColumnCount(4)
        self.alignmentTable.setRowCount(0)
        item = QtGui.QTableWidgetItem()
        self.alignmentTable.setHorizontalHeaderItem(0, item)
        item = QtGui.QTableWidgetItem()
        self.alignmentTable.setHorizontalHeaderItem(1, item)
        item = QtGui.QTableWidgetItem()
        self.alignmentTable.setHorizontalHeaderItem(2, item)
        item = QtGui.QTableWidgetItem()
        self.alignmentTable.setHorizontalHeaderItem(3, item)
        self.label_17 = QtGui.QLabel(self.centralwidget)
        self.label_17.setGeometry(QtCore.QRect(830, 150, 111, 21))
        self.label_17.setObjectName(_fromUtf8("label_17"))
        self.evalfile = QtGui.QLineEdit(self.centralwidget)
        self.evalfile.setGeometry(QtCore.QRect(950, 150, 181, 21))
        self.evalfile.setText(_fromUtf8(""))
        self.evalfile.setObjectName(_fromUtf8("evalfile"))
        self.evalfilebutton = QtGui.QPushButton(self.centralwidget)
        self.evalfilebutton.setGeometry(QtCore.QRect(1130, 150, 31, 21))
        self.evalfilebutton.setObjectName(_fromUtf8("evalfilebutton"))
        self.network1name = QtGui.QComboBox(self.centralwidget)
        self.network1name.setGeometry(QtCore.QRect(950, 60, 181, 26))
        self.network1name.setObjectName(_fromUtf8("network1name"))
        self.network2name = QtGui.QComboBox(self.centralwidget)
        self.network2name.setGeometry(QtCore.QRect(950, 90, 181, 26))
        self.network2name.setObjectName(_fromUtf8("network2name"))
        self.label.raise_()
        self.label_5.raise_()
        self.label_2.raise_()
        self.label_3.raise_()
        self.label_6.raise_()
        self.ypos.raise_()
        self.xpos.raise_()
        self.ontologyfilename.raise_()
        self.finishbutton.raise_()
        self.removeSelectedNetwork.raise_()
        self.addAlignment.raise_()
        self.label_7.raise_()
        self.label_8.raise_()
        self.addNetwork.raise_()
        self.name.raise_()
        self.line.raise_()
        self.line_2.raise_()
        self.line_3.raise_()
        self.label_4.raise_()
        self.label_9.raise_()
        self.label_10.raise_()
        self.zpos.raise_()
        self.label_11.raise_()
        self.networkfilename.raise_()
        self.networkTable.raise_()
        self.networkfileselection.raise_()
        self.ontologyfileselection.raise_()
        self.label_12.raise_()
        self.label_13.raise_()
        self.alignmentfolder.raise_()
        self.jsonfoldername.raise_()
        self.alignmentfolderbutton.raise_()
        self.jsonfolderbutton.raise_()
        self.alignmentname.raise_()
        self.label_14.raise_()
        self.label_15.raise_()
        self.taxId.raise_()
        self.removeRowId.raise_()
        self.label_16.raise_()
        self.alignmentTable.raise_()
        self.label_17.raise_()
        self.evalfile.raise_()
        self.evalfilebutton.raise_()
        self.network1name.raise_()
        self.network2name.raise_()
        MainWindow.setCentralWidget(self.centralwidget)
        self.actionExit = QtGui.QAction(MainWindow)
        self.actionExit.setObjectName(_fromUtf8("actionExit"))

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.finishbutton.setText(_translate("MainWindow", "Finish", None))
        self.removeSelectedNetwork.setText(_translate("MainWindow", "Remove\n"
"Selected", None))
        self.addAlignment.setText(_translate("MainWindow", "Add\n"
"Alignment", None))
        self.label_7.setText(_translate("MainWindow", "Id 1 Name", None))
        self.label_8.setText(_translate("MainWindow", "Id 2 Name", None))
        self.addNetwork.setText(_translate("MainWindow", "Add Network", None))
        self.label_4.setText(_translate("MainWindow", "Add New Network", None))
        self.label_9.setText(_translate("MainWindow", "Networks Added", None))
        self.label_10.setText(_translate("MainWindow", "Aligned Networks", None))
        self.label_11.setText(_translate("MainWindow", "Z Pos", None))
        item = self.networkTable.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "TaxID", None))
        item = self.networkTable.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "Name", None))
        item = self.networkTable.horizontalHeaderItem(2)
        item.setText(_translate("MainWindow", "File Location", None))
        item = self.networkTable.horizontalHeaderItem(3)
        item.setText(_translate("MainWindow", "Ontology Location", None))
        item = self.networkTable.horizontalHeaderItem(4)
        item.setText(_translate("MainWindow", "x", None))
        item = self.networkTable.horizontalHeaderItem(5)
        item.setText(_translate("MainWindow", "y", None))
        item = self.networkTable.horizontalHeaderItem(6)
        item.setText(_translate("MainWindow", "z", None))
        self.label.setText(_translate("MainWindow", "Network Name", None))
        self.label_5.setText(_translate("MainWindow", "X Pos", None))
        self.label_2.setText(_translate("MainWindow", "Network File", None))
        self.label_3.setText(_translate("MainWindow", "Ontology File", None))
        self.label_6.setText(_translate("MainWindow", "Y Pos", None))
        self.networkfileselection.setText(_translate("MainWindow", "..", None))
        self.ontologyfileselection.setText(_translate("MainWindow", "..", None))
        self.label_12.setText(_translate("MainWindow", "Alignment Folder", None))
        self.label_13.setText(_translate("MainWindow", "Json Folder", None))
        self.alignmentfolderbutton.setText(_translate("MainWindow", "..", None))
        self.jsonfolderbutton.setText(_translate("MainWindow", "..", None))
        self.label_14.setText(_translate("MainWindow", "Alignment Name", None))
        self.label_15.setText(_translate("MainWindow", "Tax Id", None))
        self.label_16.setText(_translate("MainWindow", "Remove Row Id", None))
        item = self.alignmentTable.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "Id 1", None))
        item = self.alignmentTable.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "id 2", None))
        item = self.alignmentTable.horizontalHeaderItem(2)
        item.setText(_translate("MainWindow", "alignmentfile", None))
        item = self.alignmentTable.horizontalHeaderItem(3)
        item.setText(_translate("MainWindow", "eval file", None))
        self.label_17.setText(_translate("MainWindow", "Eval Filename", None))
        self.evalfilebutton.setText(_translate("MainWindow", "..", None))
        self.actionExit.setText(_translate("MainWindow", "Exit", None))


if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    MainWindow = QtGui.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

