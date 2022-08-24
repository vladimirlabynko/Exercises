from argparse import Action
from tkinter.tix import Form
from PyQt5 import QtCore, QtGui, QtWidgets



import exercise


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.setFixedSize(1280, 720)
        
        self.horizontalLayout = QtWidgets.QHBoxLayout(Form)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")


        self.march_btn=QtWidgets.QPushButton(Form)
        self.march_btn.setObjectName("march_btn")

        self.good_m_btn = QtWidgets.QPushButton(Form)
        self.good_m_btn.setObjectName("good_m_btn")

        self.cabaret_btn = QtWidgets.QPushButton(Form)
        self.cabaret_btn.setObjectName("cabaret_btn")

        self.push_leg_btn = QtWidgets.QPushButton(Form)
        self.push_leg_btn.setObjectName("push_leg_btn")

        self.split_squat_btn = QtWidgets.QPushButton(Form)
        self.split_squat_btn.setObjectName("split_squat_btn")

        self.back_btn = QtWidgets.QPushButton(Form)
        self.back_btn.setObjectName("back_btn")

        self.verticalLayout.addWidget(self.march_btn)
        self.verticalLayout.addWidget(self.good_m_btn)
        self.verticalLayout.addWidget(self.cabaret_btn)
        self.verticalLayout.addWidget(self.push_leg_btn)
        self.verticalLayout.addWidget(self.split_squat_btn)
        self.verticalLayout.addWidget(self.back_btn)
        self.horizontalLayout.addLayout(self.verticalLayout)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)
        
        self.march_btn.clicked.connect(self.march)
        self.good_m_btn.clicked.connect(self.good_morn)
        self.cabaret_btn.clicked.connect(self.cabaret)
        self.push_leg_btn.clicked.connect(self.push_leg)
        self.split_squat_btn.clicked.connect(self.split_squat)
        self.back_btn.clicked.connect(lambda:self.closescr(Form))

    def closescr(self,Form):
        Form.close()

    def good_morn(self):
        #self.Form = QtWidgets.QWidget()
        
        self.exercise = exercise.Run_Good_Morning()
        self.exercise.show()

    def cabaret(self):
        #self.Form = QtWidgets.QWidget()
        
        self.exercise = exercise.Run_Cabaret()

        self.exercise.show()


    def march(self):
        #self.Form = QtWidgets.QWidget()
        
        self.exercise = exercise.Run_March()
        self.exercise.show()

    def push_leg(self):
        #self.Form = QtWidgets.QWidget()
        
        self.exercise = exercise.Run_Leg_Push()
        self.exercise.show()


    def split_squat(self):
        #self.Form = QtWidgets.QWidget()
        
        self.exercise = exercise.Run_Split_Squat()
        self.exercise.show()

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Choose Action"))
        self.march_btn.setText(_translate("Form", "March in place"))
        self.good_m_btn.setText(_translate("Form", "Good morning"))
        self.cabaret_btn.setText(_translate("Form", "Cabaret"))
        self.push_leg_btn.setText(_translate("Form", "Alternate leg behind"))
        self.split_squat_btn.setText(_translate("Form", "Split squat"))
        self.back_btn.setText(_translate("Form", "Back"))