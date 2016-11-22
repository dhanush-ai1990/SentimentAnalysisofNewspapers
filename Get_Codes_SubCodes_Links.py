import os
red_loc='/Users/Dhanush/Desktop/Projects/DM_project/DMProj_Data/Data_Final/Train/Class_0'
red_loc1='/Users/Dhanush/Desktop/Projects/DM_project/DMProj_Data/Data_Final/Train/Class_1'
output1=open('/Users/Dhanush/Desktop/links_train.txt', "w")
output2=open('/Users/Dhanush/Desktop/subcodes_train.txt', "w")
output3=open('/Users/Dhanush/Desktop/code_train.txt', "w")
for file in os.listdir(red_loc):
	if file.endswith(".csv")or file.endswith(".txt"):
		input1=red_loc + '/' + file 
		f1=open(input1,'r')
		output1.write(f1.readline())
		output2.write(f1.readline())
		output3.write(f1.readline())

for file in os.listdir(red_loc1):
	if file.endswith(".csv")or file.endswith(".txt"):
		input1=red_loc + '/' + file 
		f1=open(input1,'r')
		output1.write(f1.readline())
		output2.write(f1.readline())
		output3.write(f1.readline())