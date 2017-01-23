
import os
import random
loc='/Users/Dhanush/Desktop/Projects/DM_project/DMProj_Data/Data_train/CODE_'
output_loc_train='/Users/Dhanush/Desktop/Projects/DM_project/DMProj_Data/Data_Final/Train/Class_'
output_loc_test ='/Users/Dhanush/Desktop/Projects/DM_project/DMProj_Data/Data_Final/Test/Class_'
count_0=0
count_1=0

class_0_list = [a for a in range(15948)]
random.shuffle(class_0_list)
class_1_list = [b for b in range(14311)]
random.shuffle(class_1_list)
bound =int(len(class_0_list)*.80)
class_0_train = class_0_list[:bound]
class_0_test = class_0_list[bound:]
bound =int(len(class_1_list)*.80)
class_1_train =class_1_list[:bound]
print len(class_1_train)
print class_1_train[1:5]
class_1_test = class_1_list[bound:]

for i in range(1,21,1):
	red_loc='/Users/Dhanush/Desktop/Projects/DM_project/DMProj_Data/Data_train/CODE_' + str(i)
	for file in os.listdir(red_loc):
		if file.endswith(".csv")or file.endswith(".txt"):
			input1=red_loc + '/' + file 
			f1=open(input1,'r')
			data = f1.read()
			temp = [20,13,18,15,19]
			if i in temp:
				if count_0 in class_0_train:
					file_to_write=output_loc_train + str(0) + '/'+str(count_0)+ '.txt'
				else:
					file_to_write=output_loc_test + str(0) + '/'+str(count_0)+ '.txt'
				output=open(file_to_write, "w")
				output.write(data)
				count_0+=1
			else:
				if count_1 in class_1_train:
					file_to_write=output_loc_train + str(1) + '/'+str(count_1)+ '.txt'
				else:
					file_to_write=output_loc_test + str(1) + '/'+str(count_1)+ '.txt'
				output = open(file_to_write, "w")
				output.write(data)
				count_1 +=1










