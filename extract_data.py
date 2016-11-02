import newspaper
from newspaper import Article
#File to extract links from the csv file
import pandas as pd
import urlparse
import requests
import bs4
import re
from pandas import DataFrame, read_csv
from urlparse import urlparse
from bs4 import BeautifulSoup
#Location of the csv file
file_loc='/Users/Dhanush/Desktop/Projects/DM_project/DMProj_Data/event11.csv'
temp_arr=[]
file_count=0

#Reading the csv file
try:
    gdelt_event_data=pd.read_csv(file_loc)
except IOError as err:
    print "Error while reading csv file"
    print format(err)
    exit()


#Remove not necessary data. Keep EventCode, EventRootCode and SOURCEURL

for col in gdelt_event_data:
    if col!='EventCode' and col!='EventRootCode' and col!='SOURCEURL':
        gdelt_event_data=gdelt_event_data.drop(col,axis=1)

#Retrieve the url hostnames from the DataFrame

for ele in gdelt_event_data['SOURCEURL']:

    url=urlparse(ele)
    temp_arr.append(url.netloc)

link_data=pd.DataFrame(data=temp_arr,columns=['host'])

#Counting unique data
count_data= pd.value_counts(link_data['host'].values,sort=True)


#Parse the links provided from the dataframe and append the data to the existing dataframe
#Create a new dataframe and write it to the file system
for link in gdelt_event_data[
    'SOURCEURL']:

    try:
        temp_arr=[]
        temp_text=''
        temp_text=link+"\n"
        temp_text=temp_text+str(gdelt_event_data['EventCode'][file_count])+"\n"
        temp_text=temp_text+str(gdelt_event_data['EventRootCode'][file_count])+"\n"
        print link

        a = Article(link)
        a.download()
        a.parse()

        f=open('/Users/Dhanush/Desktop/Projects/DM_project/DMProj_Data/CODE_11/''_'+str(file_count)+'.txt','w')
        f.write(temp_text+a.text.encode("utf-8"))
        file_count+=1
        temp_df=None

    except Exception as err:
        print "An error occured while rerieving the article link"
        print format(err)
