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
file_loc='C:\Users\Dell PC\Desktop\GDELT Data\Express Intent to Cooperate(03)\\20161030191601.13715.events.csv'
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
for link in gdelt_event_data['SOURCEURL']:

    try:
        temp_arr=[]
        temp_text=''
        print link
        html_page_stream=requests.get(link)
        html_page=BeautifulSoup(html_page_stream.content,'html.parser')

        html_p_data=html_page.find_all('p',text=True)
        print html_p_data

        for info in html_p_data:
            temp_text+=info.string.encode('utf-8')
        if len(temp_text)>50:
            temp_arr.append(temp_text)
            temp_df=DataFrame(data=temp_arr)
            temp_df.to_csv('C:\Users\Dell PC\Desktop\GDELT Data\Express Intent to Cooperate(03)\Raw Files\\'+'_'+str(file_count)+'.txt')
            file_count+=1
        temp_df=None

    except Exception as err:
        print "An error occured while rerieving the article link"
        print format(err)