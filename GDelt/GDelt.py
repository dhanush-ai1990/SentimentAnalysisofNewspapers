# ------------------------------------------------
# GDelt.py
#
# Author: Evan Wilde          <etcwilde@uvic.ca>
# Date:   Oct 30 2016
#
# ------------------------------------------------

import requests
import lxml.html as lh
import os
import urllib
import zipfile

gdelt_base_url = 'http://data.gdeltproject.org/events/'

def loadGDeltFileList():
    """
    Gets the file list from the GDELT project
    """
    page = requests.get(gdelt_base_url + 'index.html')
    doc = lh.fromstring(page.content)
    links = doc.xpath("//*/ul/li/a/@href")
    return [x for x in links if str.isdigit(x[0:4])]

def downloadGDeltFile(fname, local_storage="./GDELT_REPOSITORY/"):
    """
    Downloads the csv zip file from the GDELT project

    Don't provide the full link, only the file itself.

    :fname: file name
    :local_storage: Where the downloaded file is stored
    :returns: The contents of the file

    """
    if not os.path.isfile(local_storage + fname):
        if not os.path.isdir(local_storage):
            os.mkdir(local_storage)
        print("Downloading", local_storage + fname)
        urllib.request.urlretrieve(url=gdelt_base_url + fname, filename = local_storage + fname)
        print("File Downloaded")

def loadGDeltFile(fname, local_storage="./GDELT_REPOSITORY/"):
    """
    Extracts the file from the GDELT and converts it into a csv object

    :fname: filename
    :local_storage: where the downloaded files are stored
    :returns: csv object of contents

    """
    if os.path.isfile(local_storage + fname) and not os.path.isfile(local_storage + "extracted/" + fname.replace(".zip", "")):
        z = zipfile.ZipFile(file=local_storage + fname, mode='r')
        z.extractall(path=local_storage + "extracted/")

def loadGDeltFiles(local_storage="./GDELT_REPOSITORY"):
    """
    loadGDeltFiles

    :local_storage: where the downloaded files are stored
    :returns: csv for each file
    """
    pass
