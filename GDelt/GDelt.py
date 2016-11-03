# ------------------------------------------------
# GDelt.py
#
# Author: Evan Wilde          <etcwilde@uvic.ca>
# Date:   Oct 30 2016
#
# ------------------------------------------------

import csv
import lxml.html as lh
import math
import os
import requests
import sys
import threading
import urllib
import xlrd
import zipfile

from newspaper import Article
from newspaper.article import ArticleException

from progress.bar import Bar
from time import sleep

gdelt_base_url = 'http://data.gdeltproject.org/events/'

def getHeaders():
    """
    getHeaders

    Provides the data schema
    :returns: The schema of the data
    """
    fname = os.path.dirname(__file__) + '/CSV.header.fieldids.xlsx'
    xl_workbook = xlrd.open_workbook(fname)
    xl_sheet = xl_workbook.sheet_by_name("Sheet1")
    return [ xl_sheet.row(ridx)[0].value for ridx in range(1, xl_sheet.nrows)]

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

    if not os.path.isfile(local_storage + "extracted/" + fname.replace(".zip", "")):
        return []  # Extracted file doesn't exists, just ignore it


    # Parse the data

    values = []
    with open(local_storage + "extracted/" + fname.replace(".zip", "")) as f:
        schema = getHeaders()
        for l in f.read().split("\n"):
            block = l.split("\t")
            if len(block) == 1:
                continue
            values.append({h: (block[idx] if block[idx] != "" else None) for idx, h in enumerate(schema)})
    return values

def loadLinks(GDeltData, threadCount=8):
    """Downloads the links and provides the downloaded content for each URL

    :GDeltData: The dataset for gdelt, obtained from loadGDeltFile
    :threadCount: Number of threads to download and parse links over
    :returns: url-content mapping

    """

    def chunkify(links, threads):
        """
        Breaks the list into sub-components that can be fed into threads
        """
        return [links[i:i+math.ceil(len(links) / threads)]
                for i in range(0, len(links), math.ceil(len(links) / threads))]

    def handleLinks(links, results, index):
        """
        downloads, parses, and inserts links into the results
        """
        for linkid, link in enumerate(links):
            for url in link:
                link[url].download()
                completed[index] += 1
                if link[url].html == '':
                    break   # A server error occured, don't parse, it won't work
                link[url].parse()
        results[index] = [{key: link[key].text} for link in links for key in link]

    # Only process unique links
    unique_links = { d['SOURCEURL'] for d in GDeltData }
    links = [{link: Article(link)} for link in unique_links]
    linklists = chunkify(links, threadCount)

    print("Loading {0} links".format(len(links)))
    bar = Bar("Processing:", max=len(links))

    threadCount = threadCount if len(linklists) > threadCount else len(linklists)

    results = [None] * threadCount
    threads = [None] * threadCount
    completed =  [0] * threadCount

    for i in range(threadCount):
        threads[i] = threading.Thread(target=handleLinks, args=(linklists[i], results, i))
        threads[i].start()

    current_total = sum(completed)
    while sum(completed) < len(links):
        sleep(0.1)
        tmp_completed = sum(completed)
        bar.next(sum(completed) - current_total)
        if sum(completed) >= len(links):
            break
        current_total = tmp_completed
    bar.finish()
    for thread in threads:
        thread.join()
    return {key: link[key] for resultsset in results for link in resultsset for key in link}
