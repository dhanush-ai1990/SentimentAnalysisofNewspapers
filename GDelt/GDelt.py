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
import csv
import xlrd

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


# Data Schema:
# -- Event ID and Date
# 0: GUID
# 1: Day (integer)  YYYYMMDD
# 2: MonthYear      YYYYMM
# 3: Year           YYYY
# 4: Fraction Date  YYYY.FFFF

# --- Actor 1 Attributes (May be blank if unidentified)
# 5:  a1code    -- complete raw cameo code for actor 1
# 6:  a1name    -- name of actor1
# 7:  a1cc      -- country (3-char cameo code)
# 8:  a1group   -- un, world bank, al-qaeda
# 9:  a1ethinic -- (probably wrong, ignore)
# 10: religion1
# 11: religion2 -- multiple religions
# 12: type1     -- role of actor (education, elites, media, refugee)
# 13: type2     -- multiple roles
# 14: type3     -- multiple roles

# -- Actor 2 Attributes (May be blank if unidentified)
# 15: a2code    -- complete raw cameo code for actor 1
# 16: a2name    -- name of actor1
# 17: a2cc      -- country (3-char cameo code)
# 18: a2group   -- un, world bank, al-qaeda
# 19: a2ethinic -- (probably wrong, ignore)
# 20: religion1
# 21: religion2 -- multiple religions
# 22: type1     -- role of actor (education, elites, media, refugee)
# 23: type2     -- multiple roles
# 24: type3     -- multiple roles

# -- Event action attributes
# 25: IsRootEvent
# 26: Event Code
# 27: Event Base code
# 28: Event Root code
# 29: QuadClass         -- Verbal Coop:1, Mat Coop:2, Verbal Con: 3, Mat Con: 4
# 30: GoldsteinScale    -- Stability of a country
# 31: NumMentions       -- Number of times mentioned across document
# 32: NumSources        -- Sources mentioning this
# 33: NumArticles       -- Number of articles mentioning this
# 34: AvgTone           -- Tone of document (-100) extremely neg, (+100) positive

# -- Actor 1 Event Geography
# 35: A1Geo_Type -- 1:country, 2:USSTATE, 3:USCITY, 4:WORLDCITY, 5:WORLDSTATE
# 36: A1Geo_Name -- Name of the place
# 37: A1Geo_CC   -- Country code of actor 1
# 38: A1Geo_AMD
# 39: A1Geo_lat
# 40: A1Geo_lon
# 41: A1Geo_Feature

# -- Actor 2 Event Geography
# 42: A1Geo_Type -- 1:country, 2:USSTATE, 3:USCITY, 4:WORLDCITY, 5:WORLDSTATE
# 43: A1Geo_Name -- Name of the place
# 44: A1Geo_CC   -- Country code of actor 1
# 45: A1Geo_AMD
# 46: A1Geo_lat
# 47: A1Geo_lon
# 48: A1Geo_Feature

# -- Data Management Fields
# 49: Date Added -- When it was added to the database
# 50: SourceURL  -- Where the data came from (not present prior to Apri 1 2013)

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

def loadGDeltFiles(local_storage="./GDELT_REPOSITORY"):
    """
    loadGDeltFiles

    :local_storage: where the downloaded files are stored
    :returns: csv for each file
    """
    pass
