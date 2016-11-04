#!/bin/env python3

# ------------------------------------------------
# LoadGDelt.py
#
# Author: Evan Wilde          <etcwilde@uvic.ca>
# Date:   Oct 30 2016
#
# ------------------------------------------------

import GDelt.GDelt as gd
import requests
from lxml import html
from newspaper import Article

# We'll download the first 3 files and play with those for now
files = gd.loadGDeltFileList()[:2]
for idx, f in enumerate(files):
    print("Processing file", idx + 1)
    gd.downloadGDeltFile(f)  # Download the file onto the computer
    data = gd.loadGDeltFile(f)[:50]  # Start with the first 50 links
    articles = gd.downloadLinks(data, 50)  # get the articles using 50 threads
    print(articles)
