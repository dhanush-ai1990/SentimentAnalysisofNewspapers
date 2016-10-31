#!/bin/env python3

# ------------------------------------------------
# LoadGDelt.py
#
# Author: Evan Wilde          <etcwilde@uvic.ca>
# Date:   Oct 30 2016
#
# ------------------------------------------------

import GDelt.GDelt as gd

# We'll download the first 3 files and play with those for now
files = gd.loadGDeltFileList()[0:2]
for f in files:
    gd.downloadGDeltFile(f)  # Download the file onto the computer
    data = gd.loadGDeltFile(f)
    print(f, data[0])

