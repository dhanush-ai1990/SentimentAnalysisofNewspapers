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
files = gd.loadGDeltFileList()[0:3]
print(files)
for f in files:
    gd.downloadGDeltFile(f)  # Download the file onto the computer
    gd.loadGDeltFile(f)      # Extracts the file and gives us a dictionary to work with

