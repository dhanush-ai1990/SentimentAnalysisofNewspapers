# ------------------------------------------------
# Utils.py
#
# Author: Evan Wilde          <etcwilde@uvic.ca>
# Date:   Nov 04 2016
#
# ------------------------------------------------

import math
import urllib
import urllib.parse


def chunkify(l, chunks):
        """
        Breaks the list into sub-components that can be fed into threads

        :l: list to be broken into chunks
        :chunks: Number of chunks to break list into
        :returns: list of at most "chunks" lists

        """
        return [l[i:i+math.ceil(len(l) / chunks)]
                for i in range(0, len(l), math.ceil(len(l) / chunks))]

def genFnameFromURL(url, extension="txt"):
    """Converts a url to a filename

    :url: url
    :extension: The file extension [txt]
    :returns: The filename

    """
    parse = urllib.parse.urlparse(url)
    domain = parse.netloc.split('.')[-2]
    path = parse.path if parse.path[-1] is not "/" else parse.path[:-1]
    query = parse.query
    if len(query) > 0 and query[-1] is "/":
        query = query[:-1]


    path = "_".join(path.split("/")[-2:])
    if "." in path:
        path = "".join(path.split(".")[:-1])
    # elif "." in path and "?" in path:
    return domain + "--" + path + (("@" + query) if query is not "" else "") + "." + extension

if __name__ == "__main__":
    print(genFnameFromURL(input("url: ")))
