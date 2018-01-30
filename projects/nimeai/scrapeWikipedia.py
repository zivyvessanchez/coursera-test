# _*_ coding:utf-8 _*_

def startScrapeWikipedia():
    import os
    import requests
    import codecs
    import re
    from bs4 import BeautifulSoup

    wikiUrl = 'https://en.wikipedia.org'
    url = wikiUrl + '/w/index.php?title=Special:AllPages'
    fileIdx = 0
    filenamePrefix = './wikiPages'
    filenameSuffix = '.txt'
    filename = filenamePrefix + str(fileIdx) + filenameSuffix
    stop = False
    idx = 0
    maxIdx = 25000 # max number of entries per file
    totalTitles = 0
    totalUrls = 0

    # delete wikiPages.
    #if os.path.exists(filename):
    #    os.remove(filename)
    
    #for count in range(0,10):
    while not stop:
        listOfTitles = []
        listOfUrls = []
    
        # Get soup of url
        print("Titles: " + str(totalTitles) +
              " - URLs: " + str(totalUrls) + " - Parsing " + url)
        response = requests.get(url)
        html = response.content

        soup = BeautifulSoup(html, 'html.parser')

        # Get the table of contents
        table = soup.find('ul', attrs={'class': 'mw-allpages-chunk'})

        # For each entry in the table of contents, get name and url link
        for row in table.findAll('li'):
            a = row.find('a', href=True)
            listOfTitles.append(str(row.text))
            listOfUrls.append(str(a['href']))

        # Save list of urls to file
        outfile = codecs.open(filename, 'a', 'utf-8')
        for i, value in enumerate(listOfTitles, start=0):
            outfile.write(listOfUrls[i] + "\n")
            idx += 1
            # If num of entries exceed maxIdx, place remaining data to next file
            if idx > maxIdx:
                idx = 0
                fileIdx += 1
                filename = filenamePrefix + str(fileIdx) + filenameSuffix
                outfile.close()
                outfile = codecs.open(filename, 'a', 'utf-8')
        outfile.close()

        # Update count of total titles and urls
        totalTitles += len(listOfTitles)
        totalUrls += len(listOfUrls)

        # Go to "Next Page" url to retrieve more urls
        url = None
        table = soup.find('div', attrs={'class': 'mw-allpages-nav'})
        for row in table.findAll('a', href=True):
            nextPage = re.search('(Next page)', row.text)
            if nextPage is not None:
                url = wikiUrl + row['href']
                break

        # If "Next Page" url is not found, stop scrape
        if url is None:
            print("No URL found! Stopping the scrape!")
            stop = True
            break

        # increment indices
        idx += 1
        if idx >= maxIdx:
            idx = 0
            fileIdx += 1
            filename = filenamePrefix + str(fileIdx) + filenameSuffix
