# _*_ coding:utf-8 _*_

import os
import sys
import requests
import codecs
import re
from bs4 import BeautifulSoup

wikiUrl = 'https://en.wikipedia.org'
outFileName = 'scraperOut.txt'

def scrape(articleName, fileName = ''):
    global outFileName
    
    # Scrape url
    urlToScrape = wikiUrl + "/wiki/" + articleName
    response = requests.get(urlToScrape)
    html = response.content

    # Convert html bytes to string, truncate at before "See Also" section
    s = str(html, 'utf-8')
    s = s[0:s.rfind("See_also")]

    soup = BeautifulSoup(s, 'html.parser')

    # Remove table of contents and other unnecessary entries
    # Remove disambiguation notice
    for s in soup.findAll('div', attrs={'class': 'hatnote'}):
        #print(s.text)
        s.decompose();

    # Remove content group listing tables
    for s in soup.findAll('table', attrs={'class': 'infobox'}):
        #print(s.text)
        s.decompose();

    # Remove table of contents
    for s in soup.findAll('div', attrs={'class': 'toc'}):
        #print(s.text)
        s.decompose();

    # Remove edit tags
    for s in soup.findAll('span', attrs={'class': 'mw-editsection'}):
        #print(s.text)
        s.decompose();

    # Remove citation-needed tags
    for s in soup.findAll('sup', attrs={'class': 'Inline-Template'}):
        #print(s.text)
        if re.search('(citation needed)', s.text) is not None:
            s.decompose();

    # Remove reference tags
    for s in soup.findAll('sup', attrs={'class': 'reference'}):
        #print(s.text)
        s.decompose();

    # Remove warning templates
    for s in soup.findAll('table', attrs={'class': 'plainlinks'}):
        #print(s.text)
        s.decompose();

    # Remove image thumbnails
    for s in soup.findAll('div', attrs={'class': 'thumb'}):
        #print(s.text)
        s.decompose();

    # Remove gallery thumbnails
    for s in soup.findAll('ul', attrs={'class': 'gallery'}):
        #print(s.text)
        s.decompose();

    # Retrieve main content section
    title = soup.find('h1', attrs={'class': 'firstHeading'})
    content = soup.find('div', attrs={'class': 'mw-content-ltr'})
    contentText = content.get_text()
    #print(content)
    #print(type(content))
    print('Parse complete: ' + urlToScrape)
    
    if(fileName is not ''):
        outFileName = fileName
    with open(outFileName, 'w+', encoding='utf-8') as outfile:
        outfile.write(title.text + '\n\n' + str(contentText))
        outfile.close()

def main():
    if (len(sys.argv) < 2):
        print("Usage: {0} <wikipedia article title> [<output filename>]".format(
            os.path.basename(__file__)))
        return

    print("Scraping " + sys.argv[1])
    if(len(sys.argv) >= 3):
        scrape(sys.argv[1], sys.argv[2])
    else:
        scrape(sys.argv[1])

if __name__ == '__main__':
    main()
