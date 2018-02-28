import nltk
import gensim
import scraper

def tokenize(content):
    return [
        gensim.utils.to_unicode(token)
        for token in gensim.utils.tokenize(content, lower=False, errors='ignore')
        if 15 >= len(token) >= 1 and not token.startswith('_')
    ]

def main():
    scraper.scrape('Anarchism')
    with open('scraperOut.txt','r',encoding='utf-8') as o:
        with open('gensimOut.txt','w',encoding='utf-8') as i:
            line = tokenize(o.read())
            i.write(str(nltk.pos_tag(line)))

if __name__ == '__main__':
    main()
