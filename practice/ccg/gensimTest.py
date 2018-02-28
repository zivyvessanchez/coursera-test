import nltk
import gensim
import utils

def tokenize(content):
    return [
        gensim.utils.to_unicode(token)
        for token in utils.tokenize(content, lower=False, errors='ignore')
        if 15 >= len(token) >= 1 and not token.startswith('_')
    ]

def main():
    with open('wiki.en.text','r',encoding='utf-8') as o:
        with open('gensimOut.txt','w',encoding='utf-8') as i:
            line = nltk.word_tokenize(o.readline())
            i.write(str(nltk.pos_tag(line)))

if __name__ == '__main__':
    main()
