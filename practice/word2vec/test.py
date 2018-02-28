import nltk

def main():
    with open('wiki.en.text', 'r', encoding='utf-8') as o:
        with open('out.txt', 'w', encoding='utf-8') as i:
            s = nltk.pos_tag(nltk.word_tokenize(o.readline()))
            i.write(str(s))

if __name__ == '__main__':
    main()
