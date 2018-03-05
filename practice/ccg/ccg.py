# Goals:
# - Parse entire CCG sentences.
# - Induce CCG tagging on raw words.

import sys
import nltk

ALLOWED_TOKENS = ['S','N','NP','ADJ','PP','/','\\','(',')']
# Penn POS Tags
SEED_ENG = ['CC','CD','DT','EX','FW',
                 'IN','JJ','JJR','JJS','LS',
                 'MD','NN','NNS','NNP','NNPS',
                 'PDT','POS','PRP','PRP$','RB',
                 'RBR','RBS','RP','SYM','TO',
                 'UH','VB','VBD','VBG','VBN',
                 'VBP','VBZ','WDT','WP','WP$',
                 'WRB',',',';']
# Penn POS -> CCG Tag dictionary
CCG_ENG = {
    'NN': ['N'],
    'NNS': ['N'],
    'NNP': ['N'],
    'PRP': ['N'],
    'DT': ['N'],
    'MD': ['S'],
    'VB': ['S'],
    'VBZ': ['S'],
    'VBG': ['S'],
    'VBN': ['S'],
    'VBD': ['S'],
    'CC': ['conj']
}
# CCG directional markers
CCG_MARKERS = ['/','\\','|']

def sanitize(string=''):
    s = string
    s = s.replace(' ', '')
    s = s.upper()
    return s

def isValid(string=''):
    string = sanitize(string)
    cur = ''
    for c in string:
        cur += c
        if cur in ALLOWED_TOKENS:
            cur = ''

    if cur == '':
        return True
    else:
        return False

def __induce(string=''):    
    # Initialize lexicion with seed knowledge
    lexicon = CCG_ENG
    new_lexicon = []
    sentences_word = []
    sentences_pos = []
    n = 10

    # Tokenize string into array of pos-tagged sentences
    string_split = string.split('.')
    for s in string_split:
        sentence = nltk.pos_tag(nltk.word_tokenize(s))
        if len(sentence) > 0:
            sentences_word.append(sentence[0])
            sentences_pos.append(sentence[1])
    
    # Perform induction loop n times
    for _ in range(n):
        # s is a single sentence's pos tags
        for s in sentences_pos:
            for i in range(len(s)):
                # First word in sentence
                if i == 1:
                    __induceRight(lexicon[s[i]], lexicon[s[i+1]], new_lexicon)
                # Last word in sentence
                elif i == (len(s)-1)
                    __induceLeft(lexicon[s[i]], lexicon[s[i-1]], new_lexicon)
                # Any other word inside sentence
                else if CCG_ENG[s[i]] != 'conj':
                    __induceLeft(lexicon[s[i]], lexicon[s[i-1]], new_lexicon)
                    __induceRight(lexicon[s[i]], lexicon[s[i+1]], new_lexicon)
        # Update lexicon with new categories
        lexicon.update(new_lexicon)
    

def __induceRight(left_pos, right_pos, lexicon):
    g = __generateCcgToken
    for r in right_pos:
        for l in left_pos:
            # Can left tag take right tag as argument?
            if __induceIsValid(g(l), g(r)):
                lr = g(l) + '/' + g(r)
                lexicon[left_pos].append(lr)

        # Can right tag be modified?
        if __induceIsValid(g(r), g(r)):
            rr = g(r) + '/' + g(r)
            lexicon[left_pos].append(rr)

def __induceLeft(left_pos, right_pos, lexicon):
    g = __generateCcgToken
    for r in right_pos:
        for l in left_pos:
            # Can right tag take left tag as argument?
            if __induceIsValid(g(r), g(l)):
                rl = g(r) + '\\' + g(l)
                lexicon[right_pos].append(rl)

        # Can left tag be modified?
        if __induceIsValid(g(l), g(l)):
            ll = g(l) + '\\' + g(l)
            lexicon[right_pos].append(ll)

def __induceIsValid(left_pos, right_pos):
    # Constraint 1: Nouns (N) do not take any arguments.
    if left_pos == 'N' and right_pos != 'N':
        return False
    return True

def __isAtomic(pos):
    for s in CCG_MARKERS:
        if s in pos:
            return False
    return True

def __generateCcgToken(pos):
    if __isAtomic(pos):
        return pos
    else:
        return '('+pos+')'

def main():
    print(isValid(sys.argv[1]))

if __name__ == '__main__':
    main()
