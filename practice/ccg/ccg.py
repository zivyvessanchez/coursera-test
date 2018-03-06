# Goals:
# - Parse entire CCG sentences.
# - Induce CCG tagging on raw words.

import sys
import nltk

from copy import deepcopy

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
    'NN': set(['N']),
    'NNS': set(['N']),
    'NNP': set(['N']),
    'PRP': set(['N']),
    'DT': set(['N']),
    'MD': set(['S']),
    'VB': set(['S']),
    'VBZ': set(['S']),
    'VBG': set(['S']),
    'VBN': set(['S']),
    'VBD': set(['S']),
    'CC': set(['conj'])
}
# CCG directional markers
CCG_MARKERS = ['/','\\','|']

def sanitize(string=''):
    s = string
    s = s.replace(' ', '')
    s = s.upper()
    return s

def analyze(string=''):
    level = 0
    arity = 0
    level_strings = {0: ''}
    string = '(' + string + ')'
    #print(string)
    for s in string:
        if s == '(':    # Up a level
            level += 1
            arity = max(level, arity)
            # Pre-init string if it does not exist
            try:
                level_strings[level]
            except:
                level_strings[level] = ''
            for i in reversed(range(level+1)):
                level_strings[i] += s
        elif s == ')':  # Down a level
            for i in reversed(range(level+1)):
                level_strings[i] += s
            #print(level, level_strings[level])
            level_strings[level] = ''
            level -= 1
        else:
            try:
                level_strings[level]
            except:
                level_strings[level] = ''
            for i in reversed(range(level+1)):
                level_strings[i] += s
    return arity

def __induce(string=''):    
    # Initialize lexicion with seed knowledge
    lexicon = CCG_ENG
    cur_lexicon = {}
    sentences_word = []
    sentences_pos = []
    n = 2

    # Tokenize string into array of pos-tagged sentences
    string_split = string.split('.')
    for s in string_split:
        # Add period back to sentence, but remove it again
        # in the final pos-tag output
        if len(s) > 0:
            s += '.'
            
        sentence = nltk.pos_tag(nltk.word_tokenize(s))
        sentence = sentence[:-1]
        if len(sentence) > 0:
            for word_pos_pair in sentence:
                sentences_word.append(word_pos_pair[0])
                sentences_pos.append(word_pos_pair[1])
    print('sentence_word is {0}'.format(sentences_word))
    print('sentence_pos is {0}'.format(sentences_pos))
    
    # Perform induction loop n times
    for _ in range(n):
        # s is a single sentence's pos tags
        for i, s in enumerate(sentences_pos):
            # Debug print
            if cur_lexicon != lexicon:
                cur_lexicon = deepcopy(lexicon)
                print('cur lexicon is {0}'.format(cur_lexicon))
                
            #print('s is {0}, {1}'.format(sentences_word[i], s))
            # First word in sentence
            if i == 0:
                #print('First word!')
                __induceRight(
                    sentences_pos[i],
                    sentences_pos[i+1],
                    lexicon
                )
            # Last word in sentence
            elif i == (len(sentences_pos)-1):
                #print('Last word!')
                __induceLeft(
                    sentences_pos[i],
                    sentences_pos[i-1],
                    lexicon
                )
            # Any other word inside sentence
            elif sentences_pos[i] in CCG_ENG and \
                 CCG_ENG[sentences_pos[i]] != 'conj':
                #print('Conj check!')
                __induceLeft(
                    sentences_pos[i],
                    sentences_pos[i-1],
                    lexicon
                )
                __induceRight(
                    sentences_pos[i],
                    sentences_pos[i+1],
                    lexicon
                )
        # Update lexicon with new categories
        #lexicon.update(new_lexicon)

def __induceRight(left_pos, right_pos, lexicon):
    g = __generateCcgToken
    left = deepcopy(lexicon.setdefault(left_pos, set([])))
    right = deepcopy(lexicon.setdefault(right_pos, set([])))
    for r in right:
        for l in left:
            # Can left tag take right tag as argument?
            if __induceIsValid(g(l), g(r), '/'):
                lr = g(l) + '/' + g(r)
                lexicon.setdefault(left_pos, set([])).add(lr)

        # Can right tag be modified?
        if __induceIsValid(g(r), g(r), '/'):
            rr = g(r) + '/' + g(r)
            lexicon.setdefault(left_pos, set([])).add(rr)

def __induceLeft(left_pos, right_pos, lexicon):
    g = __generateCcgToken
    left = deepcopy(lexicon.setdefault(left_pos, set([])))
    right = deepcopy(lexicon.setdefault(right_pos, set([])))
    for l in left:
        for r in right:
            # Can right tag take left tag as argument?
            if __induceIsValid(g(r), g(l), '\\'):
                rl = g(r) + '\\' + g(l)
                lexicon.setdefault(right_pos, set([])).add(rl)

        # Can left tag be modified?
        if __induceIsValid(g(l), g(l), '\\'):
            ll = g(l) + '\\' + g(l)
            lexicon.setdefault(right_pos, set([])).add(ll)

def __induceIsValid(left_pos, right_pos, marker):
    # Constraint 1: Nouns (N) do not take any arguments.
    if left_pos == 'N' and right_pos != 'N':
        return False
    
    # Constraint 2: The heads of sentences (S|...) and
    # modifiers (X|X, (X|X)|(X|X)) may take N or S as
    # arguments.

    # Constraint 3: Sentences (S) may only take nouns (N)
    # as arguments. We assume S\S and S/S are modifiers.

    # Constraint 4: The maximal arity of any lexical
    # category is 3.
    ccgToken = ''
    if marker in ['/','|']:
        ccgToken = left_pos + marker + right_pos
    else:
        ccgToken = right_pos + marker + left_pos

    if __isModifier(ccgToken) and analyze(ccgToken) > 2:
        return False;
    elif not __isModifier(ccgToken) and analyze(ccgToken) > 3:
        return False;

    # Constraint 5: Since (S\N)/N is completely equivalent
    # to (S/N)\N, we only allow the former category.
    if left_pos == '(S/N)' and right_pos == 'N' and marker == '\\':
        return False

    # Constraint 6: Coordinating Conjunctions are restricted
    # to conj if not sentence initial or final.
    if left_pos == 'conj' or right_pos == 'conj':
        return False

    # Constraint 7: Disallow (X/X)\X to reduce ambiguity.
    if left_pos == '(S/S)' and right_pos == 'S' and marker == '\\':
        return False
    if left_pos == '(N/N)' and right_pos == 'N' and marker == '\\':
        return False

    return True

def __isAtomic(pos):
    for s in CCG_MARKERS:
        if s in pos:
            return False
    return True

def __isModifier(pos):
    s = pos
    # Strip away non-alpha symbols
    for m in ['(',')','/','|','\\']:
        s = s.replace(m,'')
    # Get number of unique pos tags.
    # Return false if more than 1 pos tag
    tags = set([])
    for c in s:
        tags.add(c)
    if len(tags) == 1:
        return True
    return False

def __generateCcgToken(pos):
    if __isAtomic(pos):
        return pos
    else:
        return '('+pos+')'

def main():
    print(isValid(sys.argv[1]))

if __name__ == '__main__':
    main()
