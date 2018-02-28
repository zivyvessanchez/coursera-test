# Goals:
# - Parse entire CCG sentences.
# - Induce CCG tagging on raw words.

import sys

ALLOWED_TOKENS = ['S','N','NP','ADJ','PP','/','\\','(',')']

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

def ccgOpApplication(a,b,direction):
    out = ''
    
    return out

def main():
    print(isValid(sys.argv[1]))

if __name__ == '__main__':
    main()
