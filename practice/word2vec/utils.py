import re
import gensim

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    return string.strip().lower()

# Gensim
PAT_ALPHABETIC = re.compile(r'''(((?![\d])'*\w)+[\.,-]?)''', re.UNICODE)

def tokenize(text, lowercase=False, deacc=False, encoding='utf8', errors="strict", to_lower=False, lower=False):
    """Iteratively yield tokens as unicode strings, removing accent marks and optionally lowercasing string
    if any from `lowercase`, `to_lower`, `lower` set to True.
    Parameters
    ----------
    text : str
        Input string.
    lowercase : bool, optional
        If True - lowercase input string.
    deacc : bool, optional
        If True - remove accentuation from string by :func:`~gensim.utils.deaccent`.
    encoding : str, optional
        Encoding of input string, used as parameter for :func:`~gensim.utils.to_unicode`.
    errors : str, optional
        Error handling behaviour, used as parameter for :func:`~gensim.utils.to_unicode`.
    to_lower : bool, optional
        Same as `lowercase`.
    lower : bool, optional
        Same as `lowercase`.
    Yields
    ------
    str
        Contiguous sequences of alphabetic characters (no digits!), using :func:`~gensim.utils.simple_tokenize`
    Examples
    --------
    >>> from gensim.utils import tokenize
    >>> list(tokenize('Nic nemůže letět rychlostí vyšší, než 300 tisíc kilometrů za sekundu!', deacc=True))
    [u'Nic', u'nemuze', u'letet', u'rychlosti', u'vyssi', u'nez', u'tisic', u'kilometru', u'za', u'sekundu']
    """
    lowercase = lowercase or to_lower or lower
    text = gensim.utils.to_unicode(text, encoding, errors=errors)
    if lowercase:
        text = text.lower()
    if deacc:
        text = gensim.utils.deaccent(text)
    return simple_tokenize(text)


def simple_tokenize(text):
    """Tokenize input test using :const:`gensim.utils.PAT_ALPHABETIC`.
    Parameters
    ----------
    text : str
        Input text.
    Yields
    ------
    str
        Tokens from `text`.
    """
    for match in PAT_ALPHABETIC.finditer(text):
        yield match.group()
# End Gensim
