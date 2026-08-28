import re

from libs.accent import accentizer
from config import config
from libs.sql_db import sql_db

class TextParse:
    def __init__(self, accent, single_vowel=None):
        self.accent = accent
        self.single_vowel = single_vowel
        self.cust_dict = dict(sql_db.select('cust_dict',{'word':False,'transcription':False}))

    def preprocess(self, string):
        string = re.sub(r'°', 'градус', string)
        string = re.sub( '№|#', 'номер ', string)
        # "+" is a stress mark and must stay silent; only a real math
        # sign between digits is spoken as a word.
        string = re.sub(r'(?<=\d)\s*\+\s*(?=\d)', ' плюс ', string)
        string = self.profanity(string)
        string = self.len_check(string)
        string = self.replace_hrname(string)
        string = self.garbage(string)
        if self.accent:
            string = accentizer.process_accent(string,r'\+\w+|\w+\+\w+')
        string = re.sub(r'(\w)\s-\s(\w)', r'\1-\2', string)
        if self.single_vowel:
            string = self.rm_pl_single_vowel(string)
        string = string.strip()
        return string

    def len_check(self, string):
        out = []
        for word in string.split():
            out.append(word[:35])

        return ' '.join(out)

    def profanity(self, string):
        string = re.sub( r'б\*\*\*', 'бляди', string)
        string = re.sub( r'б\*\*', 'бля', string)
        string = re.sub( r'с\*\*\*', 'с+ука', string)
        string = re.sub( r'по\*\*\*', 'п+охуй', string)
        string = re.sub( r'п\*\*\*\*', 'пизд+ец', string)
        string = re.sub( r'е\*\*\*', 'ебёт', string)
        string = re.sub( r'[Тт]\. е\.', 'то есть', string)
        string = re.sub( r'[Тт]\. д\.', 'так д+алее', string)

        return string

    def garbage(self, string):
        string = re.sub(r'(\’|\'|«|»|”|„|\[|\*|\u201F|\u201C|\u2800|\(с\))', '', string)
        string = re.sub(r'^[\s\.\?!:;—-…]+(?=\s*\S)', '', string)
        string = re.sub(r'(?<=\w)[—–-](?=\w)', ' ', string)
        string = re.sub(r'(?<=\S)\s*[—–-][\s\n]*(?=[\s\n]*[A-ZА-Я])', '.', string)
        string = re.sub(r'(?<=\S)\s*[—–-][\s\n]*$', '. ', string)
        string = re.sub(r'\s*[—–-]\s*', ' ', string)
        string = re.sub(r'(?<=\S)\s*[—–-]\s*(?=\w+)', ', ', string)
        if config.punctuation:
            string = re.sub(r'(!\s|:\s|;\s|\?\s|…|\.{3})', '. ', string)
            string = re.sub(r'!|:|;|\?|\xa0', ',', string)
        else:
            string = re.sub(r'(:\s|;\s|…|\.{3})', '. ', string)
            string = re.sub(r':|;|\xa0', ',', string)
        string = re.sub(r'(\)|\()|\]', '. ', string)
        string = re.sub(r',+', ', ', string)
        string = re.sub(r'(,\.|\.\.|\.,)', '.', string)
        string = re.sub(r'(\d+)(%)', r'\1 \2', string)
        string = re.sub(r'(\-)(\w{4,})', r' \2', string)

        return string

    def replace_hrname(self, string):
        string_com = re.compile('|'.join(self.cust_dict.keys()))
        def r_name(found):
            return self.cust_dict[found.group(0)]

        string = string_com.sub(r_name, string)
        return string

    def rm_pl_single_vowel(self, text):
        def process_word(match):
            word = match.group()
            
            # Check whether the word has a plus before a vowel
            if '+' in word:
                # Find all vowels in the word
                vowels = re.findall(r'[аеёиоуыэюя]', word, flags=re.IGNORECASE)
                
                # If there is only one vowel AND a plus before the vowel
                if len(vowels) == 1:
                    # Remove all pluses from the word
                    return word.replace('+', '')
            
            return word
        
        # Find all words (including ones containing +)
        return re.sub(r'[\+а-яё]+\b', process_word, text, flags=re.IGNORECASE)