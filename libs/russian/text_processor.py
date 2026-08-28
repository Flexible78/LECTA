import re
from typing import List, Set, Union, Pattern
from .utils import cyrillization_mapping, pronunciation_map
from libs.sql_db import sql_db

db_result = sql_db.select('exc_abrs', {'abbreviation': False})
exc_abrs_list: List[str] = []
for row in db_result:
    if isinstance(row, list):
        exc_abrs_list.extend(row)
    else:
        exc_abrs_list.append(row)

exc_abrs: Set[str] = set(exc_abrs_list)

ABBR_PATTERN: Pattern = re.compile(r'\b[А-Я]{2,5}[.,]?\b')
EXC_ABR_PATTERN: Pattern = re.compile(rf'\b(?:{"|".join(map(re.escape, exc_abrs))})\b')

LATIN_SEQUENCE_PATTERN: Pattern = re.compile(r'\b[a-zA-Z]+(?:[\s\'’`´ʼ‘]+[a-zA-Z]+)*\b')
CYRILLIC_PATTERN: Pattern = re.compile(r'[а-яё]', re.IGNORECASE)

# === BULLETPROOF ENGLISH DECODING ===
def expand_english_contractions(text: str) -> str:
    # Normalize any slanted/curly apostrophes to a standard straight one
    text = re.sub(r"[’`´ʼ‘]", "'", text)
    
    # Direct replacement (100% reliable, no regex pitfalls)
    reps = {
        "I'm": "I am", "i'm": "I am",
        "I'll": "I will", "i'll": "I will",
        "I'd": "I would", "i'd": "I would",
        "I've": "I have", "i've": "I have",
        "You're": "You are", "you're": "you are",
        "You'll": "You will", "you'll": "you will",
        "He's": "He is", "he's": "he is",
        "She's": "She is", "she's": "she is",
        "It's": "It is", "it's": "it is",
        "We're": "We are", "we're": "we are",
        "We'll": "We will", "we'll": "we will",
        "They're": "They are", "they're": "they are",
        "That's": "That is", "that's": "that is",
        "Don't": "Do not", "don't": "do not",
        "Doesn't": "Does not", "doesn't": "does not",
        "Didn't": "Did not", "didn't": "did not",
        "Can't": "Cannot", "can't": "cannot",
        "Won't": "Will not", "won't": "will not",
        "Isn't": "Is not", "isn't": "is not",
        "Aren't": "Are not", "aren't": "are not",
        "Wasn't": "Was not", "wasn't": "was not",
        "Weren't": "Were not", "weren't": "were not"
    }
    for k, v in reps.items():
        text = text.replace(k, v)
    return text

def cyrillize(text: str) -> str:
    text = expand_english_contractions(text)
    words: List[str] = text.split()
    
    def process_word(word: str) -> str:
        if CYRILLIC_PATTERN.search(word):
            return word
        
        word = word.lower()
        result: List[str] = []
        i: int = 0
        
        while i < len(word):
            if i + 1 < len(word) and word[i:i+2] in cyrillization_mapping:
                result.append(cyrillization_mapping[word[i:i+2]])
                i += 2
            else:
                result.append(cyrillization_mapping.get(word[i], word[i]))
                i += 1
        
        return ''.join(result)
    
    return ' '.join(process_word(word) for word in words)

def add_comma_before_latin(text: str) -> str:
    # 1. First, reliably decode the English
    text = expand_english_contractions(text)
    
    def add_comma(match: re.Match) -> str:
        return ',' + match.group()
    
    # 2. Then put a comma before the Latin block
    return LATIN_SEQUENCE_PATTERN.sub(add_comma, text)

def replace_abbreviations(string: str) -> str:
    string = re.sub(r'http[s]?://\S+', '', string)
    if not ABBR_PATTERN.search(string):
        return string
    words: List[str] = string.split()
    processed: List[str] = []
    
    for word in words:
        if ABBR_PATTERN.match(word):
            if word in exc_abrs:
                processed.append(word.lower())
            else:
                processed.append(''.join(
                    pronunciation_map.get(char, char) for char in word
                ))
        else:
            processed.append(word)
    
    return ' '.join(processed)