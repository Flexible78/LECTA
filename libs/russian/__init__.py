from config import config
from pymorphy3 import MorphAnalyzer
morph = MorphAnalyzer()

from .number_normalizer import normalize_number_with_text, normalize_number_without_text, replace_roman
from .text_processor import replace_abbreviations, cyrillize, add_comma_before_latin

def normalize_russian(text):
    text = replace_roman(text)
    text = replace_abbreviations(text)
    text = normalize_number_with_text(text)
    text = normalize_number_without_text(text)
    if config.translit:
        text = cyrillize(text)
    else:
        text = add_comma_before_latin(text)
    return text

# Simplified interface
normalize = normalize_russian