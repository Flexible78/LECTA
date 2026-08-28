import re
from .num_to_words import num_to_words
from .utils import month_names
from . import morph
from pathlib import Path
import json

json_path = Path(__file__).parent / 'num_dict.json'
num_dict = json.loads(json_path.read_text(encoding='utf-8'))

units_abbr = num_dict['units_abbr']
buildup = num_dict['buildup']

def expand_abbr(match, cnum, data_attr, pre_attr):
    if 'Abbr' in data_attr.tag and match in units_abbr:
        bks = morph.parse(units_abbr[match])[0]
        return bks.make_agree_with_number(cnum % 10).word
    return match

def normalize_number_with_text(text):
    num_pattern = re.compile(r'''
            (\b\w+\b\s+)?          
            (\d+[.,-]\d+|\d{2}\.\d{2}\.\d{4}|\d+\S\d+|\d+)
            ([!-/:-@[-`{-~]\w{1,2}|[!-/:-@[-`{-~]\w{1,2}\s|\s|\b|\D)
            (\w{3,}|\W|$)          
        ''', re.IGNORECASE | re.VERBOSE)
        
    def normalize_num(match):
        try:
            result = [match.groups()]
            if result:
                for stor in result:
                    stor = list(stor)
                    pre_attr = morph.parse('')  
                    data_attr = morph.parse(stor[3])[0]  
                    dattr = {}
                    
                    if stor[3] in units_abbr:
                        bks = morph.parse(units_abbr[stor[3]])[0]
                        stor[3] = bks.make_agree_with_number(int(stor[1][-1])).word
                        
                    if stor[0] is not None:
                        pre_attr = morph.parse(stor[0])  
                        if stor[0].lower() == 'к' and data_attr.tag.POS != 'NOUN':
                            dattr = {'POS': 'NOUN', 'case': 'loc2', 'gender': 'masc', 'number': 'sing'}
                    else:
                        stor[0] = ''
                        
                    dmy = re.match(r'(\d{2})\.(\d{2})\.(\d{4})', stor[1])
                    nn = re.findall(r'(\d+)(\D+)(\d+)', stor[1])  
                    
                    if dmy:
                        d_month = month_names[dmy.groups(0)[1]]
                        bks = morph.parse(d_month)[0]
                        d_year = num_to_words(pre_attr, int(dmy.groups(0)[2]), bks)
                        last_word = d_month + ' ' + d_year + stor[3]
                        return stor[0] + ' ' + num_to_words(pre_attr, int(dmy.groups(0)[0]), bks) + ' ' + last_word
                        
                    if nn:
                        inter = ', '
                        dattr = {'POS': 'NOUN', 'case': 'loct', 'gender': 'femn', 'number': 'plur'}
                        if nn[0][1] == ',' or nn[0][1] == '.' or nn[0][1] == '-':
                            inter = ' и '
                            dattr = {'POS': 'NOUN', 'case': 'accs', 'gender': 'masc', 'number': 'sing'}
                        first_num = num_to_words(pre_attr, int(nn[0][0]), data_attr, dattr)
                        last_word = expand_abbr(stor[3], int(nn[0][2]), data_attr, pre_attr)
                        data_attr = morph.parse(last_word)[0]
                        second_num = num_to_words(pre_attr, int(nn[0][2]), data_attr, dattr)
                        return stor[0] + ' ' + first_num + inter + second_num + ' ' + last_word
                    
                    # === THE SAME FIX FOR THE LETTER "l" (lang) ===
                    is_buildup = stor[2] in buildup
                    if is_buildup:
                        dattr = buildup[stor[2]]
                        if pre_attr[0].tag.POS == 'PREP' and buildup[stor[2]]['case'] == 'ablt':
                            dattr['case'] = 'loct'
                    
                    last_word = expand_abbr(stor[3], int(stor[1]), data_attr, pre_attr)
                    
                    if not is_buildup:
                        if len(stor[2]) > 3:
                            last_word = re.sub(r'[!-/:-`{-~]', '', stor[2]) + last_word
                        else:
                            last_word = stor[2] + last_word
                            
                    return stor[0] + ' ' + num_to_words(pre_attr, int(stor[1]), data_attr, dattr) + ' ' + last_word
                    # ============================================
        except Exception as e:
            return match.group(0)

    normalized_text = num_pattern.sub(normalize_num, text)
    return normalized_text

def normalize_number_without_text(text):
    def replace_num(match):
        num = int(match.group(0))
        attrs = {'POS': 'NOUN', 'case': 'accs', 'gender': 'femn', 'number': 'sing'}
        return num_to_words(None, num, None, attrs)

    return re.sub(r'\b\d+\b', replace_num, text)

def replace_roman(text):
    if not isinstance(text, str):
        raise ValueError("Input must be a string")

    pattern = re.compile(r'(?<!\w)([IVXLCDM]{1,})(?=[\s,.?!/)\'\]>\.-]|$)')

    def convert_match(match):
        roman = match.group(1)
        start_idx = match.start()
        end_idx = match.end()
        
        # --- ENGLISH GUARD ---
        if roman == 'I' and end_idx < len(text) and text[end_idx] == "'":
            return roman
            
        if roman == 'I' and end_idx < len(text):
            next_chars = text[end_idx:end_idx+5].lower()
            if next_chars.startswith((' am', ' wa', ' ha', ' do', ' wi', ' wo', ' ca', ' sh', ' li', ' lo')):
                return roman

        if re.search(r'[a-z]', text):
            prefix = text[max(0, start_idx-15):start_idx].lower()
            markers = ['глава', 'часть', 'том', 'век', 'акт', 'раздел', 'пункт', 'николай', 'петр', 'александр', 'людовик', 'карл', 'екатерина']
            if not any(m in prefix for m in markers):
                return roman
        # --- End of the guard ---
        
        num = roman_to_int(roman)
        return str(num)

    return pattern.sub(convert_match, text)

def roman_to_int(s):
    rom_val = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    int_val = 0
    for i in range(len(s)):
        if i > 0 and rom_val[s[i]] > rom_val[s[i - 1]]:
            int_val += rom_val[s[i]] - 2 * rom_val[s[i - 1]]
        else:
            int_val += rom_val[s[i]]
    return int_val