from pathlib import Path
import json
from . import morph

# Загрузка словаря
json_path = Path(__file__).parent / 'num_dict.json'
num_dict = json.loads(json_path.read_text(encoding='utf-8'))

mrf = num_dict['mrf']
t_units = num_dict['t_units']
hundreds = num_dict['hundreds']

def num_to_words(attr1, n, attr2, adv_attr=None):
    # Определение частей речи, падежа, рода и числа для согласования числительного
    if adv_attr:
        pos0 = adv_attr['POS']
        pos = adv_attr['POS']
        case = adv_attr['case']
        gender = adv_attr['gender']
        ch_num = adv_attr['number']
    else:
        pos = attr2.tag.POS
        pos0 = attr1[0].tag.POS
        case = attr2.tag.case
        gender = attr2.tag.gender
        ch_num = attr2.tag.number
    
        if pos == 'NOUN':
            if case == 'accs' and gender == 'masc' and ch_num == 'sing' and \
                (pos0 == 'VERB' or pos0 == 'CONJ' or pos0 == 'PREP'):
                case = 'nomn'
            if case == 'gent' and gender == 'masc' and ch_num == 'sing' and pos0 == 'VERB':
                ch_num = 'plur'
            if case == 'nomn' and gender == 'masc' and ch_num == 'sing' and \
                (pos0 == 'ADVB' or pos0 == 'PREP'):
                ch_num = 'plur'
                gender = 'femn'
            if (case == 'loc2' or case == 'datv'):
                case = 'loct'

        elif pos == 'ADJF' or pos == 'NPRO':
            if case == 'loct' or (case == 'datv' and pos0 == 'PREP'):
                case = 'gent'
            if case == 'nomn':
                case = 'gent'
                gender ='femn'
                ch_num = 'plur'
            if (case == 'ablt' or case == 'datv') and pos0 == 'NOUN':
                case = attr1[0].tag.case
                gender = attr1[0].tag.gender
                ch_num = attr1[0].tag.number
        elif pos == 'PRTF':
                gender = 'femn'
        elif not pos:
            if pos0 == 'ADVB':
                case = 'gent'
                gender = 'masc'
                ch_num = 'plur'
            elif pos0 == 'CONJ' or pos0 == 'PREP':
                case = 'accs'
                gender = 'masc'
                ch_num = 'sing'
            elif pos0 == 'ADJF':
                case = 'loct'
                if not gender: gender = 'femn'
                if not ch_num: ch_num = 'plur'
            elif pos0 == 'NPRO':
                case = 'accs'
            elif attr1[0].tag.case:
                case = attr1[0].tag.case
                if case == 'ablt': case = 'accs'
                gender = attr1[0].tag.gender
                ch_num = attr1[0].tag.number
            else:
                case = 'accs'
                gender = 'masc'
                ch_num = 'sing'

    if not case: case = 'accs'
    if not ch_num: ch_num = 'sing'
    if not gender: gender = 'femn'

    if n == 0:
        return 'ноль'

    # --- ПРАВКА: Защита от гигантских чисел (Телефоны, Банковские счета) ---
    if n >= 1_000_000_000_000:
        digit_words = ['ноль', 'один', 'два', 'три', 'четыре', 'пять', 'шесть', 'семь', 'восемь', 'девять']
        return ' '.join(digit_words[int(d)] for d in str(n))
    # -----------------------------------------------------------------------

    # Числа от 10 до 19
    teens = ['десят','одиннадцат','двенадцат','тринадцат','четырнадцат','пятнадцат','шестнадцат','семнадцат','восемнадцат','девятнадцат']
    # Десятки
    tens = ['','десят','двадцат','тридцат','сороков','пятьдесят','шестьдесят','семьдесят','восемьдесят','девяност']    
    # Единицы измерения для миллионов, миллиардов, тысяч
    million_units = morph.parse('миллион')[0]
    billion_units = morph.parse('миллиард')[0]
    thousand_units = morph.parse('тысяча')[0]

    words = []

    # Вспомогательная функция для обработки чисел до 1000
    def under_thousand(number):
        if number == 0:
            return []
        elif number < 10:
           return [t_units[case][gender][ch_num][number]]
        elif number < 20:
            if case == 'accs':
                return [teens[number - 10] + mrf['nomn']['masc']['teens']['plur']]
            else:
                return [teens[number - 10] + mrf[case][gender]['teens'][ch_num]]
        elif number < 100 and number % 10 != 0:
            if number % 100 // 10 * 10 == 40:
                return ['сорок', t_units[case][gender][ch_num][number % 10]]
            else:
                if number // 10 == 2:
                    cif = 'двадцать'
                    if case == 'ablt':
                        cif = 'двадцати'
                    return [cif, t_units[case][gender][ch_num][number % 10]]
                if number // 10 == 3:
                    cif = 'тридцать'
                    if case == 'ablt':
                        cif = 'тридцати'
                    return [cif, t_units[case][gender][ch_num][number % 10]]
                if number // 10 == 9:
                    return ['девяносто', t_units[case][gender][ch_num][number % 10]]
                elif case == 'loct' or case == 'loc2' or (case == 'gent' and ch_num != 'plur'):
                    return [tens[number // 10] + mrf['gent']['femn']['tens']['sing'], t_units[case][gender][ch_num][number % 10]]
                elif case == 'accs':
                    return [tens[number // 10] + mrf['gent']['masc']['tens']['plur'], t_units[case][gender][ch_num][number % 10]]
                elif case == 'nomn':
                    return [tens[number // 10] + mrf['gent']['neut']['tens']['sing'], t_units[case][gender][ch_num][number % 10]]
                else:
                    return [tens[number // 10] + mrf[case][gender]['tens'][ch_num], t_units[case][gender][ch_num][number % 10]]
        elif number < 100:
            if number // 10 == 2 and ch_num == 'plur':
                return ['двадцать']
            if number // 10 == 3 and ch_num == 'plur':
                    return ['тридцать']
            if number // 10 == 4:
                if pos0 == 'ADVB' and case == 'gent':
                    return ['сорока']
                elif pos0 == 'NOUN':
                    return ['с+орок']
                else:
                    return ['сороков' + mrf[case][gender]['tens'][ch_num]]
            if number // 10 == 9:
                return ['девяносто']
            elif (case == 'loct' or case == 'loc2' and ch_num == 'sing' and gender != 'femn') \
                or (case == 'gent' and ch_num == 'plur'):
                return [tens[number // 10] + mrf['gent']['femn']['tens']['sing']]
            else:
                return [tens[number // 10] + mrf[case][gender]['tens'][ch_num]]
        else:
            if case == 'accs' or case == 'nomn':
                return [hundreds['accs'][number // 100]] + under_thousand(number % 100)
            else:
                return [hundreds['gent'][number // 100]] + under_thousand(number % 100)

    # Разбиение числа на миллиарды, миллионы, тысячи и остаток
    billions = n // 1_000_000_000
    millions = (n % 1_000_000_000) // 1_000_000
    thousands = (n % 1_000_000) // 1_000
    remainder = n % 1_000

    last = under_thousand(remainder)
    if billions:
        words += under_thousand(billions) + [billion_units.make_agree_with_number(billions).word]
    if millions:
        words += under_thousand(millions) + [million_units.make_agree_with_number(millions).word]
    if thousands:
        if thousands <= 9 and remainder == 0:
            words.append(t_units['loct']['masc']['plur'][thousands])
            words.append('тысячн' + mrf[case][gender]['teens'][ch_num])
        else:
            # Особые формы для "одна", "две" тысячи
            if thousands % 10 == 1 and thousands % 100 != 11:
                words.append('одна')
            elif thousands % 10 == 2 and thousands % 100 != 12:
                words.append('две')
            else:
                ch_num = 'plur'
                words += under_thousand(thousands)
            words.append(thousand_units.make_agree_with_number(thousands).word)
    words += last

    last_num = ' '.join(word for word in words if word)
    last_num = last_num.split()[-1]

    return ' '.join(word for word in words if word)