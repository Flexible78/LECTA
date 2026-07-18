# g2p.py
from typing import List, Tuple, Dict, Optional
import re

class G2P:
    def __init__(self, dictionary: Dict[str, str], phoneme_id_map: Dict[str, int]):
        # Константы
        self.SOFTLETTERS = set("яёюиье")
        self.STARTSYL = set("#ъьаяоёуюэеиы-")
        self.OTHERS = {"#", "+", "-", "ь", "ъ"}

        self.SOFTHARD_CONS = {
            "б": "b", "в": "v", "г": "g", "Г": "g", "д": "d", "з": "z",
            "к": "k", "л": "l", "м": "m", "н": "n", "п": "p", "р": "r",
            "с": "s", "т": "t", "ф": "f", "х": "h"
        }

        self.OTHER_CONS = {
            "ж": "zh", "ц": "c", "ч": "ch", "ш": "sh", "щ": "sch", "й": "j"
        }

        self.VOWELS = {
            "а": "a", "я": "a", "у": "u", "ю": "u", "о": "o", "ё": "o",
            "э": "e", "е": "e", "и": "i", "ы": "y"
        }

        self.dictionary = dictionary  # слово -> фонемы (строка)
        self.phoneme_id_map = phoneme_id_map

    def add_pos(self, x):
        if len(x) == 1:
            return [x[0] + "_S"]

        res = []
        for i, p in enumerate(x):
            if i == 0:
                res.append(p + "_B")
            elif i == len(x) - 1:
                res.append(p + "_E")
            else:
                res.append(p + "_I")
        return res

    def g2p_multistream(self, text, bert_embeddings, word_pos=False):
        phonemes = [("^", [], 0, 0)]

        pattern = r"(\.\.\.|- |[ ,.?!;:\"()])"
        text = text.replace(" -", "- ") # Unify dash with other punctuations

        in_quote = 0
        cur_punc = []
        bert_word_index = 1

        for word in re.split(pattern, text.lower()):
            if word == "":
                continue

            if word == "\"":
                if in_quote == 1:
                    in_quote = 0
                else:
                    in_quote = 1
                continue

            if word == "- " or word == "-":
                cur_punc.append('-')
                continue

            if re.match(pattern, word) and word != " ":
                cur_punc.append(word)
                continue

            if word == " ":
                phonemes.append((' ', cur_punc, in_quote, bert_word_index))
                cur_punc = []
                continue

            # Пропускаем символы, которых нет в словаре, или конвертируем
            if word in self.dictionary:
                word_phonemes = self.dictionary[word].split()
            else:
                # Конвертируем только допустимые символы
                cleaned_word = ''.join(ch for ch in word if ch.isalpha() or ch in "+#-")
                if not cleaned_word:
                    continue  # полностью игнорируем слово из "мусора"
                word_phonemes = self.convert(cleaned_word).split()

            if word_pos:
                word_phonemes = self.add_pos(word_phonemes)

            for p in word_phonemes:
                phonemes.append((p, [], in_quote, bert_word_index))

            cur_punc = []
            bert_word_index += 1

        phonemes.append((" ", cur_punc, in_quote, bert_word_index))
        phonemes.append(("$", [], 0, bert_word_index))

        last_punc = " "
        last_sentence_punc = " "

        lp_phonemes = []
        phone_bert_embeddings = []
        phoneme_id_map = self.phoneme_id_map

        # Замена по умолчанию — ID пробела
        default_id = phoneme_id_map.get(' ', 0)  # fallback на 0, если даже пробела нет (маловероятно)

        for p in reversed(phonemes):
            if "..." in p[1]:
                last_sentence_punc = "..."
            elif "." in p[1]:
                last_sentence_punc = "."
            elif "!" in p[1]:
                last_sentence_punc = "!"
            elif "?" in p[1]:
                last_sentence_punc = "?"
            elif "-" in p[1]:
                last_sentence_punc = "-"

            if len(p[1]) > 0:
                last_punc = p[1][0]

            if len(p[1]) > 0:
                cur_punc = p[1][0]
            else:
                cur_punc = "_"

            # Получаем ID с fallback
            p0_id = phoneme_id_map.get(p[0], default_id)
            cur_punc_id = phoneme_id_map.get(cur_punc, default_id)
            last_punc_id = phoneme_id_map.get(last_punc, default_id)
            last_sentence_punc_id = phoneme_id_map.get(last_sentence_punc, default_id)

            lp_phonemes.append((p0_id, cur_punc_id, p[2], last_punc_id, last_sentence_punc_id))
            if bert_embeddings is not None:
                phone_bert_embeddings.append(bert_embeddings[p[3]])
        
        lp_phonemes = list(reversed(lp_phonemes))
        phone_bert_embeddings = list(reversed(phone_bert_embeddings))

        return lp_phonemes, phone_bert_embeddings

    def pallatize(self, phones):
        for i, phone in enumerate(phones[:-1]):
            if phone[0] in self.SOFTHARD_CONS:
                if phones[i+1][0] in self.SOFTLETTERS:
                    phones[i] = (self.SOFTHARD_CONS[phone[0]] + "j", 0)
                else:
                    phones[i] = (self.SOFTHARD_CONS[phone[0]], 0)
            if phone[0] in self.OTHER_CONS:
                phones[i] = (self.OTHER_CONS[phone[0]], 0)

    def convert_vowels(self, phones):
        new_phones = []
        prev = ""
        for phone in phones:
            if prev in self.STARTSYL:
                if phone[0] in set(u"яюеё"):
                    new_phones.append("j")
            if phone[0] in self.VOWELS:
                new_phones.append(self.VOWELS[phone[0]] + str(phone[1]))
            else:
                new_phones.append(phone[0])
            prev = phone[0]

        return new_phones

    def convert(self, stressword):
        phones = ("#" + stressword + "#")

        # Assign stress marks
        stress_phones = []
        stress = 0
        for phone in phones:
            if phone == "+":
                stress = 1
            else:
                stress_phones.append((phone, stress))
                stress = 0

        # Pallatize
        self.pallatize(stress_phones)

        # Assign stress
        phones = self.convert_vowels(stress_phones)

        # Filter
        phones = [x for x in phones if x not in self.OTHERS]

        return " ".join(phones)
