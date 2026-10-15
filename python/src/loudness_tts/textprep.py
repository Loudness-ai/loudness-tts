# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Asad Shehzad / Loudness.ai.
# Vendored unchanged from ttsprep (https://loudness.ai/research/piper-tts-text-normalization/), the text
# preparation layer of the Loudness.ai speech engine. This file stays under the Apache License 2.0 (see
# LICENSE-APACHE in this package); the rest of loudness-tts is GPL-3.0-or-later.
"""Text preparation for the Loudness.ai speech engine.

Turns what a visitor typed into what the voice should say. Written forms that the
phonemizer (espeak-ng) reads badly become words: abbreviations, money, dates,
times, decades, units, ranges, phone numbers, fractions and Roman numerals (English
and Spanish). Emoji, list markers and markdown go. The text is cut into segments,
each followed by the right pause, and every segment keeps the span of the text it
came from, so speech can be timed against what was written.

    prepare(text, locale, us_words=False) -> [Segment]

us_words: the voice reads American English but was trained on espeak's British
phonemes, so a short list of words gets American pronunciations as raw phonemes
(only symbols such voices were trained on).
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Segment:
    text: str   # what the voice reads; may hold [[raw phonemes]]
    start: int  # span of the source text it came from
    end: int
    pause: str  # pause after it: clause, ellipsis, sentence, line, paragraph or '' (last)
    tokens: tuple = ()  # (word read, source start, source end) for each word of text


# ---------------------------------------------------------------- word lists

MONTHS = {
    'en': ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
           'November', 'December'],
    'es': ['enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio', 'julio', 'agosto', 'septiembre', 'octubre',
           'noviembre', 'diciembre'],
}
MONTH_ABBR_EN = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Sept': 9,
                 'Oct': 10, 'Nov': 11, 'Dec': 12}
# Words a long unpunctuated run may be split before (a comma pause goes there).
CONJUNCTIONS = {
    'en': {'and', 'but', 'or', 'so', 'because', 'which', 'while', 'when', 'where', 'although', 'though', 'if',
           'then', 'whereas', 'unless', 'until', 'since'},
    'es': {'y', 'pero', 'o', 'porque', 'cuando', 'donde', 'aunque', 'mientras', 'si', 'pues', 'sino', 'entonces'},
    'de': {'und', 'aber', 'oder', 'weil', 'dass', 'wenn', 'denn', 'sondern', 'während', 'obwohl'},
    'fr': {'et', 'mais', 'ou', 'parce', 'quand', 'car', 'donc', 'puis', 'lorsque', 'pendant'},
    'it': {'e', 'ma', 'o', 'perché', 'quando', 'mentre', 'se', 'quindi', 'poi'},
    'pt': {'e', 'mas', 'ou', 'porque', 'quando', 'enquanto', 'se', 'então', 'pois'},
    'nl': {'en', 'maar', 'of', 'omdat', 'wanneer', 'terwijl', 'als', 'dus', 'want'},
}
RANGE_WORD = {'en': 'to', 'es': 'a', 'de': 'bis', 'fr': 'à'}
PLUS_WORD = {'en': 'plus', 'es': 'más', 'de': 'plus', 'fr': 'plus', 'it': 'più', 'pt': 'mais', 'nl': 'plus'}
# A single initial before one of these is more likely the end of a sentence.
SENTENCE_STARTERS = {
    'en': {'The', 'This', 'That', 'These', 'Those', 'Then', 'There', 'They', 'It', 'We', 'He', 'She', 'I', 'You',
           'But', 'And', 'So', 'A', 'An', 'In', 'On', 'At', 'If', 'When', 'After', 'Before', 'Now', 'Today', 'Our',
           'My', 'Your', 'His', 'Her', 'Their', 'What', 'Why', 'How', 'Who', 'Where', 'Yes', 'No', 'Next', 'Also'},
    'es': {'El', 'La', 'Los', 'Las', 'Un', 'Una', 'Y', 'Pero', 'Luego', 'Entonces', 'Es', 'Hoy', 'Mi', 'Su', 'Sí',
           'No', 'En', 'Con', 'Por', 'Para', 'Si', 'Cuando', 'Después', 'Antes', 'Ahora', 'Qué', 'Cómo', 'Ella', 'Él'},
}
REGNAL_NAMES = {
    'en': {'Henry', 'Edward', 'George', 'William', 'Elizabeth', 'Charles', 'Louis', 'James', 'Richard', 'Mary',
           'John', 'Paul', 'Pius', 'Leo', 'Benedict', 'Gregory', 'Innocent', 'Alexander', 'Peter', 'Catherine',
           'Philip', 'Frederick', 'Ramesses', 'Constantine', 'Napoleon', 'Edmund', 'Harold', 'Stephen', 'Anne',
           'Victoria', 'Nicholas', 'Ivan', 'Francis', 'Clement', 'Urban', 'Boniface', 'Sixtus', 'Felipe', 'Juan',
           'Carlos', 'Alfonso', 'Fernando', 'Isabella', 'Ferdinand', 'Otto', 'Wilhelm', 'Gustav', 'Olaf', 'Haakon'},
    'es': {'Felipe', 'Juan', 'Pablo', 'Carlos', 'Alfonso', 'Fernando', 'Isabel', 'Luis', 'Enrique', 'Pedro',
           'Jaime', 'Benedicto', 'Pío', 'León', 'Gregorio', 'Clemente', 'Inocencio', 'Urbano', 'Sixto', 'Alejandro',
           'Napoleón', 'Ricardo', 'Eduardo', 'Jorge', 'Guillermo', 'Catalina', 'Federico', 'María', 'Francisco'},
}
FEMININE_ES = {'Isabel', 'Catalina', 'María'}
ORDINAL_ES = ['', 'primero', 'segundo', 'tercero', 'cuarto', 'quinto', 'sexto', 'séptimo', 'octavo', 'noveno',
              'décimo']
ROMAN_KEYWORDS_EN = ('chapter', 'part', 'volume', 'book', 'act', 'scene', 'section', 'phase', 'level', 'season',
                     'episode', 'stage', 'round', 'grade', 'class', 'type', 'title', 'article', 'appendix', 'psalm',
                     'war', 'bowl')
DECADES_EN = {1: 'tens', 2: 'twenties', 3: 'thirties', 4: 'forties', 5: 'fifties', 6: 'sixties', 7: 'seventies',
              8: 'eighties', 9: 'nineties'}
FRACTIONS = {
    'en': {(1, 2): 'one half', (1, 3): 'one third', (2, 3): 'two thirds', (1, 4): 'one quarter',
           (3, 4): 'three quarters', (1, 8): 'one eighth', (3, 8): 'three eighths', (5, 8): 'five eighths',
           (7, 8): 'seven eighths'},
    'es': {(1, 2): 'un medio', (1, 3): 'un tercio', (2, 3): 'dos tercios', (1, 4): 'un cuarto',
           (3, 4): 'tres cuartos', (1, 8): 'un octavo', (3, 8): 'tres octavos', (5, 8): 'cinco octavos',
           (7, 8): 'siete octavos'},
}
# Currency: (singular, plural, minor singular, minor plural).
CURRENCY_EN = {'$': ('dollar', 'dollars', 'cent', 'cents'), '€': ('euro', 'euros', 'cent', 'cents'),
               '£': ('pound', 'pounds', 'penny', 'pence'), '¥': ('yen', 'yen', '', ''),
               '₹': ('rupee', 'rupees', 'paisa', 'paise'), 'USD': ('US dollar', 'US dollars', 'cent', 'cents'),
               'EUR': ('euro', 'euros', 'cent', 'cents'), 'GBP': ('pound', 'pounds', 'penny', 'pence')}
CURRENCY_ES = {'€': ('euro', 'euros', 'céntimo', 'céntimos'), '£': ('libra', 'libras', 'penique', 'peniques'),
               '¥': ('yen', 'yenes', '', ''), 'USD': ('dólar estadounidense', 'dólares estadounidenses', 'centavo', 'centavos'),
               'EUR': ('euro', 'euros', 'céntimo', 'céntimos')}
DOLLAR_ES = ('dólar', 'dólares', 'centavo', 'centavos')
PESO_ES = ('peso', 'pesos', 'centavo', 'centavos')
SCALES_EN = {'k': 'thousand', 'thousand': 'thousand', 'm': 'million', 'mn': 'million', 'million': 'million',
             'b': 'billion', 'bn': 'billion', 'billion': 'billion', 'trillion': 'trillion'}
# Units after a number: (singular, plural). Matched case-sensitively, longest first.
UNITS_EN = {
    'km/h': ('kilometer per hour', 'kilometers per hour'), 'kph': ('kilometer per hour', 'kilometers per hour'),
    'kmph': ('kilometer per hour', 'kilometers per hour'),
    'mph': ('mile per hour', 'miles per hour'), 'm/s': ('meter per second', 'meters per second'),
    'km': ('kilometer', 'kilometers'), 'cm': ('centimeter', 'centimeters'), 'mm': ('millimeter', 'millimeters'),
    'kg': ('kilogram', 'kilograms'), 'mg': ('milligram', 'milligrams'), 'lbs': ('pound', 'pounds'),
    'lb': ('pound', 'pounds'), 'oz': ('ounce', 'ounces'), 'mi': ('mile', 'miles'), 'ft': ('foot', 'feet'),
    'yd': ('yard', 'yards'), 'ml': ('milliliter', 'milliliters'), 'mL': ('milliliter', 'milliliters'),
    '°C': ('degree Celsius', 'degrees Celsius'), '°F': ('degree Fahrenheit', 'degrees Fahrenheit'),
    'kWh': ('kilowatt hour', 'kilowatt hours'), 'kW': ('kilowatt', 'kilowatts'), 'kHz': ('kilohertz', 'kilohertz'),
    'MHz': ('megahertz', 'megahertz'), 'GHz': ('gigahertz', 'gigahertz'), 'Hz': ('hertz', 'hertz'),
    'dB': ('decibel', 'decibels'), 'KB': ('kilobyte', 'kilobytes'), 'kB': ('kilobyte', 'kilobytes'),
    'MB': ('megabyte', 'megabytes'), 'GB': ('gigabyte', 'gigabytes'), 'TB': ('terabyte', 'terabytes'),
    'Mbps': ('megabit per second', 'megabits per second'), 'ms': ('millisecond', 'milliseconds'),
    'hrs': ('hour', 'hours'), 'hr': ('hour', 'hours'), 'fps': ('frame per second', 'frames per second'),
    'bpm': ('beat per minute', 'beats per minute'), 'BPM': ('beat per minute', 'beats per minute'),
    '°': ('degree', 'degrees'),
}
# Units that need a space before them (too ambiguous when attached to the number).
SPACED_UNITS_EN = {'m': ('meter', 'meters'), 'g': ('gram', 'grams'), 'min': ('minute', 'minutes'),
                   'L': ('liter', 'liters'), 'h': ('hour', 'hours')}
# (singular, plural, feminine)
UNITS_ES = {
    'km/h': ('kilómetro por hora', 'kilómetros por hora', False), 'km': ('kilómetro', 'kilómetros', False),
    'cm': ('centímetro', 'centímetros', False), 'mm': ('milímetro', 'milímetros', False),
    'kg': ('kilo', 'kilos', False), 'mg': ('miligramo', 'miligramos', False), 'ml': ('mililitro', 'mililitros', False),
    '°C': ('grado Celsius', 'grados Celsius', False), '°F': ('grado Fahrenheit', 'grados Fahrenheit', False),
    'kWh': ('kilovatio hora', 'kilovatios hora', False), 'kHz': ('kilohercio', 'kilohercios', False),
    'Hz': ('hercio', 'hercios', False), 'dB': ('decibelio', 'decibelios', False),
    'KB': ('kilobyte', 'kilobytes', False), 'MB': ('megabyte', 'megabytes', False),
    'GB': ('gigabyte', 'gigabytes', False), 'TB': ('terabyte', 'terabytes', False),
    'bpm': ('pulso por minuto', 'pulsos por minuto', False), 'BPM': ('pulso por minuto', 'pulsos por minuto', False),
    '°': ('grado', 'grados', False),
}
SPACED_UNITS_ES = {'m': ('metro', 'metros', False), 'g': ('gramo', 'gramos', False), 'l': ('litro', 'litros', False),
                   'L': ('litro', 'litros', False), 'h': ('hora', 'horas', True), 'min': ('minuto', 'minutos', False)}
UNITS_MX = {'dB': ('decibel', 'decibeles', False), 'Hz': ('hertz', 'hertz', False), 'kHz': ('kilohertz', 'kilohertz', False)}

# Abbreviations. Titles come before a name; street words after one.
TITLES_EN = {'Mr': 'Mister', 'Mrs': 'Mrs', 'Ms': 'Miz', 'Dr': 'Doctor', 'Prof': 'Professor', 'Rev': 'Reverend',
             'Hon': 'Honorable', 'Gen': 'General', 'Col': 'Colonel', 'Capt': 'Captain', 'Lt': 'Lieutenant',
             'Sgt': 'Sergeant', 'Gov': 'Governor', 'Sen': 'Senator', 'Rep': 'Representative', 'Pres': 'President',
             'Fr': 'Father', 'Mt': 'Mount', 'Ft': 'Fort', 'St': 'Saint'}
TITLES_WITHOUT_DOT_EN = {'Mr', 'Mrs', 'Ms', 'Dr', 'Prof', 'St', 'Mt', 'Capt', 'Sgt', 'Lt'}
STREETS_EN = {'St': 'Street', 'Ave': 'Avenue', 'Blvd': 'Boulevard', 'Rd': 'Road', 'Ln': 'Lane', 'Dr': 'Drive',
              'Hwy': 'Highway', 'Pkwy': 'Parkway', 'Sq': 'Square'}
AFTER_NAME_EN = {'Jr': 'Junior', 'Sr': 'Senior', 'Inc': 'Incorporated', 'Ltd': 'Limited', 'Corp': 'Corporation',
                 'Co': 'Company', 'Bros': 'Brothers'}
BEFORE_NUMBER_EN = {'no': 'number', 'nos': 'numbers', 'fig': 'figure', 'figs': 'figures', 'vol': 'volume',
                    'ch': 'chapter', 'p': 'page', 'pp': 'pages', 'art': 'article', 'sec': 'section',
                    'est': 'established', 'ext': 'extension', 'tel': 'telephone'}
ANYWHERE_EN = {'approx': 'approximately', 'dept': 'department', 'govt': 'government', 'misc': 'miscellaneous',
               'vs': 'versus'}
TITLES_ES = {'Sr': 'señor', 'Sra': 'señora', 'Srta': 'señorita', 'Sres': 'señores', 'Sras': 'señoras',
             'Dr': 'doctor', 'Dra': 'doctora', 'Lic': 'licenciado', 'Lcda': 'licenciada', 'Lda': 'licenciada',
             'Ing': 'ingeniero', 'Prof': 'profesor', 'Profa': 'profesora', 'Dña': 'doña', 'Sto': 'santo',
             'Sta': 'santa', 'Gral': 'general', 'Av': 'avenida', 'Avda': 'avenida', 'Pza': 'plaza', 'Pº': 'paseo',
             'c/': 'calle', 'C/': 'calle'}
ANYWHERE_ES = {'Ud': 'usted', 'Uds': 'ustedes', 'Vd': 'usted', 'Vds': 'ustedes', 'aprox': 'aproximadamente',
               'Aprox': 'aproximadamente', 'dpto': 'departamento', 'Dpto': 'departamento', 'tel': 'teléfono',
               'Tel': 'teléfono', 'admón': 'administración'}
BEFORE_NUMBER_ES = {'núm': 'número', 'Núm': 'número', 'nº': 'número', 'Nº': 'número', 'n.º': 'número',
                    'No': 'número', 'no': 'número', 'pág': 'página', 'págs': 'páginas', 'Pág': 'página',
                    'cap': 'capítulo', 'Cap': 'capítulo', 'vol': 'volumen', 'Vol': 'volumen'}

# Brand names said the way the brand is said, in languages that would read them otherwise.
BRAND = {'es': {'loudness.ai': 'Láudnes punto ei ai', 'loudness': 'Láudnes'}}

# American pronunciations, in the British espeak symbol set (what the community US
# voices were trained on), for words whose British form these voices would copy.
US_WORDS = {
    'schedule': 'skˈɛdʒuːl', 'schedules': 'skˈɛdʒuːlz', 'scheduled': 'skˈɛdʒuːld', 'scheduling': 'skˈɛdʒuːlɪŋ',
    'tomato': 'təmˈeɪtəʊ', 'tomatoes': 'təmˈeɪtəʊz', 'either': 'ˈiːðə', 'neither': 'nˈiːðə',
    'privacy': 'pɹˈaɪvəsi', 'vitamin': 'vˈaɪtəmɪn', 'vitamins': 'vˈaɪtəmɪnz', 'garage': 'ɡəɹˈɑːʒ',
    'garages': 'ɡəɹˈɑːʒɪz', 'leisure': 'lˈiːʒə', 'advertisement': 'ˌadvətˈaɪzmənt',
    'advertisements': 'ˌadvətˈaɪzmənts', 'herb': 'ˈɜːb', 'herbs': 'ˈɜːbz', 'herbal': 'ˈɜːbəl',
    'oregano': 'ɔːɹˈɛɡənəʊ', 'zebra': 'zˈiːbɹə', 'yogurt': 'jˈəʊɡət', 'mobile': 'mˈəʊbəl', 'missile': 'mˈɪsəl',
    'missiles': 'mˈɪsəlz', 'fertile': 'fˈɜːtəl', 'hostile': 'hˈɒstəl', 'fragile': 'fɹˈadʒəl', 'futile': 'fjˈuːtəl',
    'sterile': 'stˈɛɹəl', 'agile': 'ˈadʒəl', 'docile': 'dˈɒsəl', 'versatile': 'vˈɜːsətəl',
    'lieutenant': 'luːtˈɛnənt', 'vase': 'vˈeɪs', 'clerk': 'klˈɜːk', 'process': 'pɹˈɒsɛs', 'processes': 'pɹˈɒsɛsɪz',
    'processing': 'pɹˈɒsɛsɪŋ', 'processed': 'pɹˈɒsɛst', 'progress': 'pɹˈɒɡɹɛs', 'patriot': 'pˈeɪtɹiət',
    'patriotic': 'pˌeɪtɹiˈɒtɪk', 'dynasty': 'dˈaɪnəsti', 'ballet': 'balˈeɪ', 'glacier': 'ɡlˈeɪʃə',
    'laboratory': 'lˈabɹətˌɔːɹi', 'inventory': 'ˈɪnvəntˌɔːɹi', 'territory': 'tˈɛɹɪtˌɔːɹi',
    'territories': 'tˈɛɹɪtˌɔːɹiz', 'dictionary': 'dˈɪkʃənˌɛɹi', 'dictionaries': 'dˈɪkʃənˌɛɹiz',
    'secretary': 'sˈɛkɹətˌɛɹi', 'necessary': 'nˈɛsəsˌɛɹi', 'unnecessary': 'ʌnnˈɛsəsˌɛɹi',
    'military': 'mˈɪlɪtˌɛɹi', 'library': 'lˈaɪbɹˌɛɹi', 'libraries': 'lˈaɪbɹˌɛɹiz', 'ordinary': 'ˈɔːdɪnˌɛɹi',
    'extraordinary': 'ɛkstɹˈɔːdɪnˌɛɹi', 'temporary': 'tˈɛmpəɹˌɛɹi', 'category': 'kˈatəɡˌɔːɹi',
    'categories': 'kˈatəɡˌɔːɹiz', 'mandatory': 'mˈandətˌɔːɹi', 'february': 'fˈɛbɹuːˌɛɹi',
    'january': 'dʒˈanjuːˌɛɹi', 'primary': 'pɹˈaɪmˌɛɹi', 'voluntary': 'vˈɒlənˌtɛɹi', 'stationary': 'stˈeɪʃənˌɛɹi',
    'dietary': 'dˈaɪətˌɛɹi', 'secondary': 'sˈɛkənˌdɛɹi', 'preliminary': 'pɹɪlˈɪmɪnˌɛɹi',
    'vocabulary': 'vəʊkˈabjʊlˌɛɹi', 'cemetery': 'sˈɛmɪtˌɛɹi', 'monastery': 'mˈɒnəstˌɛɹi', 'contrary': 'kˈɒntɹˌɛɹi',
    'arbitrary': 'ˈɑːbɪtɹˌɛɹi', 'sanitary': 'sˈanɪtˌɛɹi', 'legendary': 'lˈɛdʒənˌdɛɹi',
    'revolutionary': 'ɹˌɛvəlˈuːʃənˌɛɹi', 'imaginary': 'ɪmˈadʒɪnˌɛɹi', 'customary': 'kˈʌstəmˌɛɹi',
    'literary': 'lˈɪtəɹˌɛɹi', 'observatory': 'əbzˈɜːvətˌɔːɹi', 'dormitory': 'dˈɔːmɪtˌɔːɹi',
    'explanatory': 'ɛksplˈanətˌɔːɹi', 'repository': 'ɹɪpˈɒzɪtˌɔːɹi', "can't": 'kˈant', 'half': 'hˈaf',
    'halves': 'hˈavz', 'calf': 'kˈaf', 'rather': 'ɹˈaðə', 'banana': 'bənˈanə', 'been': 'bɪn', 'router': 'ɹˈaʊtə',
    'routers': 'ɹˈaʊtəz', 'lever': 'lˈɛvə', 'niche': 'nˈɪtʃ', 'pasta': 'pˈɑːstə', 'basil': 'bˈeɪzəl',
    'charade': 'ʃəɹˈeɪd', 'depot': 'dˈiːpəʊ', 'new': 'nˈuː', 'news': 'nˈuːz', 'knew': 'nˈuː',
    'student': 'stˈuːdənt', 'students': 'stˈuːdənts', 'tune': 'tˈuːn', 'tunes': 'tˈuːnz', 'duty': 'dˈuːti',
    'during': 'dˈʊəɹɪŋ', 'tuesday': 'tˈuːzdeɪ', 'due': 'dˈuː', 'introduce': 'ˌɪntɹədˈuːs',
    'produce': 'pɹədˈuːs', 'reduce': 'ɹɪdˈuːs', 'avenue': 'ˈavənuː', 'z': 'zˈiː',
}

# Raw phoneme blocks for US_WORDS. Punctuation right after a word goes inside its block:
# Piper drops punctuation between a raw block and the following text.
US_PATTERN = re.compile(r"(?<![\w@./'\u2019-])(" + '|'.join(re.escape(w) for w in sorted(US_WORDS, key=len, reverse=True))
                        + r")(?![\w@'\u2019-]|\.\w)([,.;:!?)]*)", re.IGNORECASE)


def us_block(m: re.Match) -> str:
    return '[[' + US_WORDS[m.group(1).lower().replace('\u2019', "'")] + m.group(2) + ']]'


EMOJI = re.compile('[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U00002B00-\U00002BFF\U0001F1E6-\U0001F1FF'
                   '\U000E0020-\U000E007F︎️‍⃣]+')
SENTENCE_END = re.compile('[.!?。！？։۔؟।॥]+["\'”’»)\\]]*(?=\\s|$)')
CLOSERS = '"\'”’»)]'
RUN_BREAK = re.compile(r'[,;:—–،、，]$|^[—–-]$')
MAX_RUN_WORDS = 28


def ordinal_en(n: int) -> str:
    suffix = 'th' if 11 <= n % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f'{n}{suffix}'


def roman_value(s: str) -> int:
    values = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100}
    total = 0
    for i, ch in enumerate(s):
        v = values[ch]
        total += -v if i + 1 < len(s) and values[s[i + 1]] > v else v
    return total


def valid_roman(s: str) -> bool:
    return bool(re.fullmatch(r'(?:C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})', s)) and bool(s)


# A number with thousands grouped by spaces, and an amount after a number.
GROUPED = r'\d{1,3}(?:[ \u00A0\u202F]\d{3})+(?!\d)'
AMOUNT_AFTER = re.compile(r'\s?(?:[$€£¥₹%]|euros?|d[oó]lar(?:es)?|pesos?|libras?|millones|mil)\b', re.IGNORECASE)


def parse_amount(text: str) -> tuple[int, int | None]:
    """'1,250.50' / '1.250,50' / '12.99' -> (whole, cents). A last separator followed
    by one or two digits is the decimal point; any other separator groups thousands."""
    m = re.fullmatch(r'(.*?)(?:[.,](\d{1,2}))?', text)
    whole, cents = m.group(1), m.group(2)
    whole = re.sub(r'[.,\s  ]', '', whole) or '0'
    return int(whole), (int(cents.ljust(2, '0')) if cents else None)


def capital_start(word: str) -> bool:
    return bool(word) and word[0].isupper()


# ---------------------------------------------------------------- rules

class _Line:
    """One line of text: rules claim spans of the original and say what to read instead."""

    def __init__(self, text: str, base: int, post=None, disabled=frozenset()):
        self.text = text
        self.base = base
        self.post = post  # applied to every replacement (pronunciations of generated words)
        self.disabled = disabled  # rule groups switched off (research ablations only)
        self.group = ''
        self.claims: list[tuple[int, int, str, str]] = []

    def free(self, start: int, end: int) -> bool:
        return all(end <= s or e <= start for s, e, _, _ in self.claims)

    def apply(self, pattern: re.Pattern, replace) -> None:
        if self.group in self.disabled:
            return
        for m in pattern.finditer(self.text):
            if m.start() == m.end() or not self.free(m.start(), m.end()):
                continue
            out = replace(m)
            if out is None:
                continue
            text, brk = out if isinstance(out, tuple) else (out, '')
            start = m.start()
            # A rule that keeps the first word ("Felipe VI" -> "Felipe sexto") claims
            # only the rest, so each written word keeps its own spoken words.
            lead = re.match(r'\S+\s+', m.group(0))
            if lead and text.startswith(lead.group(0)) and len(text) > len(lead.group(0)):
                start += len(lead.group(0))
                text = text[len(lead.group(0)):]
            if self.post and text and '[[' not in text:
                text = self.post(text)
            self.claims.append((start, m.end(), text, brk))

    def build(self):
        chars, starts, ends, breaks = [], [], [], {}
        pos = 0

        def keep(upto):
            for i in range(pos, upto):
                chars.append(self.text[i])
                starts.append(self.base + i)
                ends.append(self.base + i + 1)

        for s, e, text, brk in sorted(self.claims):
            keep(s)
            for ch in text:
                chars.append(ch)
                starts.append(self.base + s)
                ends.append(self.base + e)
            if brk:
                breaks[len(chars)] = brk
            pos = e
        keep(len(self.text))
        return ''.join(chars), starts, ends, breaks


def clock_with_seconds(m: re.Match, hours, minutes, seconds, joiner: str, one=None) -> str:
    """14:10:30 as "14 hours 10 minutes and 30 seconds" (parts that are zero are left out)."""
    parts = []
    for value, names in zip((int(m.group(1)), int(m.group(2)), int(m.group(3))), (hours, minutes, seconds)):
        if value or (not parts and names is hours):
            count = (one or {}).get(names[0], '1') if value == 1 else str(value)
            parts.append(f'{count} {names[0] if value == 1 else names[1]}')
    return parts[0] if len(parts) == 1 else ' '.join(parts[:-1]) + f' {joiner} ' + parts[-1]


def sentence_follows(m: re.Match) -> bool:
    """After an abbreviation's dot: does a new sentence start (capital letter or end of line)?"""
    rest = m.string[m.end():]
    return not rest.strip() or bool(re.match(r'\s+["\'“‘(]?[A-ZÁÉÍÓÚÑ]', rest))


def expansion(word: str, m: re.Match, dotted: bool):
    """An expanded abbreviation keeps its dot as a sentence end when a sentence follows."""
    if dotted and sentence_follows(m):
        return word + '.', 'sentence'
    return word


class Preparer:
    def __init__(self, locale: str, us_words: bool, disabled=()):
        self.disabled = frozenset(disabled)
        self.locale = locale
        self.lang, _, self.region = locale.partition('-')
        self.us_words = us_words and locale == 'en-US'
        self.spaced = self.lang not in ('zh', 'ja', 'th')  # words separated by spaces
        # A currency or unit after a number, for languages with money and unit rules.
        units, spaced = {'en': (UNITS_EN, SPACED_UNITS_EN),
                         'es': (dict(UNITS_ES, **(UNITS_MX if self.region == 'MX' else {})), SPACED_UNITS_ES)}.get(self.lang, ({}, {}))
        names = '|'.join(re.escape(n) for n in sorted(units, key=len, reverse=True))
        self.symbol_after = re.compile(r'\s?(?:[$€£¥₹]|USD|EUR|GBP|' + names + r')(?![\w/])|\s(?:'
                                       + '|'.join(re.escape(n) for n in spaced) + r')(?![\w/]|\.\w)') if units else None

    # ---- rules shared by every language
    def generic_rules(self, line: _Line, first: bool) -> None:
        line.group = 'addresses'
        line.apply(re.compile(r'[\w.+-]+@[\w-]+(?:\.[\w-]+)+'), lambda m: m.group(0))
        line.apply(re.compile(r'(?:https?://)(?:www\.)?[^\s]*[^\s.,;:!?)\]]'),
                   lambda m: re.sub(r'^https?://(?:www\.)?', '', m.group(0)).rstrip('/'))
        line.apply(re.compile(r'(?<![\w@.])www\.([^\s]*[^\s.,;:!?)\]])'), lambda m: m.group(1).rstrip('/'))
        line.apply(re.compile(r'(?<![\w@./])[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.(?:com|org|net|ai|io|co|uk|es|mx|de|fr|'
                              r'it|app|dev|edu|gov|info|me|tv|us|ca|au|in|pk)\b(?:/[^\s]*[^\s.,;:!?)\]])?'),
                   lambda m: m.group(0).rstrip('/'))
        line.group = 'lists'
        if first:  # list markers and markdown headings at the start of a line
            line.apply(re.compile(r'^\s*#{1,6}\s+'), lambda m: '')
            line.apply(re.compile(r'^\s*(?:[-*•‣◦▪●·–—]+)\s+'), lambda m: '')
            line.apply(re.compile(r'^\s*(\d{1,3})[.)]\s+'), lambda m: m.group(1) + ', ')
            line.apply(re.compile(r'^\s*([a-zA-Z])\)\s+'), lambda m: '')

    def late_generic_rules(self, line: _Line) -> None:
        line.group = 'markup'
        line.apply(re.compile(r'(\*\*|__)(?=\S)(.+?)(?<=\S)\1'), lambda m: m.group(2))
        line.apply(re.compile(r'(?<![\w*])\*(?=\S)([^*]+?)(?<=\S)\*(?![\w*])'), lambda m: m.group(1))
        line.apply(re.compile(r'(?<!\w)[*`]+|[*`]+(?!\w)'), lambda m: '')
        line.apply(re.compile(r'#(?=[^\W\d_])'), lambda m: '')
        line.apply(EMOJI, lambda m: '')
        line.group = 'pauses'
        line.apply(re.compile(r'(?:\.\s?){2,}\.|…'), self.ellipsis)
        line.apply(re.compile(r'(?<=\S)\s+[-–—]+\s+(?=\S)'), lambda m: ', ')
        line.apply(re.compile(r'(?<=\w)[—](?=\w)'), lambda m: ', ')

    def ellipsis(self, m: re.Match):
        rest = m.string[m.end():]
        if not rest.strip():
            return '.', 'sentence'
        nxt = rest.lstrip()[:1]
        if nxt and nxt in '"\'“¿¡':  # an opening mark: the word after it decides
            nxt = rest.lstrip()[1:2] or nxt
        if rest[:1].isspace() and nxt and nxt.isupper():
            return '.', 'sentence'
        if nxt in '?!.,;:':
            return ''
        return ',', 'ellipsis'

    def phone_rule(self, line: _Line) -> None:
        if self.lang in ('zh', 'ja'):
            return
        plus = PLUS_WORD.get(self.lang, '')
        line.group = 'phone'

        def phone(m: re.Match):
            raw = m.group(0)
            digits = re.sub(r'\D', '', raw)
            groups = re.findall(r'\d+', raw)
            if not 7 <= len(digits) <= 15 or re.fullmatch(r'\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[-.]\d{1,2}[-.]\d{2,4}', raw):
                return None
            prefixed = raw.startswith('+') or '(' in raw
            dashed = bool(re.search(r'\d[-.]\d', raw))
            # Spain writes its nine-digit numbers in groups ("912 345 678", "91 234 56 78"); a round
            # number or an amount ("900 000 000 de personas", "600 000 000 euros") is not one.
            spain = (self.lang == 'es' and len(digits) == 9 and digits[0] in '6789'
                     and tuple(map(len, groups)) in ((3, 3, 3), (2, 3, 2, 2), (3, 2, 2, 2))
                     and groups[-1] != '000' and not AMOUNT_AFTER.match(m.string, m.end()))
            if not (prefixed or spain or (dashed and len(groups) >= 3) or (len(groups) >= 3 and any(len(g) == 4 for g in groups))
                    or (dashed and len(groups) == 2 and len(groups[0]) == 3 and len(groups[1]) == 4)):
                return None
            spoken = ', '.join(' '.join(g) for g in groups)
            return (f'{plus} ' if raw.startswith('+') and plus else '') + spoken
        line.apply(re.compile(r'(?<![\w+.,/])(?:\+\d{1,3}[\s.-]?)?(?:\(\d{1,4}\)[\s.-]?)?\d{2,4}(?:[\s.-]\d{2,4}){1,4}'
                              r'(?![\w.,/]\d|\d)'), phone)

    def number_rules(self, line: _Line) -> None:
        line.group = 'numbers'
        symbol = self.symbol_after

        def grouped(m: re.Match):
            # "$1 000" and "1 500 kg" are read by the money and unit rules.
            if symbol and (re.search(r'[$€£¥₹]\s?$', m.string[:m.start()]) or symbol.match(m.string, m.end())):
                return None
            return re.sub(r'\s', '', m.group(0))
        # Thousands grouped with spaces ("10 000", "3 000 000"), read as one number.
        line.apply(re.compile(r'(?<![\d.,])\d{1,3}(?:[ \u00A0\u202F]\d{3})+(?!\d)(?![.,]\d)'), grouped)
        if self.lang == 'es':  # "10.º", "1.ª": espeak reads ordinals only without the dot
            line.apply(re.compile(r'(\d+)\.([ºª])'), lambda m: m.group(1) + m.group(2))

    # ---- English
    def english_rules(self, line: _Line) -> None:
        months = MONTHS['en']
        us = self.region != 'GB' and self.region != 'IE'

        def year(digits: str) -> str:
            # American dates say years in pairs: "twenty twenty-six", "nineteen oh five".
            y = int(digits)
            if not (self.region == 'US' and (1100 <= y <= 1999 or 2010 <= y <= 2099)):
                return digits
            tail = digits[2:]
            return f'{digits[:2]} hundred' if tail == '00' else f'{digits[:2]} oh {tail[1]}' if tail[0] == '0' else f'{digits[:2]} {tail}'

        def date(day: int, month: int, year_digits: str) -> str | None:
            if not (1 <= month <= 12 and 1 <= day <= 31):
                return None
            said = year(year_digits)
            return f'{months[month - 1]} {ordinal_en(day)}, {said}' if us else f'{ordinal_en(day)} {months[month - 1]} {said}'

        line.group = 'dates'
        line.apply(re.compile(r'\b(\d{4})-(\d{2})-(\d{2})\b'),
                   lambda m: date(int(m.group(3)), int(m.group(2)), m.group(1)))
        line.apply(re.compile(r'\b(\d{4})/(\d{1,2})/(\d{1,2})\b'),
                   lambda m: date(int(m.group(3)), int(m.group(2)), m.group(1)))
        line.apply(re.compile(r'\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b'),
                   lambda m: date(int(m.group(1)), int(m.group(2)), m.group(3)))

        def slash_date(m: re.Match):
            a, b, year = int(m.group(1)), int(m.group(2)), m.group(3)
            if len(year) == 2:
                year = ('20' if int(year) < 50 else '19') + year
            month, day = (a, b) if us and a <= 12 else (b, a)
            if month > 12:
                month, day = day, month
            return date(day, month, year)
        line.apply(re.compile(r'\b(\d{1,2})/(\d{1,2})/(\d{4}|\d{2})\b'), slash_date)

        def time(m: re.Match):
            hour, minute, suffix = int(m.group(1)), m.group(2), m.group(3) or ''
            half = f' {suffix.replace(".", "").replace(" ", "").upper()}' if suffix else ''
            if minute == '00':
                if suffix:
                    said = f'{hour}{half}'
                elif hour == 12:
                    said = 'noon' if self.region == 'US' else "12 o'clock"
                else:
                    said = f'{hour} hundred' if hour >= 13 or hour == 0 else f"{hour} o'clock"
            elif minute.startswith('0'):
                said = f'{hour} oh {int(minute)}{half}'
            elif suffix.endswith('.'):
                said = f'{hour}:{minute}{half}'
            else:
                return None
            # "at 9:05 a.m. The…": the dot of "a.m." also ends the sentence.
            return expansion(said, m, suffix.endswith('.'))
        line.group = 'times'
        line.apply(re.compile(r'\b(\d{1,2}):([0-5]\d):([0-5]\d)\b'),
                   lambda m: clock_with_seconds(m, ('hour', 'hours'), ('minute', 'minutes'), ('second', 'seconds'), 'and'))
        line.apply(re.compile(r'\b([01]?\d|2[0-3]):([0-5]\d)(?!\d)(\s?[aApP]\.?\s?[mM]\.?(?!\w))?'), time)

        def dot_time(m: re.Match):
            # "9.05 am", common in British writing: only with am/pm, so decimals stay decimals.
            hour, minute, suffix = int(m.group(1)), m.group(2), m.group(3)
            half = ' ' + suffix.replace('.', '').replace(' ', '').upper()
            said = (f'{hour}{half}' if minute == '00' else f'{hour} oh {int(minute)}{half}' if minute[0] == '0'
                    else f'{hour}:{minute}{half}')
            return expansion(said, m, suffix.endswith('.'))
        line.apply(re.compile(r'\b(1[0-2]|0?[1-9])\.([0-5]\d)\s?([aApP]\.?\s?[mM]\.?)(?!\w)'), dot_time)
        line.group = 'money'
        self.money_en(line)

        def decade(m: re.Match):
            century, tens = m.group(1), int(m.group(2))
            if tens == 0:
                if not century:
                    return None
                return f'{century[:2]} hundreds' if century[1] != '0' else f'{century[0]} thousands'
            return (f'{century} ' if century else '') + DECADES_EN[tens]
        line.group = 'decades'
        line.apply(re.compile(r"(?<![\w'])'?(\d{2})?(\d)0s\b"), decade)
        line.group = 'ranges'
        self.ranges(line)
        line.group = 'fractions'
        self.fractions(line)
        line.group = 'units'
        self.units(line, UNITS_EN, SPACED_UNITS_EN, lambda n, unit: unit[0] if n == '1' else unit[1])
        line.group = 'roman'

        def roman_keyword(m: re.Match):
            return f'{m.group(1)} {roman_value(m.group(2))}' if valid_roman(m.group(2)) else None
        line.apply(re.compile(r'\b(' + '|'.join(ROMAN_KEYWORDS_EN) + r'|' + '|'.join(w.title() for w in ROMAN_KEYWORDS_EN)
                              + r'|Vol\.?|vol\.?)\s+([IVXLC]+)\b(?![\'’]?\w)'), roman_keyword)

        def roman_name(m: re.Match):
            if not valid_roman(m.group(2)):
                return None
            value = roman_value(m.group(2))
            return f'{m.group(1)} the {ordinal_en(value)}' if m.group(1) in REGNAL_NAMES['en'] else f'{m.group(1)} {value}'
        line.apply(re.compile(r'\b([A-Z][a-z]+)\s+([IVX]{2,})\b(?![\'’]?[a-z])'), roman_name)

        line.group = 'dates'
        short = '|'.join(MONTH_ABBR_EN)
        line.apply(re.compile(r'\b(' + short + r')\.?\s?(\d{1,2})(?:st|nd|rd|th)?,?\s(\d{4})\b', re.IGNORECASE),
                   lambda m: f'{months[MONTH_ABBR_EN[m.group(1).title()] - 1]} {ordinal_en(int(m.group(2)))}, {year(m.group(3))}'
                   if 1 <= int(m.group(2)) <= 31 else None)
        line.apply(re.compile(r'\b(' + short + r')\.?\s?(\d{1,2})(?![\d:]|st|nd|rd|th)\b', re.IGNORECASE),
                   lambda m: f'{months[MONTH_ABBR_EN[m.group(1).title()] - 1]} {ordinal_en(int(m.group(2)))}'
                   if 1 <= int(m.group(2)) <= 31 else None)
        line.apply(re.compile(r'\b(' + short + r')\.?(?=\s?\d)', re.IGNORECASE),
                   lambda m: months[MONTH_ABBR_EN[m.group(1).title()] - 1])
        named = '|'.join(months)
        line.apply(re.compile(r'\b(' + named + r')\s(\d{1,2})(?:st|nd|rd|th)?,?\s(\d{4})\b', re.IGNORECASE),
                   lambda m: f'{m.group(1)} {ordinal_en(int(m.group(2)))}, {year(m.group(3))}' if 1 <= int(m.group(2)) <= 31 else None)
        line.apply(re.compile(r'\b(' + named + r')\s(\d{4})\b', re.IGNORECASE), lambda m: f'{m.group(1)} {year(m.group(2))}')
        line.apply(re.compile(r'\b(' + named + r')\s(\d{1,2})(?![\d:]|st|nd|rd|th)\b', re.IGNORECASE),
                   lambda m: f'{m.group(1)} {ordinal_en(int(m.group(2)))}' if 1 <= int(m.group(2)) <= 31 else None)
        line.apply(re.compile(r'\b(\d{1,2})\s(' + named + r')\b', re.IGNORECASE),
                   lambda m: f'{ordinal_en(int(m.group(1)))} {m.group(2)}' if 1 <= int(m.group(1)) <= 31 else None)
        line.group = 'abbreviations'
        self.abbreviations_en(line)
        line.group = 'initials'
        self.initials(line)
        line.group = 'symbols'
        line.apply(re.compile(r'#\s?(?=\d)'), lambda m: 'number ')
        line.apply(re.compile(r'~\s?(?=\d)'), lambda m: 'about ')
        line.apply(re.compile(r'(?<![\w/])N/A(?![\w/])|(?<![\w/])n/a(?![\w/])'), lambda m: 'not applicable')
        line.apply(re.compile(r'(?<![\w/])w/(?=\s)'), lambda m: 'with')
        line.group = 'slashes'
        self.word_slashes(line, 'or', {'and/or': 'and or'})

    def money_en(self, line: _Line) -> None:
        def say(symbol: str, amount: str, scale: str | None) -> str | None:
            names = CURRENCY_EN[symbol]
            if scale:
                word = SCALES_EN.get(scale.lower())
                if not word:
                    return None
                number = re.sub(r',(?=\d{3})|\s', '', amount)
                return f'{number} {word} {names[1]}'
            whole, cents = parse_amount(amount)
            parts = []
            if whole or not cents:
                parts.append(f'{whole} {names[0] if whole == 1 else names[1]}')
            if cents and names[2]:
                parts.append(f'{cents} {names[2] if cents == 1 else names[3]}')
            return ' and '.join(parts)
        scale = r'(?:\s?(thousand|million|billion|trillion|bn|mn|[kKmMbB])\b)?'
        number = r'(\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?|' + GROUPED + r'(?:\.\d{1,2})?|\d+(?:[.,]\d{1,2})?)'
        line.apply(re.compile(r'(?<![\w.,])([$€£¥₹])\s?' + number + scale),
                   lambda m: say(m.group(1), m.group(2), m.group(3)))
        line.apply(re.compile(r'(?<![\w.,])' + number + r'\s?([€£¥₹]|USD|EUR|GBP)(?!\w)'),
                   lambda m: say(m.group(2), m.group(1), None))

    def abbreviations_en(self, line: _Line) -> None:
        starters = SENTENCE_STARTERS['en']

        def street(m: re.Match):
            # "Elm St." / "5th Ave", but "St. Louis" is a saint: a following name decides.
            following = re.match(r'\s+([A-Z][a-z]+)', m.string[m.end():])
            if following and following.group(1) not in starters:
                return None
            out = expansion(STREETS_EN[m.group(2)], m, bool(m.group(3)))
            return (f'{m.group(1)} {out[0]}', out[1]) if isinstance(out, tuple) else f'{m.group(1)} {out}'
        line.apply(re.compile(r'\b([A-Z][a-z]+|\d+(?:st|nd|rd|th))\s(' + '|'.join(STREETS_EN) + r')(\.?)(?![\w.])'),
                   street)
        dotless = '|'.join(TITLES_WITHOUT_DOT_EN)
        line.apply(re.compile(r'\b(' + '|'.join(TITLES_EN) + r')\.(?=\s+[A-Z])'), lambda m: TITLES_EN[m.group(1)])
        line.apply(re.compile(r'\b(' + dotless + r')(?=\s+[A-Z][a-z])'), lambda m: TITLES_EN[m.group(1)])
        # A person's title with a dot is never really a sentence end ("Mr. and Mrs. Jones").
        people = '|'.join(t for t in TITLES_EN if t not in ('St', 'Mt', 'Ft'))
        line.apply(re.compile(r'\b(' + people + r')\.(?!\w)'), lambda m: expansion(TITLES_EN[m.group(1)], m, True))

        def after(m: re.Match):
            out = expansion(AFTER_NAME_EN[m.group(2)], m, True)
            return (m.group(1) + out[0], out[1]) if isinstance(out, tuple) else m.group(1) + out
        line.apply(re.compile(r'(?<=\S)(,?\s)(' + '|'.join(AFTER_NAME_EN) + r')\.(?!\w)'), after)
        line.apply(re.compile(r'\b(' + '|'.join(k for k in BEFORE_NUMBER_EN) + '|' +
                              '|'.join(k.title() for k in BEFORE_NUMBER_EN) + r')\.\s?(?=\d)'),
                   lambda m: BEFORE_NUMBER_EN[m.group(1).lower()] + ' ')
        line.apply(re.compile(r'\b(' + '|'.join(ANYWHERE_EN) + '|' + '|'.join(k.title() for k in ANYWHERE_EN)
                              + r'|VS)(\.?)(?!\w)'),
                   lambda m: expansion(ANYWHERE_EN[m.group(1).lower()], m, bool(m.group(2)))
                   if m.group(2) or m.group(1).lower() == 'vs' else None)

    # ---- Spanish
    def spanish_rules(self, line: _Line) -> None:
        months = MONTHS['es']

        def date(day: int, month: int, year: str) -> str | None:
            if not (1 <= month <= 12 and 1 <= day <= 31):
                return None
            return f'{"primero" if day == 1 else day} de {months[month - 1]} de {year}'
        line.group = 'dates'
        line.apply(re.compile(r'\b(\d{4})-(\d{2})-(\d{2})\b'),
                   lambda m: date(int(m.group(3)), int(m.group(2)), m.group(1)))

        def numeric_date(m: re.Match):
            year = m.group(3)
            if len(year) == 2:
                year = ('20' if int(year) < 50 else '19') + year
            return date(int(m.group(1)), int(m.group(2)), year)
        line.apply(re.compile(r'\b(\d{1,2})[/.](\d{1,2})[/.](\d{4}|\d{2})\b'), numeric_date)

        def hour_word(hour: int) -> str:
            return 'una' if hour == 1 else 'veintiuna' if hour == 21 else str(hour)

        def ampm(m: re.Match):
            hour, minute, half = int(m.group(1)), m.group(2), m.group(3).lower()
            if not 1 <= hour <= 12:
                return None
            when = 'de la mañana' if half == 'a' else ('del mediodía' if hour == 12 else 'de la tarde' if hour <= 7
                                                       else 'de la noche')
            return expansion(f'{hour_word(hour)}{self.minutes_es(minute)} {when}', m, m.group(0).endswith('.'))
        line.group = 'times'
        line.apply(re.compile(r'\b(\d{1,2})(?::(\d{2}))?\s?([aApP])\.?\s?[mM]\.?(?!\w)'), ampm)

        def time(m: re.Match):
            hour, minute = int(m.group(1)), m.group(2)
            if minute == '00':
                return f'{hour_word(hour)} horas' if hour >= 13 or hour == 0 else f'{hour_word(hour)} en punto'
            return f'{hour_word(hour)}{self.minutes_es(minute)}'
        line.apply(re.compile(r'\b(\d{1,2}):([0-5]\d):([0-5]\d)\b'),
                   lambda m: clock_with_seconds(m, ('hora', 'horas'), ('minuto', 'minutos'), ('segundo', 'segundos'), 'y',
                                                one={'hora': 'una', 'minuto': 'un', 'segundo': 'un'}))
        line.apply(re.compile(r'\b([01]?\d|2[0-3]):([0-5]\d)(?!\d)'), time)
        line.group = 'money'
        self.money_es(line)
        line.group = 'ranges'
        self.ranges(line)
        line.group = 'fractions'
        self.fractions(line)
        line.group = 'units'
        units = dict(UNITS_ES, **(UNITS_MX if self.region == 'MX' else {}))

        def unit_word(n: str, unit) -> str:
            return unit[0] if n == '1' else unit[1]
        self.units(line, units, SPACED_UNITS_ES, unit_word, one=lambda unit: 'una' if unit[2] else 'un')

        def roman_name(m: re.Match):
            if not valid_roman(m.group(2)) or m.group(1) not in REGNAL_NAMES['es']:
                return None
            value = roman_value(m.group(2))
            if value > 10:
                return f'{m.group(1)} {value}'
            word = ORDINAL_ES[value]
            return f'{m.group(1)} {word[:-1] + "a" if m.group(1) in FEMININE_ES else word}'
        line.group = 'roman'
        line.apply(re.compile(r'\b([A-ZÁÉÍÓÚ][a-záéíóúñ]+)\s+([IVX]+)\b(?![\'’]?[a-z])'), roman_name)
        line.group = 'abbreviations'
        line.apply(re.compile(r'\bEE\.\s?UU\.|\bEEUU\b'), lambda m: expansion('Estados Unidos', m, m.group(0).endswith('.')))
        line.apply(re.compile(r'(?<![\w/])([cC]/)\s?(?=[A-ZÁÉÍÓÚ])'), lambda m: 'calle ')
        titles = '|'.join(re.escape(k) for k in TITLES_ES if not k.endswith('/'))
        line.apply(re.compile(r'(?<!\w)(' + titles + r')\.(?=\s+[A-ZÁÉÍÓÚÑ])'), lambda m: TITLES_ES[m.group(1)])
        line.apply(re.compile(r'(?<!\w)(Av|Avda|Pza|Pº)\.?(?=\s+del?\s)'), lambda m: TITLES_ES[m.group(1)])
        line.apply(re.compile(r'(?<!\w)(' + '|'.join(re.escape(k) for k in BEFORE_NUMBER_ES) + r')\.?\s?(?=\d)'),
                   lambda m: BEFORE_NUMBER_ES[m.group(1)] + ' ' if m.group(1) not in ('No', 'no') or
                   m.group(0).rstrip().endswith('.') else None)
        line.apply(re.compile(r'(?<!\w)(' + '|'.join(re.escape(k) for k in ANYWHERE_ES) + r')\.(?!\w)'),
                   lambda m: expansion(ANYWHERE_ES[m.group(1)], m, True))
        line.group = 'initials'
        self.initials(line)
        line.group = 'symbols'
        line.apply(re.compile(r'#\s?(?=\d)'), lambda m: 'número ')
        line.apply(re.compile(r'~\s?(?=\d)'), lambda m: 'unos ')
        line.group = 'slashes'
        self.word_slashes(line, 'o', {'y/o': 'y o'})

    @staticmethod
    def minutes_es(minute: str | None) -> str:
        if not minute or minute == '00':
            return ''
        return {'15': ' y cuarto', '30': ' y media'}.get(minute, f' y {int(minute)}')

    def money_es(self, line: _Line) -> None:
        dollar = PESO_ES if self.region == 'MX' else DOLLAR_ES

        def say(symbol: str, amount: str) -> str:
            names = dollar if symbol == '$' else CURRENCY_ES[symbol]
            whole, cents = parse_amount(amount)
            text = '' if not whole and cents else f'{"un" if whole == 1 else whole} {names[0] if whole == 1 else names[1]}'
            if cents and names[2]:
                piece = f'{"un" if cents == 1 else cents} {names[2] if cents == 1 else names[3]}'
                text = f'{text} con {piece}' if text else piece
            return text
        number = r'(\d{1,3}(?:[.,\s]\d{3})+(?:[.,]\d{1,2})?|\d+(?:[.,]\d{1,2})?)'
        line.apply(re.compile(r'(?<![\w.,])([$€£¥])\s?' + number + r'(?![\d])'), lambda m: say(m.group(1), m.group(2)))
        line.apply(re.compile(r'(?<![\w.,])' + number + r'\s?([€£¥$]|USD|EUR)(?![\w$])'), lambda m: say(m.group(2), m.group(1)))

    # ---- shared helpers
    def ranges(self, line: _Line) -> None:
        word = RANGE_WORD.get(self.lang)
        if word:
            line.apply(re.compile(r'(?<![\w.,/:+-])(\d{1,4})\s?[-–]\s?(\d{1,4})(?![\w.,/:-]?\d)(?![\w-])'),
                       lambda m: f'{m.group(1)} {word} {m.group(2)}')

    def fractions(self, line: _Line) -> None:
        words = FRACTIONS.get(self.lang, {})

        def fraction(m: re.Match):
            pair = (int(m.group(1)), int(m.group(2)))
            return words.get(pair) or f'{m.group(1)} {m.group(2)}'
        line.apply(re.compile(r'(?<![\w./])(\d{1,3})/(\d{1,3})(?![\w/]|\.\d)'), fraction)

    def units(self, line: _Line, table: dict, spaced: dict, word, one=None) -> None:
        names = sorted(table, key=len, reverse=True)

        def say(m: re.Match, unit):
            number = re.sub(r'\s', '', m.group(1))
            single = re.fullmatch(r'0*1(?:[.,]0+)?', number) is not None
            if one and single:
                return f'{one(unit)} {unit[0]}'
            return f'{number} {word("1" if single else number, unit)}'
        pattern = '|'.join(re.escape(n) for n in names)
        number = r'(?<![\w.,])((?:' + GROUPED + r'|\d+)(?:[.,]\d+)?)'
        line.apply(re.compile(number + r'\s?(' + pattern + r')(?![\w/])'), lambda m: say(m, table[m.group(2)]))
        line.apply(re.compile(number + r'\s(' + '|'.join(spaced) + r')(?![\w/]|\.\w)'),
                   lambda m: say(m, spaced[m.group(2)]))

    def initials(self, line: _Line) -> None:
        # Initials before a name ("J. K. Rowling"), unless a sentence seems to start.
        starters = SENTENCE_STARTERS.get(self.lang, set())
        line.apply(re.compile(r'\b((?:[A-Z]\.\s?){2,})(?=([A-Z][a-z]+))'),
                   lambda m: None if m.group(2) in starters else ' '.join(re.findall(r'[A-Z]', m.group(1))) + ' ')
        for m in re.finditer(r'\b[A-Z][a-z]+ ([A-Z])\.(?= ([A-Z][a-z]+))', line.text):
            if m.group(2) not in starters and line.free(m.start(1), m.end(1) + 1):
                line.claims.append((m.start(1), m.end(1) + 1, m.group(1), ''))

    def word_slashes(self, line: _Line, joiner: str, fixed: dict) -> None:
        def slash(m: re.Match):
            token = m.group(0)
            if token.lower() in fixed:
                return fixed[token.lower()]
            a, b = token.split('/')
            if (len(a) == 1 and len(b) == 1) or (a.isupper() and b.isupper()):
                return f'{a} {b}'
            return f'{a} {joiner} {b}'
        line.apply(re.compile(r'(?<![\w./@:-])[^\W\d_]+/[^\W\d_]+(?![\w./@:-])'), slash)

    def lexicon(self, line: _Line) -> None:
        line.group = 'lexicon'
        brand = BRAND.get(self.lang, {})
        if brand:
            keys = sorted(brand, key=len, reverse=True)
            line.apply(re.compile(r'(?<![\w@./-])(' + '|'.join(re.escape(k) for k in keys) + r')(?![\w@-]|\.\w)',
                                  re.IGNORECASE), lambda m: brand[m.group(1).lower()])
        if self.us_words:
            line.apply(US_PATTERN, us_block)
            line.apply(re.compile(r'(?<=[A-Za-z]-)[Zz](?![\w@-])([,.;:!?)]*)'),
                       lambda m: '[[' + US_WORDS['z'] + m.group(1) + ']]')

    @staticmethod
    def american(text: str) -> str:
        return US_PATTERN.sub(us_block, text)

    # ---- whole text
    def prepare(self, text: str) -> list[Segment]:
        lines = [(m.start(), m.end()) for m in re.finditer(r'[^\n]+', text)]
        segments: list[Segment] = []
        for index, (start, end) in enumerate(lines):
            raw = text[start:end]
            if not raw.strip():
                continue
            line = _Line(raw, start, self.american if self.us_words else None, self.disabled)
            # Visitors cannot send raw phonemes; brand names come before web addresses.
            line.apply(re.compile(r'\[\[|\]\]'), lambda m: m.group(0)[0])
            self.lexicon(line)
            self.generic_rules(line, True)
            self.phone_rule(line)
            self.number_rules(line)
            if self.lang == 'en':
                self.english_rules(line)
            elif self.lang == 'es':
                self.spanish_rules(line)
            else:
                line.group = 'ranges'
                self.ranges(line)
            self.late_generic_rules(line)
            line_segments = self.split(*line.build())
            if not line_segments:
                continue
            following = text[end:lines[index + 1][0]] if index + 1 < len(lines) else ''
            last = line_segments[-1]
            kind = '' if index + 1 == len(lines) else ('paragraph' if following.count('\n') >= 2 else 'line')
            line_segments[-1] = replace(last, text=self.terminate(last.text), pause=kind)
            segments.extend(line_segments)
        if segments:
            final = segments[-1]
            segments[-1] = replace(final, pause='')
        return segments

    @staticmethod
    def terminate(text: str) -> str:
        core = text.rstrip(CLOSERS + ' ')
        if core and unicodedata.category(core[-1]).startswith('P') and core[-1] not in '-–—':
            return text
        if core.endswith((',', '—', '–', '-')):
            return core.rstrip(',-–— ') + '.'
        return text.rstrip() + '.'

    def split(self, norm: str, starts: list[int], ends: list[int], breaks: dict[int, str]) -> list[Segment]:
        cuts: dict[int, str] = dict(breaks)
        for m in SENTENCE_END.finditer(norm):
            rest = norm[m.end():].lstrip()
            if m.group(0)[0] == '.' and rest[:1].islower():
                continue
            cuts.setdefault(m.end(), 'sentence')
        out: list[Segment] = []
        prev = 0
        for pos in sorted(cuts) + [len(norm)]:
            if pos <= prev:
                continue
            out.extend(self.piece(norm, starts, ends, prev, pos, cuts.get(pos, '')))
            prev = pos
        return out

    def piece(self, norm, starts, ends, a, b, kind) -> list[Segment]:
        """A stretch between cuts, with long unpunctuated runs split at a comma pause."""
        words = [(m.start() + a, m.end() + a) for m in re.finditer(r'\S+', norm[a:b])]
        if not words or not any(ch.isalnum() for ch in norm[a:b]):
            return []
        pieces, run, first = [], 0, 0
        conj = CONJUNCTIONS.get(self.lang, set())
        if self.spaced and 'long-runs' not in self.disabled:
            i = 0
            while i < len(words):
                token = norm[words[i][0]:words[i][1]]
                run += 1
                if RUN_BREAK.search(token):
                    run = 0
                elif run > MAX_RUN_WORDS:
                    start_of_run = i - run + 1
                    window = range(start_of_run + 10, min(i + 1, start_of_run + 27))
                    choice = min((j for j in window if norm[words[j][0]:words[j][1]].lower() in conj),
                                 key=lambda j: abs(j - start_of_run - 18), default=start_of_run + 18)
                    pieces.append((first, choice))
                    first = choice
                    run = i - choice + 1
                i += 1
        pieces.append((first, len(words)))
        segs = []
        for n, (w0, w1) in enumerate(pieces):
            s, e = words[w0][0], words[w1 - 1][1]
            text = re.sub(r'\s+', ' ', norm[s:e])
            last = n + 1 == len(pieces)
            if not last:
                text = text.rstrip() + ','
            tokens = tuple((norm[ws:we], min(starts[ws:we]), max(ends[ws:we])) for ws, we in words[w0:w1])
            segs.append(Segment(text, starts[s], ends[e - 1], kind if last else 'clause', tokens))
        return segs


def cut_points(counts: list, total_sounds: int, word_starts: set) -> list:
    """Where each word's sounds begin: as close as possible to its expected share,
    preferring gaps between spoken words, every word getting at least one sound
    (a small dynamic programme around the proportional guess)."""
    scale = total_sounds / sum(counts)
    expected = [count * scale for count in counts]
    layers = [{0: (0.0, None)}]
    done = 0.0
    for k in range(1, len(counts)):
        done += expected[k - 1]
        low, high = max(1, int(done) - 8), min(total_sounds - 1, int(done) + 9)
        layer = {}
        for cut in range(low, high + 1):
            best = None
            for previous, (cost, _) in layers[-1].items():
                if previous >= cut and total_sounds >= len(counts):
                    continue
                step = cost + abs((cut - previous) - expected[k - 1]) + (0.0 if cut in word_starts else 1.5)
                if best is None or step < best[0]:
                    best = (step, previous)
            if best:
                layer[cut] = best
        layers.append(layer or {min(total_sounds, max(layers[-1]) + 1): (0.0, max(layers[-1]))})
    last = min(layers[-1], key=lambda cut: layers[-1][cut][0] + abs((total_sounds - cut) - expected[-1]))
    bounds = [total_sounds, last]
    for layer in reversed(layers[1:]):
        bounds.append(layer[bounds[-1]][1])
    return bounds[::-1]


# Rule groups that can be switched off for research ablations (never on the website).
GROUPS = ('addresses', 'lists', 'lexicon', 'phone', 'numbers', 'dates', 'times', 'money', 'decades', 'ranges', 'fractions',
          'units', 'roman', 'abbreviations', 'initials', 'symbols', 'slashes', 'markup', 'pauses', 'long-runs')


def prepare(text: str, locale: str, us_words: bool = False, disabled=()) -> list[Segment]:
    return Preparer(locale, us_words, disabled).prepare(text)
