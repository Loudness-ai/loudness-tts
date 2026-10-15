"""Tests for ttsprep: python -m unittest discover -s tests (from the package root, with src on the path)."""
import unittest

from loudness_tts.textprep import cut_points, prepare, US_WORDS


def said(text, locale='en-US', us_words=False):
    return ' '.join(segment.text for segment in prepare(text, locale, us_words))


def pauses(text, locale='en-US'):
    return [segment.pause for segment in prepare(text, locale)]


class English(unittest.TestCase):
    def test_titles_do_not_end_sentences(self):
        self.assertEqual(said("Dr. Smith lives on Elm St. near St. Mary's church."),
                         "Doctor Smith lives on Elm Street near Saint Mary's church.")
        self.assertEqual(said('Mr. and Mrs. Jones met Prof Brown.'), 'Mister and Mrs Jones met Professor Brown.')
        self.assertEqual(pauses('Dr. Smith arrived. He sat down.'), ['sentence', ''])
        self.assertEqual([x.text for x in prepare('We met on Elm St. They paid. Visit St. Louis.', 'en-US')],
                         ['We met on Elm Street.', 'They paid.', 'Visit Saint Louis.'])

    def test_abbreviation_at_sentence_end_keeps_the_sentence_end(self):
        segments = prepare('I work at Apple Inc. We make phones.', 'en-US')
        self.assertEqual([s.text for s in segments], ['I work at Apple Incorporated.', 'We make phones.'])

    def test_money(self):
        self.assertEqual(said('It costs $5.50, or €12.99.'), 'It costs 5 dollars and 50 cents, or 12 euros and 99 cents.')
        self.assertEqual(said('$1.5 million, $1 or $0.99, £20'), '1.5 million dollars, 1 dollar or 99 cents, 20 pounds.')

    def test_dates_and_times(self):
        self.assertEqual(said('Call me on 4/5/2026 at 9:05.'), "Call me on April 5th, 20 26 at 9 oh 5.")
        self.assertEqual(said('On 4/5/2026.', 'en-GB'), 'On 4th May 2026.')
        self.assertEqual([s.text for s in prepare('It starts at 9:05 a.m. The doors open at 8:30 p.m. sharp.', 'en-US')],
                         ['It starts at 9 oh 5 AM.', 'The doors open at 8:30 PM sharp.'])
        self.assertEqual(said('Released 2026-10-15, Jan. 5 and at 15:00.'),
                         'Released October 15th, 20 26, January 5th and at 15 hundred.')
        self.assertEqual(said('From 2016/07/03 to 15.01.2020, july 1876 and March 3, 1905.'),
                         'From July 3rd, 20 16 to January 15th, 20 20, july 18 76 and March 3rd, 19 oh 5.')
        self.assertEqual(said('At 14:10:30 or 10:00:00.'), 'At 14 hours 10 minutes and 30 seconds or 10 hours.')
        self.assertEqual(said('At 9.05 AM, 6.30 pm or 11.00 a.m. The bus runs at 60 kmph, 9.05 kg.'),
                         'At 9 oh 5 AM, 6:30 PM or 11 AM. The bus runs at 60 kilometers per hour, 9.05 kilograms.')
        self.assertEqual(said('On june 20 2014, JAN. 15 2020 and 10 000 people.'),
                         'On june 20th, 20 14, January 15th, 20 20 and 10000 people.')
        self.assertEqual(said('It costs $1 000.50, weighs 1 500 kg and is 10 000 km away.'),
                         'It costs 1000 dollars and 50 cents, weighs 1500 kilograms and is 10000 kilometers away.')

    def test_decades_units_ranges_fractions(self):
        self.assertEqual(said("The 1990s, the '80s and the 2000s."), 'The 19 nineties, the eighties and the 2 thousands.')
        self.assertEqual(said('5km away at 60 km/h, 25°C, 44.1kHz, 120 bpm, 5 min.'),
                         '5 kilometers away at 60 kilometers per hour, 25 degrees Celsius, 44.1 kilohertz, '
                         '120 beats per minute, 5 minutes.')
        self.assertEqual(said('Pages 10-20, 1/2 cup, 24/7 support.'), 'Pages 10 to 20, one half cup, 24 7 support.')
        self.assertEqual(said('A 3m wall.'), 'A 3m wall.')  # ambiguous: left to the phonemizer

    def test_numbers_names_and_symbols(self):
        self.assertEqual(said('Henry VIII, World War II, chapter IV and Rocky II.'),
                         'Henry the 8th, World War 2, chapter 4 and Rocky 2.')
        self.assertEqual(said('J. K. Rowling met John F. Kennedy.'), 'J K Rowling met John F Kennedy.')
        self.assertEqual(said('We are #1 vs. them, No. 5 and Fig. 3.'), 'We are number 1 versus them, number 5 and figure 3.')
        self.assertEqual(said('Call +1 555-123-4567.'), 'Call plus 1, 5 5 5, 1 2 3, 4 5 6 7.')
        self.assertEqual(said('A/B tests, his/her notes and/or N/A.'), 'A B tests, his or her notes and or not applicable.')

    def test_web_addresses_are_kept_whole(self):
        self.assertEqual(said('Go to https://www.example.com/pricing/ or mail hi@loudness.ai.'),
                         'Go to example.com/pricing or mail hi@loudness.ai.')

    def test_sentence_ends_are_not_invented(self):
        self.assertEqual(len(prepare('See the examples, e.g. songs, etc. and more.', 'en-US')), 1)
        self.assertEqual([s.text for s in prepare('The U.S. economy grew. The U.S. Then it fell.', 'en-US')],
                         ['The U.S. economy grew.', 'The U.S.', 'Then it fell.'])


class Layout(unittest.TestCase):
    def test_paragraphs_lines_and_lists(self):
        segments = prepare('Introduction\n\nFirst sentence. Second one\n- Point one\n1. Point two', 'en-US')
        self.assertEqual([(s.text, s.pause) for s in segments], [
            ('Introduction.', 'paragraph'), ('First sentence.', 'sentence'), ('Second one.', 'line'),
            ('Point one.', 'line'), ('1, Point two.', '')])

    def test_ellipsis_pauses(self):
        self.assertEqual([(s.text, s.pause) for s in prepare('Wait... what? I see... Then we left...', 'en-US')], [
            ('Wait,', 'ellipsis'), ('what?', 'sentence'), ('I see.', 'sentence'), ('Then we left.', '')])

    def test_ellipsis_before_spanish_question(self):
        self.assertEqual([(s.text, s.pause) for s in prepare('Espera... ¿de verdad? Sí... ¿Qué pasó?', 'es-ES')], [
            ('Espera,', 'ellipsis'), ('¿de verdad?', 'sentence'), ('Sí.', 'sentence'), ('¿Qué pasó?', '')])

    def test_emoji_markdown_and_raw_phonemes_are_removed(self):
        self.assertEqual(said('## Great job 👍 **bold** `code` [[hack]]'), 'Great job bold code [hack].')

    def test_long_runs_get_comma_pauses(self):
        text = ('this sentence keeps going without any punctuation because people often paste long transcripts '
                'or captions that were never punctuated and the voice has to read all of it in one breath which '
                'can make the timing drift and the words run together')
        segments = prepare(text, 'en-US')
        self.assertGreater(len(segments), 1)
        self.assertTrue(all(s.pause == 'clause' and s.text.endswith(',') for s in segments[:-1]))
        self.assertTrue(all(len(s.text.split()) <= 28 for s in segments))

    def test_kept_first_words_keep_their_own_span(self):
        text = 'Felipe VI vive en Elm St.'
        tokens = {tok: text[a:b] for tok, a, b in prepare(text, 'es-ES')[0].tokens}
        self.assertEqual(tokens['Felipe'], 'Felipe')
        self.assertEqual(tokens['sexto'], 'VI')
        english = 'Henry VIII lives on Elm St.'
        tokens = {tok: english[a:b] for tok, a, b in prepare(english, 'en-US')[0].tokens}
        self.assertEqual((tokens['Henry'], tokens['8th'], tokens['Street.']), ('Henry', 'VIII', 'St.'))

    def test_spans_point_at_the_source_text(self):
        text = 'Dr. Smith paid $5.50.\n\nThe end... really'
        for segment in prepare(text, 'en-US'):
            self.assertTrue(text[segment.start:segment.end].strip())
        self.assertEqual([text[s.start:s.end] for s in prepare(text, 'en-US')],
                         ['Dr. Smith paid $5.50.', 'The end...', 'really'])

    def test_other_languages_keep_their_text(self):
        self.assertEqual(said('Das Treffen beginnt um neun. 10-20 Seiten', 'de-DE'),
                         'Das Treffen beginnt um neun. 10 bis 20 Seiten.')
        self.assertEqual(said('नमस्ते। यह परीक्षण है', 'ne-NP'), 'नमस्ते। यह परीक्षण है.')


class AmericanWords(unittest.TestCase):
    def test_only_for_voices_that_need_them(self):
        self.assertEqual(said('The schedule.', us_words=False), 'The schedule.')
        self.assertEqual(said('The schedule.', 'en-GB', us_words=True), 'The schedule.')

    def test_raw_blocks_keep_their_punctuation(self):
        self.assertEqual(said("Dance, can't, half, class.", us_words=True), 'Dance, [[kˈant,]] [[hˈaf,]] class.')
        self.assertEqual(said('Gen Z from A to Z.', us_words=True), 'Gen [[zˈiː]] from A to [[zˈiː.]]')

    def test_generated_words_get_them_too(self):
        self.assertEqual(said('Meet on Jan. 5, use 1/2 cup.', us_words=True),
                         'Meet on [[dʒˈanjuːˌɛɹi]] 5th, use one [[hˈaf]] cup.')

    def test_not_inside_addresses(self):
        self.assertEqual(said('See schedule.com now', us_words=True), 'See schedule.com now.')

    def test_symbols_are_ones_the_voices_were_trained_on(self):
        # The British espeak symbol set of the community voices (no ɚ, ɾ, ᵻ, æ or oʊ).
        allowed = set("abdefhijklmnprstuvwzðŋɐɑɒɔəɛɜɡɪɹʃʊʌʒθˈˌː,.;:!?)' ")
        for word, phonemes in US_WORDS.items():
            self.assertFalse(set(phonemes) - allowed, word)


class Timing(unittest.TestCase):
    def test_cuts_follow_word_gaps(self):
        self.assertEqual(cut_points([5, 5, 2], 12, {0, 5, 10}), [0, 5, 10, 12])

    def test_every_word_gets_sounds(self):
        # "policy is necessary": a cut halfway between two gaps goes to the right one.
        cuts = cut_points([7, 2, 10], 20, {0, 8, 10})
        self.assertEqual(cuts, [0, 8, 10, 20])

    def test_words_read_together_are_shared(self):
        # espeak read "for the" as one word: the cut falls inside it.
        self.assertEqual(cut_points([3, 2, 4], 8, {0, 4}), [0, 2, 4, 8])

    def test_one_word(self):
        self.assertEqual(cut_points([3], 7, {0}), [0, 7])


class Spanish(unittest.TestCase):
    def test_titles_money_dates_times(self):
        self.assertEqual(said('El Sr. García y la Dra. López llegan a las 3:30.', 'es-ES'),
                         'El señor García y la doctora López llegan a las 3 y media.')
        self.assertEqual(said('Cuesta 5,50 € o $12.99.', 'es-ES'), 'Cuesta 5 euros con 50 céntimos o 12 dólares con 99 centavos.')
        self.assertEqual(said('Cuesta $1.500.', 'es-MX'), 'Cuesta 1500 pesos.')
        self.assertEqual(said('Nos vemos el 15/10/2026 a las 10 a. m. o a las 15:00.', 'es-ES'),
                         'Nos vemos el 15 de octubre de 2026 a las 10 de la mañana o a las 15 horas.')
        self.assertEqual([s.text for s in prepare('Abrimos a las 9 a. m. Luego cerramos.', 'es-ES')],
                         ['Abrimos a las 9 de la mañana.', 'Luego cerramos.'])

    def test_addresses_names_units(self):
        self.assertEqual(said('Vivo en la c/ Mayor, en EE. UU. y en la Av. de Mayo.', 'es-ES'),
                         'Vivo en la calle Mayor, en Estados Unidos y en la avenida de Mayo.')
        self.assertEqual(said('Felipe VI, Isabel II y Luis XIV.', 'es-ES'), 'Felipe sexto, Isabel segunda y Luis 14.')
        self.assertEqual(said('Hay 25 °C, 1 h y 60 km/h.', 'es-ES'), 'Hay 25 grados Celsius, una hora y 60 kilómetros por hora.')

    def test_numbers_ordinals_and_prices(self):
        self.assertEqual(said('Hay 3 000 000 y 10 000. El 10.º y 1,50 $. Llegó en 01 h.', 'es-ES'),
                         'Hay 3000000 y 10000. El 10º y un dólar con 50 centavos. Llegó en una hora.')
        self.assertEqual(said('Cuesta 1 000 €, pesa 1 500 kg y hay 900 000 000 de personas.', 'es-ES'),
                         'Cuesta 1000 euros, pesa 1500 kilos y hay 900000000 de personas.')

    def test_phone_numbers_written_in_spain(self):
        self.assertEqual(said('Llama al 912 345 678, al 91 234 56 78 o al 612 34 56 78.', 'es-ES'),
                         'Llama al 9 1 2, 3 4 5, 6 7 8, al 9 1, 2 3 4, 5 6, 7 8 o al 6 1 2, 3 4, 5 6, 7 8.')
        self.assertEqual(said('Son 712 345 678 euros.', 'es-ES'), 'Son 712345678 euros.')
        self.assertEqual(said('A 2*8 board, **bold** and a note*'), 'A 2*8 board, bold and a note.')

    def test_brand(self):
        self.assertEqual(said('Visita loudness.ai.', 'es-ES'), 'Visita Láudnes punto ei ai.')


if __name__ == '__main__':
    unittest.main()
