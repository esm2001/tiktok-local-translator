"""Integrity checks on real observed translation failures, without an LLM call.

Spanish fixtures exercise the requested preservation of names and complete
captions. These are validator tests, not an application translation substitute.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "translator"))
from src.config import Settings
from src.translation.ollama import OllamaTranslator, _causative_reflexive_form_issues, _format_english_layout, _jerga_agreement_issues, _lexical_argument_issues, _ocr_evidence, _tokens, validate_translation


HEGEL_LINES = [
    "You never admit",
    "you're wrong.",
    "Hegel",
    "He rewrote realty",
    "to fit his theory.",
]
HEGEL_ORIGINAL = "You never admit you're wrong.\nHegel. He rewrote reality to fit his theory."
HEGEL_SPANISH = "Nunca admites que te equivocas.\nHegel. Reescribió la realidad para que encajara con su teoría."


class EnglishLayoutTests(unittest.TestCase):
    def test_same_row_fragments_follow_image_geometry_without_reordering_next_row(self):
        # Real detector geometry from the full-guide cover slide.
        evidence = [
            {'text': 'Guide)', 'confidence': .995, 'polygon': [[808,1481],[1269,1498],[1262,1672],[801,1654]]},
            {'text': '(Full', 'confidence': .914, 'polygon': [[496,1496],[822,1496],[822,1664],[496,1664]]},
            {'text': 'Next paragraph.', 'confidence': .96, 'polygon': [[400,1800],[1400,1800],[1400,1900],[400,1900]]},
        ]
        self.assertEqual([line['text'] for line in _ocr_evidence(evidence)], ['(Full', 'Guide)', 'Next paragraph.'])

    def test_name_separator_preserves_every_caption_sentence_and_word(self):
        lines = ["You drive others away.", "Clara", "Lived it. Wrote about it.", "Changed nothing."]
        original = "You drive others away. Clara Lived it. Wrote about it. Changed nothing."
        formatted = _format_english_layout(original, lines)
        self.assertEqual(formatted, "You drive others away.\nClara.\nLived it. Wrote about it. Changed nothing.")
        self.assertEqual(_tokens(formatted), _tokens(original))

    def test_existing_name_punctuation_is_not_duplicated_and_layout_is_idempotent(self):
        expected = "You never admit you're wrong.\nHegel.\nHe rewrote reality to fit his theory."
        for separator in [" ", ". ", ":\n", ".\n"]:
            original = "You never admit you're wrong. Hegel" + separator + "He rewrote reality to fit his theory."
            with self.subTest(separator=separator):
                formatted = _format_english_layout(original, HEGEL_LINES)
                self.assertEqual(formatted, expected)
                self.assertEqual(_format_english_layout(formatted, HEGEL_LINES), formatted)

    def test_name_at_start_does_not_create_an_empty_heading(self):
        result = _format_english_layout("Clara. She wrote about music.", ["Clara", "She wrote about music."])
        self.assertEqual(result, "Clara.\nShe wrote about music.")

    def test_ambiguous_repeated_name_is_not_split_at_the_wrong_occurrence(self):
        original = "You admire Clara. Clara. She writes books."
        self.assertEqual(_format_english_layout(original, ["You admire Clara.", "Clara", "She writes books."]), original)

    def test_possessive_name_is_not_rewritten(self):
        original = "Clara's notebook was her favorite possession."
        self.assertEqual(_format_english_layout(original, ["Clara", "Her notebook was her favorite possession."]), original)

    def test_no_name_preserves_question_and_call_to_action_unchanged(self):
        original = "Which option belongs to you?\nChoose the one you like."
        self.assertEqual(_format_english_layout(original, ["Which option belongs to you?", "Choose the one you like."]), original)

    def test_list_line_boundaries_survive_missing_bullets_without_changing_words(self):
        lines = ['Resources', '- Audio guides', '- Reading lists', 'Practice tests', 'Study sites']
        original = 'Resources - Audio guides - Reading lists Practice tests Study sites'
        formatted = _format_english_layout(original, lines)
        self.assertEqual(formatted, 'Resources\n- Audio guides\n- Reading lists\nPractice tests\nStudy sites')
        self.assertEqual(_tokens(formatted), _tokens(original))
        self.assertEqual(_format_english_layout(formatted, lines), formatted)

    def test_wrapped_clauses_reflow_but_heading_and_name_remain_separate(self):
        lines = ['2. Compare two options.', 'Consider your first', 'idea carefully. Then',
                 'look at the alternative.', 'Clara', 'She examined both', 'before choosing.']
        original = '\n'.join(lines)
        formatted = _format_english_layout(original, lines)
        self.assertEqual(formatted, '2. Compare two options.\nConsider your first idea carefully. Then look at the alternative.\nClara.\nShe examined both before choosing.')
        self.assertEqual(_tokens(formatted), _tokens(original))
        self.assertEqual(_format_english_layout(formatted, lines), formatted)

    def test_numbered_heading_boundary_does_not_split_protected_marker(self):
        lines = ['7. Observe two leaves.', '__KEEP_EN_0__']
        expected = '\n'.join(lines)
        formatted = _format_english_layout(expected, lines)
        self.assertEqual(formatted, expected)
        self.assertEqual(_format_english_layout(formatted, lines), expected)


class TranslationIntegrityTests(unittest.TestCase):
    def valid(self, **changes):
        return {"original": HEGEL_ORIGINAL, "spanish": HEGEL_SPANISH, **changes}

    def test_transitive_self_sabotage_cannot_become_reflexive_subject_destruction(self):
        source = 'Lena self-sabotages her promotion.'
        with self.assertRaisesRegex(RuntimeError, 'SELF_SABOTAGE_ARGUMENT'):
            validate_translation('1', [source], {'original': source, 'spanish': 'Lena se autodestruye con su ascenso.'})

    def test_self_sabotage_allows_different_target_predicates_with_an_object(self):
        source = 'Lena self-sabotages her promotion.'
        for target in ['Lena sabotea sus posibilidades de ascenso.', 'Lena perjudica sus propias opciones de ascender.']:
            with self.subTest(target=target):
                self.assertEqual(validate_translation('1', [source], {'original': source, 'spanish': target})['spanish'], target)

    def test_self_destruct_and_intransitive_self_sabotage_are_outside_object_guard(self):
        cases = [('The capsule will self-destruct.', 'La cápsula se autodestruirá.'),
                 ('They self-sabotage.', 'Se autodestruyen.'),
                 ('They self-sabotage every day.', 'Se autodestruyen todos los días.')]
        for source, target in cases:
            with self.subTest(source=source):
                validate_translation('1', [source], {'original': source, 'spanish': target})

    def test_external_force_imperative_flags_only_awkward_reflexive_causative_form(self):
        source = 'Force your team to wait.'
        for target in ['Oblígate a que tu equipo espere.', 'Fuérzate a que el equipo espere.']:
            with self.subTest(target=target), self.assertRaisesRegex(RuntimeError, 'CAUSATIVE_REFLEXIVE_FORM.*naturalidad'):
                validate_translation('8', [source], {'original': source, 'spanish': target})

    def test_external_force_accepts_direct_and_other_natural_causative_forms(self):
        source = 'Force your team to wait.'
        for target in ['Obliga a tu equipo a esperar.', 'Fuerza a tu equipo a esperar.', 'Haz que tu equipo espere.']:
            with self.subTest(target=target):
                validate_translation('8', [source], {'original': source, 'spanish': target})

    def test_force_yourself_is_outside_external_causative_naturalness_guard(self):
        cases = [('Force yourself to wait.', 'Oblígate a esperar.'),
                 ('Force yourself to let the team wait.', 'Oblígate a que el equipo espere.')]
        for source, target in cases:
            with self.subTest(source=source):
                validate_translation('8', [source], {'original': source, 'spanish': target})

    def test_causative_naturalness_guard_excludes_nonimperative_negative_and_mixed_context(self):
        sources = ['They force the team to wait.', 'Do not force the team to wait.',
                   'Force yourself to watch. Force your team to wait.', 'Force your own self to wait.']
        for source in sources:
            with self.subTest(source=source):
                self.assertEqual(_causative_reflexive_form_issues(source, 'Oblígate a que tu equipo espere.'), [])

    def test_external_and_own_self_commands_do_not_assume_target_clause_alignment(self):
        source = 'Force the gate to open. Force your own self to stop shaking.'
        target = 'Obliga al portón a abrirse. Oblígate a que tu temblor se detenga.'
        self.assertEqual(_causative_reflexive_form_issues(source, target), [])

    def test_causative_literal_quotes_are_mentions_and_invented_quotes_do_not_bypass_guard(self):
        cases = [('The instruction reads "Force your team to wait."', 'El texto dice "Oblígate a que tu equipo espere."'),
                 ('Force your team to wait. The label reads "Oblígate a que tu equipo espere".',
                  'Obliga a tu equipo a esperar. La etiqueta dice "Oblígate a que tu equipo espere".')]
        for source, target in cases:
            with self.subTest(source=source):
                self.assertEqual(_causative_reflexive_form_issues(source, target), [])
        self.assertTrue(_causative_reflexive_form_issues('Force your team to wait.', '"Oblígate a que tu equipo espere."'))

    def test_causative_force_source_retained_as_protected_english_remains_literal(self):
        source = 'Force your team to wait.\nThen watch the clock.'
        target = '<KEEP_EN>Force your team to wait.</KEEP_EN>\nDespués mira el reloj.'
        result = validate_translation('8', source.splitlines(), {'original': source, 'spanish': target})
        self.assertEqual(result['spanish'], 'Force your team to wait.\nDespués mira el reloj.')

    def test_deliberation_sleep_on_cannot_be_literal_on_an_abstract_object(self):
        source = 'Sleep on a choice until tomorrow.'
        with self.assertRaisesRegex(RuntimeError, 'SLEEP_ON_DECISION'):
            validate_translation('1', [source], {'original': source, 'spanish': 'Duerme sobre una elección hasta mañana.'})

    def test_sleep_on_accepts_different_deliberation_paraphrases(self):
        source = 'Sleep on a choice until tomorrow.'
        for target in ['Espera a mañana para decidir.', 'Piénsalo con calma y decide mañana.']:
            with self.subTest(target=target):
                validate_translation('1', [source], {'original': source, 'spanish': target})

    def test_sleep_physical_objects_and_ignored_opportunities_are_excluded(self):
        cases = [('Sleep on the floor.', 'Duerme en el suelo.'),
                 ('Do not sleep on this opportunity.', 'No dejes pasar esta oportunidad.'),
                 ('The child sleeps on a decision paper.', 'El niño duerme sobre un papel de decisiones.')]
        for source, target in cases:
            with self.subTest(source=source):
                validate_translation('1', [source], {'original': source, 'spanish': target})

    def test_short_sleep_on_heading_cannot_lose_its_temporal_advice(self):
        source = '6. Sleep on a choice.\nChoose tomorrow.'
        for heading in ['6. Considera una elección.', '6. Sueña con una elección.']:
            with self.subTest(heading=heading), self.assertRaisesRegex(RuntimeError, 'SLEEP_ON_DECISION'):
                validate_translation('4', source.splitlines(), {'original': source,
                    'spanish': heading + '\nElige mañana.'})

    def test_sleep_on_heading_accepts_varied_temporal_advice(self):
        source = '6. Sleep on a choice.\nChoose tomorrow.'
        for heading in ['6. Deja reposar la elección.', '6. Espera para elegir.',
                        '6. Decide después de dormir.', '6. Aplaza la elección hasta otro día.']:
            with self.subTest(heading=heading):
                validate_translation('4', source.splitlines(), {'original': source,
                    'spanish': heading + '\nElige mañana.'})

    def test_sleeping_later_in_body_does_not_repair_missing_header_advice(self):
        source = 'Sleep on a choice.\nDecide after resting.'
        target = 'Considera la elección.\nDecide después de dormir.'
        with self.assertRaisesRegex(RuntimeError, 'SLEEP_ON_DECISION'):
            validate_translation('4', source.splitlines(), {'original': source, 'spanish': target})

    def test_header_temporal_guard_excludes_physical_informal_negative_and_ambiguous_uses(self):
        cases = [('Sleep on the couch.', 'Duerme en el sofá.'),
                 ("Don't sleep on these options. They are underrated.", 'No pases por alto estas opciones. Están infravaloradas.'),
                 ('Sleep on an idea?', '¿Reflexionar sobre una idea?'),
                 ('Sleep on an option.', 'Considera una opción.'),
                 ('Sleep on the matter.\nIt is literally printed on a paper sheet.', 'Duerme sobre el asunto.\nEstá literalmente impreso en una hoja de papel.')]
        for source, target in cases:
            with self.subTest(source=source):
                self.assertFalse(any('SLEEP_ON_DECISION' in issue for issue in _lexical_argument_issues(source, target)))

    def test_sleep_on_header_declared_and_retained_in_english_is_exempt(self):
        source = '6. Sleep on a choice.\nChoose tomorrow.'
        target = '<KEEP_EN>6. Sleep on a choice.</KEEP_EN>\nElige mañana.'
        result = validate_translation('4', source.splitlines(), {'original': source, 'spanish': target})
        self.assertEqual(result['spanish'], '6. Sleep on a choice.\nElige mañana.')

    def test_common_jerga_rejects_adjacent_masculine_determiners(self):
        source = 'Explain this jargon clearly.'
        for phrase in ['el jerga', 'del jerga', 'un jerga', 'este jerga', 'El Jerga', 'los jergas']:
            with self.subTest(phrase=phrase), self.assertRaisesRegex(RuntimeError, 'JERGA_GENDER'):
                validate_translation('3', [source], {'original': source, 'spanish': 'Explica ' + phrase + ' con claridad.'})

    def test_jerga_accepts_feminine_or_nonadjacent_determiners(self):
        source = 'Explain this jargon clearly.'
        for phrase in ['esta jerga', 'la jerga', 'el uso de jerga', 'un poco de jerga', 'su jerga']:
            with self.subTest(phrase=phrase):
                validate_translation('3', [source], {'original': source, 'spanish': 'Explica ' + phrase + ' con claridad.'})

    def test_jerga_exact_quoted_mentions_and_source_identified_names_are_exempt(self):
        cases = [('The label reads "el jerga".', 'La etiqueta dice "el jerga".'),
                 ("The label reads 'el jerga'.", "La etiqueta dice 'el jerga'."),
                 ('A band called El Jerga plays tonight.', 'La banda El Jerga actúa esta noche.'),
                 ('Jerga\nThe character spoke.', 'El Jerga habló.')]
        for source, target in cases:
            with self.subTest(source=source):
                self.assertEqual(_jerga_agreement_issues(source, target), [])
        self.assertTrue(_jerga_agreement_issues('Explain jargon.', 'Explica "el jerga".'))

    def test_jerga_in_exact_declared_english_is_not_subject_to_spanish_agreement(self):
        source = 'The label says el jerga. Keep it.'
        target = 'La etiqueta dice <KEEP_EN>el jerga</KEEP_EN>. Consérvala.'
        result = validate_translation('3', [source], {'original': source, 'spanish': target})
        self.assertIn('el jerga', result['spanish'])

    def test_human_brain_input_cannot_be_a_bare_physical_entrance_clause(self):
        source = 'Rest your brain. No input. No notifications.'
        with self.assertRaisesRegex(RuntimeError, 'HUMAN_INFORMATION_INPUT'):
            validate_translation('1', [source], {'original': source, 'spanish': 'Descansa el cerebro. Sin entrada. Sin notificaciones.'})

    def test_human_input_allows_information_or_stimulus_renderings(self):
        source = 'Rest your brain. No input. No notifications.'
        for target in ['Descansa el cerebro. Sin estímulos. Sin notificaciones.',
                       'Descansa el cerebro. Sin entrada de información. Sin notificaciones.']:
            with self.subTest(target=target):
                validate_translation('1', [source], {'original': source, 'spanish': target})

    def test_input_without_human_context_and_technical_brain_are_excluded(self):
        cases = [('No input. The door is closed.', 'Sin entrada. La puerta está cerrada.'),
                 ('The brain of the device hears music. No input.', 'El cerebro del dispositivo escucha música. Sin entrada.')]
        for source, target in cases:
            with self.subTest(source=source):
                validate_translation('1', [source], {'original': source, 'spanish': target})

    def test_complete_reconstruction_allows_obvious_ocr_correction(self):
        result = validate_translation("2", HEGEL_LINES, self.valid())
        self.assertEqual(result["original"], HEGEL_ORIGINAL)
        self.assertEqual(result["spanish"], HEGEL_SPANISH)

    def test_reconstruction_cannot_append_a_fluent_unseen_sentence(self):
        with self.assertRaises(RuntimeError):
            validate_translation("2", HEGEL_LINES, self.valid(
                original=HEGEL_ORIGINAL + " His destiny was written in the stars."))

    def test_unseen_quoted_token_cannot_be_corrected_as_a_one_character_typo(self):
        source = 'They said "glorpz" today.'
        with self.assertRaisesRegex(RuntimeError, 'cita literal'):
            validate_translation('1', [source], {'original': 'They said "glorps" today.', 'spanish': 'Hoy dijeron «glorps».'})

    def test_reconstruction_cannot_remove_negation_with_a_small_character_edit(self):
        with self.assertRaisesRegex(RuntimeError, "negación"):
            validate_translation("2", HEGEL_LINES, self.valid(
                original=HEGEL_ORIGINAL.replace("never", "ever")))

    def test_spanish_cannot_reverse_the_only_explicit_negation(self):
        with self.assertRaisesRegex(RuntimeError, "negación"):
            validate_translation("2", HEGEL_LINES, self.valid(
                spanish=HEGEL_SPANISH.replace("Nunca", "Siempre")))

    def test_short_negation_region_cannot_silently_disappear(self):
        with self.assertRaisesRegex(RuntimeError, "negación"):
            validate_translation("1", ["No", "You admit it."], {
                "original": "You admit it.", "spanish": "Lo admites."})

    def test_single_numbered_heading_cannot_be_changed_to_the_json_id(self):
        source = '2. Consider another option.'
        with self.assertRaisesRegex(RuntimeError, 'número explícito'):
            validate_translation('3', [source], {'original': source, 'spanish': '3. Considera otra opción.'})

    def test_numeric_values_and_repeated_counts_must_survive_spanish(self):
        source = 'Allow 10 minutes twice, then another 10 minutes.'
        with self.assertRaisesRegex(RuntimeError, 'número explícito'):
            validate_translation('1', [source], {'original': source, 'spanish': 'Dedica 10 minutos dos veces y otros 20 minutos.'})

    def test_parenthesized_guide_title_is_translatable_without_a_false_name(self):
        result = validate_translation("1", ["5 ways to increase", "your thinking ability", "(Full", "Guide)"], {
            "original": "5 ways to increase your thinking ability (Full Guide)",
            "spanish": "5 formas de mejorar tu capacidad de pensar (guía completa)",
        })
        self.assertIn("guía completa", result['spanish'])

    def test_ordinary_heading_is_not_a_persons_name(self):
        result = validate_translation("1", ["Freedom", "You are free."], {
            "original": "Freedom. You are free.", "spanish": "Libertad. Eres libre.",
        })
        self.assertIn("Libertad", result['spanish'])

    def test_reconstruction_may_join_broken_lines_and_whitespace(self):
        result = validate_translation("2", HEGEL_LINES, self.valid(original="  You never admit\nyou're wrong. Hegel. He rewrote reality\nto fit his theory.  "))
        self.assertTrue(result["original"].startswith("You never admit"))

    def test_caption_omitted_is_rejected_even_when_length_check_passes(self):
        partial = "You never admit you're wrong. Hegel."
        self.assertGreater(len(partial), len(" ".join(HEGEL_LINES)) * 0.40)
        with self.assertRaises(RuntimeError):
            validate_translation("2", HEGEL_LINES, self.valid(original=partial))

    def test_caption_cannot_be_replaced_by_an_unrelated_long_sentence(self):
        with self.assertRaises(RuntimeError):
            validate_translation("2", HEGEL_LINES, self.valid(original="You never admit you're wrong. Hegel. His most important books concern philosophy, history, and human thought."))

    def test_one_missing_caption_sentence_is_rejected(self):
        lines = ["You push everyone away", "then wonder why", "you're alone.", "Nietzsche", "Lived it. Wrote about it.", "Changed nothing."]
        item = {
            "original": "You push everyone away then wonder why you're alone. Nietzsche. Lived it. Wrote about it.",
            "spanish": "Apartas a todo el mundo y luego te preguntas por qué estás solo. Nietzsche. Lo vivió. Escribió sobre ello.",
        }
        with self.assertRaises(RuntimeError):
            validate_translation("3", lines, item)

    def test_proper_name_substitution_is_rejected(self):
        with self.assertRaises(RuntimeError):
            validate_translation("2", HEGEL_LINES, self.valid(original=HEGEL_ORIGINAL.replace("Hegel", "Kant")))

    def test_proper_name_missing_from_spanish_is_rejected(self):
        with self.assertRaises(RuntimeError):
            validate_translation("2", HEGEL_LINES, self.valid(spanish=HEGEL_SPANISH.replace("Hegel.", "")))

    def test_spanish_is_not_an_english_copy(self):
        with self.assertRaises(RuntimeError):
            validate_translation("2", HEGEL_LINES, self.valid(spanish=HEGEL_ORIGINAL))

    def test_spanish_cannot_mix_a_long_untranslated_heading(self):
        with self.assertRaises(RuntimeError):
            validate_translation("2", HEGEL_LINES, self.valid(spanish="You never admit you're wrong.\nHegel. Reescribió la realidad para que encajara con su teoría."))

    def test_spanish_cannot_copy_an_untranslated_caption(self):
        with self.assertRaises(RuntimeError):
            validate_translation("2", HEGEL_LINES, self.valid(spanish="Nunca admites que te equivocas.\nHegel. He rewrote reality to fit his theory."))

    def test_single_copied_english_verb_is_rejected(self):
        with self.assertRaises(RuntimeError):
            validate_translation("2", HEGEL_LINES, self.valid(spanish="Nunca admites que te equivocas. Hegel. Rewrote la realidad para que encajara con su teoría."))

    def test_shared_spanish_cognate_is_allowed(self):
        result = validate_translation("1", ["Your social life is important."], {
            "original": "Your social life is important.",
            "spanish": "Tu vida social es importante.",
        })
        self.assertIn("social", result["spanish"])

    def test_spanish_plural_cognate_is_not_mistaken_for_english(self):
        result = validate_translation('2', ['Your best ideas are useful.'], {
            'original': 'Your best ideas are useful.', 'spanish': 'Tus mejores ideas son útiles.',
        })
        self.assertIn('ideas', result['spanish'])

    def test_english_plural_is_not_accepted_from_a_shared_singular(self):
        with self.assertRaisesRegex(RuntimeError, 'inglesas'):
            validate_translation('1', ['They love animals.'], {
                'original': 'They love animals.', 'spanish': 'Les gustan los animals.',
            })

    def test_brand_inside_a_sentence_is_not_an_untranslated_english_word(self):
        result = validate_translation('1', ['Join the ExampleBrand in my bio.'], {
            'original': 'Join the ExampleBrand in my bio.',
            'spanish': 'Únete a ExampleBrand en mi biografía.',
        })
        self.assertIn('ExampleBrand', result['spanish'])

    def test_shared_anatomical_adjective_and_loanword_are_allowed(self):
        for english, spanish in [
            ('The prefrontal cortex.', 'La corteza prefrontal.'),
            ('Premium guides.', 'Guías premium.'),
            ('Simple tests.', 'Tests simples.'),
        ]:
            with self.subTest(english=english):
                result = validate_translation('1', [english], {'original': english, 'spanish': spanish})
                self.assertEqual(result['spanish'], spanish)

    def test_non_cognate_complex_is_still_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'palabras inglesas'):
            validate_translation('1', ['This complex task is useful.'], {
                'original': 'This complex task is useful.', 'spanish': 'Esta tarea complex es útil.',
            })

    def test_writing_about_an_object_cannot_become_writing_the_object(self):
        lines = ['She wrote about music.']
        for spanish in ['La escribió.', 'La ha escrito.']:
            with self.subTest(spanish=spanish), self.assertRaisesRegex(RuntimeError, 'write about'):
                validate_translation('1', lines, {'original': lines[0], 'spanish': spanish})
        result = validate_translation('1', lines, {'original': lines[0], 'spanish': 'Escribió acerca de la música.'})
        self.assertIn('acerca', result['spanish'])

    def test_explicit_you_cannot_become_an_indefinite_new_actor(self):
        original = 'You drive people away.'
        with self.assertRaisesRegex(RuntimeError, 'actor indefinido'):
            validate_translation('1', [original], {'original': original, 'spanish': 'Alguien aleja a la gente.'})
        self.assertEqual(validate_translation('1', [original], {
            'original': original, 'spanish': 'Alejas a la gente.'})['spanish'], 'Alejas a la gente.')

    def test_identity_question_cannot_become_a_possession_question(self):
        for original in ['Who are you?', 'Which one are you?', 'What kind of person are you?']:
            with self.subTest(original=original), self.assertRaisesRegex(RuntimeError, 'identidad'):
                validate_translation('1', [original], {'original': original, 'spanish': '¿Quién es el tuyo?'})
        result = validate_translation('1', ['Who are you?'], {'original': 'Who are you?', 'spanish': '¿Quién eres?'})
        self.assertEqual(result['spanish'], '¿Quién eres?')

    def test_location_and_possession_questions_keep_their_own_meaning(self):
        for original, spanish in [('Where are you?', '¿Dónde estás?'),
                                  ('Which one belongs to you?', '¿Cuál es el tuyo?')]:
            with self.subTest(original=original):
                self.assertEqual(validate_translation('1', [original], {
                    'original': original, 'spanish': spanish})['spanish'], spanish)

    def test_declared_uncertainty_requires_a_verbatim_source_span(self):
        original = 'I saw her duck. Then I walked home.'
        spanish = '<KEEP_EN>I saw her duck.</KEEP_EN> Después caminé a casa.'
        result = validate_translation('1', [original], {'original': original, 'spanish': spanish})
        self.assertEqual(result['spanish'], 'I saw her duck. Después caminé a casa.')
        with self.assertRaisesRegex(RuntimeError, 'evidencia original'):
            validate_translation('1', [original], {
                'original': original, 'spanish': '<KEEP_EN>I bought a spaceship.</KEEP_EN> Después caminé a casa.'})

    def test_malformed_uncertainty_marker_is_not_a_valid_translation(self):
        with self.assertRaisesRegex(RuntimeError, 'incompleto'):
            validate_translation('1', ['I saw her duck.'], {
                'original': 'I saw her duck.', 'spanish': '<KEEP_EN>I saw her duck.'})

    def test_translation_cannot_connect_separate_list_items_in_one_sentence(self):
        original = 'Resources\n- Audio guides\n- Reading lists\nPractice tests\nStudy sites'
        with self.assertRaisesRegex(RuntimeError, 'lista'):
            validate_translation('1', original.splitlines(), {
                'original': original,
                'spanish': 'Recursos - Guías de audio - Listas de lectura Pruebas prácticas de sitios de estudio',
            })

    def test_missing_or_non_string_fields_are_rejected(self):
        invalid = [
            {"original": HEGEL_ORIGINAL},
            {"spanish": HEGEL_SPANISH},
            {"original": [HEGEL_ORIGINAL], "spanish": HEGEL_SPANISH},
            {"original": HEGEL_ORIGINAL, "spanish": None},
        ]
        for item in invalid:
            with self.subTest(item=item), self.assertRaises(RuntimeError):
                validate_translation("2", HEGEL_LINES, item)

    def test_readable_text_cannot_return_empty_or_half_empty_fields(self):
        for original, spanish in [("", ""), (HEGEL_ORIGINAL, ""), ("", HEGEL_SPANISH)]:
            with self.subTest(original=original, spanish=spanish), self.assertRaises(RuntimeError):
                validate_translation("2", HEGEL_LINES, {"original": original, "spanish": spanish})

    def test_compound_ocr_word_can_be_split_without_losing_coverage(self):
        lines = ["You self-sabotage", "every good thing", "in your life.", "Schopenhauer", "Suffering was his", "comfortzone."]
        item = {
            "original": "You self-sabotage every good thing in your life.\nSchopenhauer. Suffering was his comfort zone.",
            "spanish": "Saboteas todo lo bueno de tu vida.\nSchopenhauer. El sufrimiento era su zona de confort.",
        }
        self.assertEqual(validate_translation("4", lines, item)["original"], item["original"])

    def test_traditional_name_accents_are_preserved(self):
        cases = [
            (["Socrates", "They killed him for it."], "Socrates. They killed him for it.", "Sócrates. Lo mataron por ello."),
            (["Diogenes", "Homeless by choice.", "No regrets."], "Diogenes. Homeless by choice. No regrets.", "Diógenes. Sin hogar por elección. Sin remordimientos."),
        ]
        for lines, original, spanish in cases:
            with self.subTest(name=lines[0]):
                result = validate_translation("1", lines, {"original": original, "spanish": spanish})
                self.assertEqual(result["spanish"], spanish)

    def test_singular_male_caption_object_cannot_become_plural(self):
        for clitic, verb in [("Los", "mataron"), ("Los", "detuvieron"),
                             ("Los", "detenían"), ("Los", "arrestaban"),
                             ("Los", "han detenido")]:
            with self.subTest(verb=verb), self.assertRaisesRegex(RuntimeError, "objeto singular"):
                validate_translation("5", ["Socrates", "They killed him for it."], {
                    "original": "Socrates. They killed him for it.",
                    "spanish": f"Sócrates. {clitic} {verb} por ello.",
                })

    def test_singular_female_caption_object_cannot_become_plural(self):
        for verb in ["arrestó", "arrestaron"]:
            with self.subTest(verb=verb), self.assertRaisesRegex(RuntimeError, "objeto singular"):
                validate_translation("1", ["Clara", "She arrested her."], {
                    "original": "Clara. She arrested her.",
                    "spanish": f"Clara. Las {verb}.",
                })

    def test_singular_female_caption_object_is_accepted(self):
        result = validate_translation("1", ["Clara", "She arrested her."], {
            "original": "Clara. She arrested her.",
            "spanish": "Clara. La arrestó.",
        })
        self.assertIn("La arrestó", result["spanish"])

    def test_plural_article_before_people_does_not_trigger_clitic_check(self):
        result = validate_translation("5", ["Socrates", "They arrested him."], {
            "original": "Socrates. They arrested him.",
            "spanish": "Sócrates. Las personas lo detuvieron.",
        })
        self.assertIn("Las personas", result["spanish"])

    def test_singular_object_check_uses_final_sentence_without_name(self):
        with self.assertRaisesRegex(RuntimeError, "objeto singular"):
            validate_translation("1", ["Your life changes.", "They killed him for it."], {
                "original": "Your life changes. They killed him for it.",
                "spanish": "Tu vida cambia. Los mataron por ello.",
            })

    def test_caption_object_check_skips_ambiguous_mixed_objects(self):
        result = validate_translation("1", ["Socrates", "They remembered him and them."], {
            "original": "Socrates. They remembered him and them.",
            "spanish": "Sócrates. Los recordaron.",
        })
        self.assertIn("Los recordaron", result["spanish"])

    def test_possessive_her_does_not_require_a_singular_object(self):
        result = validate_translation("1", ["Clara", "She remembered her friends."], {
            "original": "Clara. She remembered her friends.",
            "spanish": "Clara. Los recordaba.",
        })
        self.assertIn("Los recordaba", result["spanish"])

    def test_plural_english_caption_object_remains_plural(self):
        result = validate_translation("1", ["Clara", "She arrested them."], {
            "original": "Clara. She arrested them.",
            "spanish": "Clara. Las arrestaron.",
        })
        self.assertIn("Las arrestaron", result["spanish"])

    def test_named_third_person_caption_cannot_switch_to_first_person(self):
        lines = ["Nietzsche", "Lived it.", "Wrote about it.", "Changed nothing."]
        original = "Nietzsche.\nLived it. Wrote about it. Changed nothing."
        for caption in ["Lo viví. Escribió sobre ello. No cambió nada.",
                        "Lo vivió. Lo escribí. No cambió nada.",
                        "Lo vivió. Escribió sobre ello. No cambié nada.",
                        "Lo viví. Lo escribí. No cambié nada."]:
            with self.subTest(caption=caption), self.assertRaisesRegex(RuntimeError, "Nietzsche en tercera persona"):
                validate_translation("3", lines, {"original": original,
                    "spanish": "Nietzsche.\n" + caption})

    def test_named_third_person_caption_preserves_third_person(self):
        lines = ["Nietzsche", "Lived it.", "Wrote about it.", "Changed nothing."]
        spanish = "Nietzsche.\nLo vivió. Escribió sobre ello. No cambió nada."
        result = validate_translation("3", lines, {
            "original": "Nietzsche.\nLived it. Wrote about it. Changed nothing.",
            "spanish": spanish,
        })
        self.assertEqual(result["spanish"], spanish)

    def test_named_caption_rejects_explicit_first_person_and_irregular_past(self):
        for caption in ["Yo tomé una decisión.", "Nosotros tomamos una decisión.",
                        "Nosotras tomamos una decisión.", "Hice una elección.",
                        "Fui valiente.", "Dije la verdad.", "Tuve una idea.",
                        "Fuimos valientes."]:
            with self.subTest(caption=caption), self.assertRaisesRegex(RuntimeError, "primera persona"):
                validate_translation("1", ["Clara", "She made a decision."], {
                    "original": "Clara. She made a decision.", "spanish": "Clara. " + caption,
                })

    def test_quoted_first_person_caption_is_allowed(self):
        result = validate_translation("1", ["Clara", 'She said: "I lived it."'], {
            "original": 'Clara. She said: "I lived it."',
            "spanish": "Clara. Dijo: «Lo viví».",
        })
        self.assertIn("Lo viví", result["spanish"])

    def test_explicit_first_person_english_caption_is_allowed(self):
        cases = [("We were brave.", "Fuimos valientes."),
                 ("My choices made a difference.", "Mis elecciones cambiaron las cosas.")]
        for source, spanish in cases:
            with self.subTest(source=source):
                result = validate_translation("1", ["Clara", source], {
                    "original": "Clara. " + source, "spanish": "Clara. " + spanish,
                })
                self.assertEqual(result["spanish"], "Clara. " + spanish)

    def test_accented_caption_nouns_are_not_first_person_verbs(self):
        spanish = "Clara. No café. No té. No puré."
        result = validate_translation("1", ["Clara", "No coffee. No tea. No puree."], {
            "original": "Clara. No coffee. No tea. No puree.", "spanish": spanish,
        })
        self.assertEqual(result["spanish"], spanish)

    def test_the_ego_noun_is_not_an_explicit_first_person_subject(self):
        result = validate_translation("1", ["Clara", "She studied the ego."], {
            "original": "Clara. She studied the ego.", "spanish": "Clara. Estudió el yo.",
        })
        self.assertIn("el yo", result["spanish"])

    def test_suit_noun_and_accented_adverb_are_not_first_person_verbs(self):
        cases = [("She wore a suit.", "Llevó un traje."),
                 ("Not like that.", "No así."),
                 ("She hoped he was not there.", "Esperaba que no esté allí.")]
        for original, spanish in cases:
            with self.subTest(spanish=spanish):
                result = validate_translation("1", ["Clara", original], {
                    "original": "Clara. " + original, "spanish": "Clara. " + spanish,
                })
                self.assertEqual(result["spanish"], "Clara. " + spanish)

    def test_first_person_heading_does_not_trigger_named_caption_guard(self):
        result = validate_translation("1", ["I made a mistake.", "Clara", "She was kind."], {
            "original": "I made a mistake.\nClara.\nShe was kind.",
            "spanish": "Yo cometí un error.\nClara.\nFue amable.",
        })
        self.assertIn("Yo cometí", result["spanish"])


def response(content, done_reason="stop"):
    import json
    result = Mock(status_code=200)
    result.json.return_value = {"message": {"content": json.dumps(content)}, "done": True,
                               "done_reason": done_reason, "prompt_eval_count": 10, "eval_count": 20,
                               "total_duration": 1_000_000}
    return result




class TranslationStagesTests(unittest.TestCase):
    def run_mocked(self, replies, slides=None, settings=None):
        slides = slides or {2: HEGEL_LINES}
        translator = OllamaTranslator(settings or Settings(ollama_url="http://127.0.0.1:11434", ollama_model="qwen2.5:3b", ollama_translation_model=""))
        with patch("src.translation.ollama.requests.Session") as factory:
            session = factory.return_value.__enter__.return_value
            session.post.side_effect = replies
            output = translator.translate(slides)
        return translator, session, output

    def test_two_stages_generate_only_one_language_each_with_required_ids(self):
        translator, session, output = self.run_mocked([response({"2": HEGEL_ORIGINAL}), response({"2": HEGEL_SPANISH})])
        self.assertEqual(output['2']['original'], _format_english_layout(HEGEL_ORIGINAL, HEGEL_LINES))
        self.assertEqual(output['2']['spanish'], HEGEL_SPANISH)
        self.assertFalse(output['2']['fallback_to_english'])
        self.assertFalse(output['2']['translation']['semantic_equivalence_verified'])
        self.assertEqual(session.post.call_count, 2)
        first, second = [call.kwargs["json"] for call in session.post.call_args_list]
        for payload in [first, second]:
            self.assertEqual(payload["format"]["required"], ["2"])
            self.assertFalse(payload["format"]["additionalProperties"])
            self.assertEqual(payload["model"], "qwen2.5:3b")
            self.assertEqual(payload["options"]["num_ctx"], 4096)
            self.assertEqual(payload["options"]["repeat_penalty"], 1.0)
        self.assertEqual(first['format']['properties']['2'], {'type': 'string'})
        self.assertEqual(second['format']['properties']['2']['required'], ['uncertain_spans', 'spanish'])
        self.assertIn("English OCR", first["messages"][0]["content"])
        self.assertIn("castellano", second["messages"][0]["content"])
        self.assertEqual(translator.metrics["request_count"], 2)
        self.assertEqual(translator.metrics["input_tokens"], 20)
        self.assertEqual(translator.metrics["output_tokens"], 40)

    def test_spanish_contract_assesses_source_before_whole_slide_translation_without_scores(self):
        translator = OllamaTranslator(Settings())
        messages = translator._messages("translation", {"9": "Text to translate."}, translator._schema({"9": ""}))
        self.assertEqual(len(messages), 2)
        self.assertIn('diapositiva COMPLETA', messages[0]['content'])
        self.assertIn('source EXACTO', messages[0]['content'])
        schema = translator._schema({'9': ''}, 'translation')['properties']['9']
        self.assertEqual(schema['required'], ['uncertain_spans', 'spanish'])
        self.assertNotIn('confidence', schema['properties'])

    def test_free_source_schema_keeps_runtime_exact_membership_and_current_data_only(self):
        import json
        data = {'3': 'Observe violet petals.', '8': 'Check bronze hinges.'}
        translator = OllamaTranslator(Settings())
        schema = translator._schema(data, 'translation')
        for key in data:
            source_schema = schema['properties'][key]['properties']['uncertain_spans']['items']['properties']['source']
            self.assertEqual(source_schema, {'type': 'string'})
        messages = translator._messages('translation', data, schema)
        self.assertEqual([message['role'] for message in messages], ['system', 'user'])
        self.assertEqual(json.loads(messages[-1]['content'].split('\nDATA / DATOS:\n')[1]), data)
        self.assertNotIn('FORMAT EXAMPLE', str(messages))
        for invented in [data['8'], 'preserve every number', 'Write about a topic means producing text']:
            with self.subTest(invented=invented), self.assertRaisesRegex(RuntimeError, 'substring exacto'):
                translator._translation_content('3', [data['3']], data['3'], {
                    'uncertain_spans': [{'source': invented, 'reason': 'AMBIGUOUS_SOURCE'}],
                    'spanish': invented})

    def test_reconstruction_prompt_uses_a_generic_merged_word_example(self):
        translator = OllamaTranslator(Settings())
        messages = translator._messages("reconstruction", {"9": ["Text."]}, translator._schema({"9": ""}))
        system = messages[0]["content"]
        self.assertIn("Separate words accidentally merged by OCR", system)
        self.assertIn("livingroom -> living room", system)
        self.assertNotIn("comfortzone", system)
        self.assertNotIn("realty", system)

    def test_two_models_are_unloaded_and_routed_in_order_without_images(self):
        settings = Settings(ollama_url="http://127.0.0.1:11434", ollama_model="qwen2.5:3b", ollama_translation_model="gemma3:4b")
        translator, session, output = self.run_mocked([
            response({}), response({"2": HEGEL_ORIGINAL}), response({}), response({"2": HEGEL_SPANISH}),
        ], settings=settings)
        calls = session.post.call_args_list
        self.assertEqual([call.args[0].rsplit("/", 1)[-1] for call in calls], ["generate", "chat", "generate", "chat"])
        self.assertEqual([call.kwargs["json"]["model"] for call in calls], ["gemma3:4b", "qwen2.5:3b", "qwen2.5:3b", "gemma3:4b"])
        self.assertEqual(calls[0].kwargs["json"], {"model": "gemma3:4b", "keep_alive": 0})
        self.assertEqual(calls[2].kwargs["json"], {"model": "qwen2.5:3b", "keep_alive": 0})
        for call in (calls[1], calls[3]):
            payload = call.kwargs["json"]
            self.assertEqual(payload["keep_alive"], "30m")
            self.assertEqual(payload["options"]["num_gpu"], 0)
            self.assertNotIn("images", payload)
            self.assertTrue(all("images" not in message for message in payload["messages"]))
        import json
        spanish_values = json.loads(calls[3].kwargs["json"]["messages"][-1]["content"].split("\nDATA / DATOS:\n")[1])
        self.assertEqual(spanish_values["2"], _format_english_layout(HEGEL_ORIGINAL, HEGEL_LINES))
        self.assertEqual(output["2"]["original"], spanish_values["2"])
        self.assertEqual(translator.metrics["request_count"], 2)
        self.assertEqual(translator.metrics["model_switch_count"], 2)
        self.assertEqual([call["model"] for call in translator.metrics["calls"]], ["qwen2.5:3b", "gemma3:4b"])
        self.assertTrue(all(switch["completed"] for switch in translator.metrics["model_switches"]))
        self.assertGreaterEqual(translator.metrics["model_switch_seconds"], 0)

    def test_same_model_skips_memory_switches(self):
        settings = Settings(ollama_url="http://127.0.0.1:11434", ollama_model="qwen2.5:3b", ollama_translation_model="qwen2.5:3b")
        translator, session, _ = self.run_mocked([response({"2": HEGEL_ORIGINAL}), response({"2": HEGEL_SPANISH})], settings=settings)
        self.assertEqual(session.post.call_count, 2)
        self.assertTrue(all(call.args[0].endswith("/api/chat") for call in session.post.call_args_list))
        self.assertEqual(translator.metrics["model_switch_count"], 0)

    def test_secondary_model_retry_stays_on_secondary_without_another_switch(self):
        settings = Settings(ollama_url="http://127.0.0.1:11434", ollama_model="qwen2.5:3b", ollama_translation_model="gemma3:4b")
        invalid = "Nunca admites que te equivocas. Hegel. Rewrote la realidad para que encajara con su teoría."
        translator, session, output = self.run_mocked([
            response({}), response({"2": HEGEL_ORIGINAL}), response({}),
            response({"2": invalid}), response({"2": HEGEL_SPANISH}),
        ], settings=settings)
        self.assertEqual(output["2"]["spanish"], HEGEL_SPANISH)
        self.assertEqual(session.post.call_count, 5)
        self.assertEqual(session.post.call_args_list[-1].kwargs["json"]["model"], "gemma3:4b")
        self.assertEqual(translator.metrics["retry_count"], 1)
        self.assertEqual(translator.metrics["model_switch_count"], 2)

    def test_failed_memory_unload_aborts_before_generating_with_second_model(self):
        import requests
        failure = Mock(status_code=500)
        failure.raise_for_status.side_effect = requests.HTTPError("unload failed")
        settings = Settings(ollama_url="http://127.0.0.1:11434", ollama_model="qwen2.5:3b", ollama_translation_model="gemma3:4b")
        translator = OllamaTranslator(settings)
        with patch("src.translation.ollama.requests.Session") as factory:
            session = factory.return_value.__enter__.return_value
            session.post.return_value = failure
            with self.assertRaisesRegex(RuntimeError, "memoria"):
                translator.translate({2: HEGEL_LINES})
        self.assertEqual(session.post.call_count, 1)
        self.assertTrue(session.post.call_args.args[0].endswith("/api/generate"))
        self.assertEqual(translator.metrics["request_count"], 0)
        self.assertFalse(translator.metrics["model_switches"][0]["completed"])

    def test_each_bounded_batch_releases_secondary_before_next_reconstruction(self):
        settings = Settings(ollama_url="http://127.0.0.1:11434", ollama_model="qwen2.5:3b", ollama_translation_model="gemma3:4b")
        original = "Socrates. They killed him for it."
        with patch("src.translation.ollama.MAX_BATCH_SLIDES", 1):
            translator, session, output = self.run_mocked([
                response({}), response({"2": HEGEL_ORIGINAL}), response({}), response({"2": HEGEL_SPANISH}),
                response({}), response({"5": original}), response({}), response({"5": "Sócrates. Lo mataron por ello."}),
            ], {2: HEGEL_LINES, 5: ["Socrates", "They killed him for it."]}, settings=settings)
        self.assertEqual(set(output), {"2", "5"})
        self.assertEqual(session.post.call_count, 8)
        self.assertEqual(translator.metrics["batches"], 2)
        self.assertEqual(translator.metrics["model_switch_count"], 4)

    def test_only_invalid_reconstruction_receives_an_individual_retry(self):
        socrates_lines = ["Socrates", "They killed him for it."]
        socrates_original = "Socrates. They killed him for it."
        replies = [response({"2": "You never admit you're wrong. Hegel.", "5": socrates_original}),
                   response({"2": HEGEL_ORIGINAL}),
                   response({"2": HEGEL_SPANISH, "5": "Sócrates. Lo mataron por ello."})]
        translator, session, output = self.run_mocked(replies, {2: HEGEL_LINES, 5: socrates_lines})
        self.assertEqual(set(output), {"2", "5"})
        self.assertEqual(session.post.call_count, 3)
        self.assertEqual(session.post.call_args_list[1].kwargs["json"]["format"]["required"], ["2"])
        self.assertEqual(translator.metrics["retry_count"], 1)

    def test_missing_batch_id_is_repaired_and_never_silently_omitted(self):
        translator, session, output = self.run_mocked([response({}), response({"2": HEGEL_ORIGINAL}), response({"2": HEGEL_SPANISH})])
        self.assertIn("2", output)
        self.assertEqual(translator.metrics["retry_count"], 1)

    def test_untranslated_verb_receives_one_spanish_only_retry(self):
        invalid = "Nunca admites que te equivocas. Hegel. Rewrote la realidad para que encajara con su teoría."
        translator, session, output = self.run_mocked([response({"2": HEGEL_ORIGINAL}), response({"2": invalid}), response({"2": HEGEL_SPANISH})])
        self.assertEqual(output["2"]["spanish"], HEGEL_SPANISH)
        self.assertEqual(translator.metrics["calls"][-1]["stage"], "translation")
        self.assertTrue(translator.metrics["calls"][-1]["retry"])
        self.assertEqual(session.post.call_count, 3)

    def test_pluralized_singular_caption_object_receives_existing_retry(self):
        lines = ["Socrates", "They killed him for it."]
        original, spanish = "Socrates. They killed him for it.", "Sócrates. Lo mataron por ello."
        translator, session, output = self.run_mocked([
            response({"5": original}), response({"5": "Sócrates. Los mataron por ello."}),
            response({"5": spanish}),
        ], {5: lines})
        self.assertEqual(output["5"]["spanish"], spanish)
        self.assertEqual(session.post.call_count, 3)
        self.assertEqual(translator.metrics["retry_count"], 1)
        last_request = session.post.call_args_list[-1].kwargs["json"]
        self.assertEqual(last_request["format"]["required"], ["5"])
        self.assertIn("objeto singular", last_request["messages"][-1]["content"])

    def test_first_person_named_caption_receives_one_translation_retry(self):
        lines = ["Nietzsche", "Lived it.", "Wrote about it.", "Changed nothing."]
        original = "Nietzsche.\nLived it. Wrote about it. Changed nothing."
        invalid = "Nietzsche.\nLo viví. Lo escribí. No cambié nada."
        valid = "Nietzsche.\nLo vivió. Escribió sobre ello. No cambió nada."
        settings = Settings(ollama_url="http://127.0.0.1:11434", ollama_model="qwen2.5:3b", ollama_translation_model="gemma3:4b")
        translator, session, output = self.run_mocked([
            response({}), response({"3": original}), response({}),
            response({"3": invalid}), response({"3": valid}),
        ], {3: lines}, settings=settings)
        self.assertEqual(output["3"]["spanish"], valid)
        self.assertEqual(session.post.call_count, 5)
        self.assertEqual(translator.metrics["retry_count"], 1)
        self.assertEqual(translator.metrics["model_switch_count"], 2)
        request = session.post.call_args_list[-1].kwargs["json"]
        self.assertEqual(request["model"], "gemma3:4b")
        self.assertEqual(request["format"]["required"], ["3"])
        self.assertIn("Nietzsche en tercera persona", request["messages"][-1]["content"])

    def test_second_invalid_result_preserves_english_after_bounded_retry(self):
        invalid = "You never admit you're wrong. Hegel."
        translator, session, output = self.run_mocked([response({"2": invalid}), response({"2": invalid})])
        self.assertEqual(session.post.call_count, 2)
        self.assertTrue(output['2']['fallback_to_english'])
        self.assertEqual(output['2']['spanish'], output['2']['original'])
        self.assertIn('He rewrote realty', output['2']['spanish'])
        self.assertFalse(output['2']['reconstruction']['audit']['accepted'])
        self.assertEqual(translator.metrics['fallback_slides'], ['2'])
        self.assertEqual(output['2']['failure']['code'], 'OUTPUT_VALIDATION_FAILED')
        self.assertFalse(output['2']['retry_recommended'])

    def test_generation_length_is_not_accepted_as_a_complete_result(self):
        _, session, output = self.run_mocked([response({"2": HEGEL_ORIGINAL}, "length"), response({"2": HEGEL_ORIGINAL}, "length")])
        self.assertEqual(session.post.call_count, 2)
        self.assertEqual(output['2']['translation']['status'], 'ENGLISH_FALLBACK')
        self.assertFalse(output['2']['translation']['integrity_validated'])
        self.assertTrue(output['2']['retry_recommended'])

    def test_low_confidence_fragment_is_preserved_while_reliable_context_is_translated(self):
        lines = [{'text': 'You are brave.', 'confidence': .97},
                 {'text': 'unreadable fragm3nt', 'confidence': .4}]
        original = 'You are brave.\n__KEEP_EN_0__'
        spanish = 'Eres valiente.\n__KEEP_EN_0__'
        _, session, output = self.run_mocked([response({'1': original}), response({'1': spanish})], {1: lines})
        item = output['1']
        self.assertEqual(item['spanish'], 'Eres valiente.\nunreadable fragm3nt')
        self.assertEqual(item['translation']['status'], 'PARTIAL_ENGLISH_FALLBACK')
        self.assertTrue(item['uncertain_fragments'][0]['preserved_in_english'])
        self.assertEqual(item['uncertain_fragments'][0]['ocr_confidence'], .4)
        request = session.post.call_args_list[0].kwargs['json']
        self.assertIn('"confidence": 0.4', request['messages'][-1]['content'])
        self.assertTrue(all('images' not in message for message in request['messages']))

    def test_dropping_protected_fragment_does_not_create_a_silent_success(self):
        lines = [{'text': 'You are brave.', 'confidence': .97}, {'text': 'unknown fragment', 'confidence': .4}]
        original = 'You are brave.\n__KEEP_EN_0__'
        _, session, output = self.run_mocked([
            response({'1': original}), response({'1': 'Eres valiente.'}), response({'1': 'Eres valiente.'}),
        ], {1: lines})
        self.assertEqual(session.post.call_count, 3)
        self.assertEqual(output['1']['spanish'], output['1']['original'])
        self.assertIn('unknown fragment', output['1']['spanish'])
        self.assertFalse(output['1']['translation']['integrity_validated'])

    def test_model_declared_semantic_uncertainty_keeps_only_the_exact_source_fragment(self):
        original = 'I saw her duck.\nThen I walked home.'
        spanish = '<KEEP_EN>I saw her duck.</KEEP_EN>\nDespués caminé a casa.'
        _, session, output = self.run_mocked([response({'1': original}), response({'1': spanish})], {
            1: [{'text': 'I saw her duck.', 'confidence': .99}, {'text': 'Then I walked home.', 'confidence': .98}],
        })
        result = output['1']
        self.assertEqual(session.post.call_count, 2)
        self.assertEqual(result['spanish'], 'I saw her duck.\nDespués caminé a casa.')
        self.assertEqual(result['translation']['status'], 'PARTIAL_ENGLISH_FALLBACK')
        self.assertEqual(result['uncertain_fragments'][0]['reason'], 'SEMANTIC_UNCERTAINTY_DECLARED')
        self.assertTrue(result['fallback_to_english'])
        self.assertEqual(result['confidence_level'], 'LOW')
        self.assertFalse(result['retry_recommended'])
        self.assertFalse(result['translation']['semantic_equivalence_verified'])

    def test_structured_source_uncertainty_preserves_exact_fragment_and_translates_body(self):
        original = 'I saw her duck.\nThen I walked home.'
        structured = {'uncertain_spans': [{'source': 'I saw her duck.', 'reason': 'AMBIGUOUS_SOURCE'}],
                      'spanish': 'I saw her duck.\nDespués caminé a casa.'}
        _, session, output = self.run_mocked([response({'1': original}), response({'1': structured})], {
            1: [{'text': 'I saw her duck.', 'confidence': .99}, {'text': 'Then I walked home.', 'confidence': .98}],
        })
        result = output['1']
        self.assertEqual(session.post.call_count, 2)
        self.assertEqual(result['spanish'], structured['spanish'])
        self.assertEqual(result['uncertain_fragments'][0]['uncertainty_kind'], 'AMBIGUOUS_SOURCE')
        self.assertEqual(result['reconstruction']['confidence_level'], 'HIGH')
        self.assertEqual(result['translation']['confidence_level'], 'LOW')
        self.assertIsNone(result['translation']['confidence'])

    def test_structured_uncertainty_cannot_substitute_an_unseen_source_phrase(self):
        original = 'I saw her duck.'
        invalid = {'uncertain_spans': [{'source': 'I saw her ship.', 'reason': 'AMBIGUOUS_SOURCE'}],
                   'spanish': 'I saw her ship.'}
        _, session, output = self.run_mocked([response({'1': original}), response({'1': invalid}), response({'1': invalid})], {1: [original]})
        self.assertEqual(session.post.call_count, 3)
        self.assertEqual(output['1']['spanish'], original)
        self.assertTrue(output['1']['fallback_to_english'])

    def test_declared_ambiguity_cannot_be_translated_instead_of_preserved(self):
        original = 'I saw her duck. Then I walked home.'
        invalid = {'uncertain_spans': [{'source': 'I saw her duck.', 'reason': 'AMBIGUOUS_SOURCE'}],
                   'spanish': 'Vi su pato. Después caminé a casa.'}
        _, session, output = self.run_mocked([response({'1': original}), response({'1': invalid}),
                                             response({'1': invalid})], {1: [original]})
        self.assertEqual(output['1']['spanish'], original)
        self.assertIn('COPIARSE literalmente', output['1']['warnings'][0])
        self.assertEqual(session.post.call_count, 3)

    def test_structured_output_budget_accounts_for_copied_source_and_target(self):
        # A 723-character, three-slide source exhausted the old 457-token cap.
        # The expanded schema must budget for assessment plus complete Spanish.
        values = {str(index): 'a' * 241 for index in range(3)}
        budget, characters = OllamaTranslator._output_token_budget('translation', values)
        reconstruction_budget, _ = OllamaTranslator._output_token_budget('reconstruction', values)
        self.assertEqual(characters, 723)
        self.assertGreater(budget, 723)
        self.assertGreater(budget, reconstruction_budget)
        self.assertLessEqual(budget, 2800)

    def test_translation_request_and_metrics_record_the_structured_budget(self):
        translator, session, _ = self.run_mocked([response({'2': HEGEL_ORIGINAL}), response({'2': HEGEL_SPANISH})])
        request = session.post.call_args_list[-1].kwargs['json']
        call = translator.metrics['calls'][-1]
        self.assertEqual(call['num_predict'], request['options']['num_predict'])
        self.assertEqual(call['num_ctx'], 4096)
        self.assertEqual(call['source_characters'], len(HEGEL_ORIGINAL))

    def test_high_confidence_quoted_unknown_token_is_protected_before_both_stages(self):
        masked = 'When someone says "__KEEP_EN_0__", I know the meaning.'
        spanish = 'Cuando alguien dice "__KEEP_EN_0__", conozco el significado.'
        translator, session, output = self.run_mocked([response({'1': {
            'uncertain_spans': [], 'spanish': spanish}})], {
            1: [{'text': 'When someone says "glorpz", I know the meaning.', 'confidence': .99}],
        })
        result = output['1']
        self.assertIn('"glorpz"', result['original'])
        self.assertIn('"glorpz"', result['spanish'])
        self.assertEqual(result['uncertain_fragments'][0]['reason'], 'QUOTED_TOKEN_PRESERVED')
        self.assertEqual(result['reconstruction']['confidence_level'], 'HIGH')
        self.assertEqual(session.post.call_count, 1)
        self.assertEqual(translator.metrics['literal_reconstruction_slides'], ['1'])
        for call in session.post.call_args_list:
            self.assertIn('__KEEP_EN_0__', call.kwargs['json']['messages'][-1]['content'])

    def test_redeclared_exact_protected_marker_is_idempotent(self):
        source = 'Someone said "glorpz" today.'
        spanish = 'Hoy alguien dijo "__KEEP_EN_0__".'
        invalid = {'uncertain_spans': [{'source': '__KEEP_EN_0__', 'reason': 'UNKNOWN_TERM'}],
                   'spanish': spanish}
        _, session, output = self.run_mocked([response({'1': invalid})], {1: [{'text': source, 'confidence': .99}]})
        self.assertEqual(output['1']['spanish'], 'Hoy alguien dijo "glorpz".')
        self.assertEqual(session.post.call_count, 1)
        assessment = output['1']['translation']['source_assessment']
        self.assertEqual(assessment['uncertain_spans'], [])
        self.assertEqual(assessment['already_protected_spans'][0]['source'], '__KEEP_EN_0__')
        self.assertTrue(assessment['already_protected_spans'][0]['already_protected'])
        self.assertEqual(len(output['1']['uncertain_fragments']), 1)

    def test_marker_redeclaration_cannot_invent_extend_change_or_omit_content(self):
        translator = OllamaTranslator(Settings())
        original = 'They said "__KEEP_EN_0__" and "__KEEP_EN_1__".'
        correct = 'Dijeron "__KEEP_EN_0__" y "__KEEP_EN_1__".'
        cases = [('__KEEP_EN_99__', correct),
                 ('"__KEEP_EN_0__"', correct),
                 ('__KEEP_EN_0__', correct.replace('__KEEP_EN_0__', '')),
                 ('__KEEP_EN_0__', correct.replace('__KEEP_EN_0__', '__KEEP_EP_0__')),
                 ('__KEEP_EN_0__', 'Dijeron "__KEEP_EN_1__" y "__KEEP_EN_0__".'),
                 ('__KEEP_EN_0__', correct + ' __KEEP_EN_0__')]
        for source, spanish in cases:
            with self.subTest(source=source, spanish=spanish), self.assertRaises(RuntimeError):
                translator._translation_content('1', [original], original, {
                    'uncertain_spans': [{'source': source, 'reason': 'UNKNOWN_TERM'}], 'spanish': spanish})

    def test_lexical_failure_receives_english_gloss_only_in_bounded_retry(self):
        source = 'Sleep on a choice until tomorrow.'
        invalid = 'Duerme sobre una elección hasta mañana.'
        accepted = 'Espera a mañana para decidir.'
        translator, session, output = self.run_mocked([response({'1': source}), response({'1': invalid}),
            response({'1': accepted})], {1: [source]})
        self.assertEqual(output['1']['spanish'], accepted)
        self.assertEqual(session.post.call_count, 3)
        first = session.post.call_args_list[1].kwargs['json']['messages'][-1]['content']
        retry = session.post.call_args_list[2].kwargs['json']['messages'][-1]['content']
        self.assertNotIn('DICTIONARY MEANINGS', first)
        self.assertIn('postponing a decision until the next day', retry)
        self.assertNotIn(accepted, retry)
        self.assertEqual(translator.metrics['calls'][-1]['retry_glosses'], ['SLEEP_ON_DECISION'])

    def test_lost_write_about_relation_receives_only_generic_english_retry_guidance(self):
        import json
        source, invalid, accepted = 'She wrote about the harbor.', 'Escribió el puerto.', 'Escribió sobre el puerto.'
        translator, session, output = self.run_mocked([response({'1': source}), response({'1': invalid}),
            response({'1': accepted})], {1: [source]})
        self.assertEqual(output['1']['spanish'], accepted)
        self.assertEqual(session.post.call_count, 3)
        initial = session.post.call_args_list[1].kwargs['json']['messages'][-1]['content']
        retry = session.post.call_args_list[2].kwargs['json']['messages'][-1]['content']
        self.assertNotIn('DICTIONARY MEANINGS', initial)
        self.assertIn('WRITE_ABOUT_TOPIC', retry)
        self.assertIn('whose subject is that topic', retry)
        self.assertNotIn(accepted, retry)
        self.assertEqual(json.loads(retry.split('\nDATA / DATOS:\n')[1]), {'1': source})
        self.assertEqual(translator.metrics['calls'][-1]['retry_glosses'], ['WRITE_ABOUT_TOPIC'])

    def test_write_direct_object_without_about_does_not_trigger_topic_retry_gloss(self):
        source, target = 'She wrote a letter.', 'Escribió una carta.'
        translator, session, output = self.run_mocked([response({'1': source}), response({'1': target})], {1: [source]})
        self.assertEqual(output['1']['spanish'], target)
        self.assertEqual(session.post.call_count, 2)
        self.assertEqual(translator.metrics['retry_count'], 0)
        self.assertFalse(any('WRITE_ABOUT_TOPIC' in call.kwargs['json']['messages'][-1]['content']
                             for call in session.post.call_args_list))

    def test_causative_naturalness_failure_uses_one_retry_with_english_guidance_outside_data(self):
        import json
        source, invalid, accepted = 'Force your team to wait.', 'Oblígate a que tu equipo espere.', 'Obliga a tu equipo a esperar.'
        translator, session, output = self.run_mocked([response({'1': source}), response({'1': invalid}),
            response({'1': accepted})], {1: [source]})
        self.assertEqual(output['1']['spanish'], accepted)
        self.assertEqual(session.post.call_count, 3)
        initial = session.post.call_args_list[1].kwargs['json']['messages'][-1]['content']
        retry = session.post.call_args_list[2].kwargs['json']['messages'][-1]['content']
        self.assertNotIn('DICTIONARY MEANINGS', initial)
        self.assertIn('idiomatic causative phrasing keeping pressure on that stated object', retry)
        self.assertNotIn(accepted, retry)
        self.assertEqual(json.loads(retry.split('\nDATA / DATOS:\n')[1]), {'1': source})
        self.assertEqual(translator.metrics['calls'][-1]['retry_glosses'], ['CAUSATIVE_REFLEXIVE_FORM'])
        self.assertFalse(output['1']['translation']['semantic_equivalence_verified'])

    def test_persistent_causative_naturalness_failure_returns_english_and_explicit_reason(self):
        source, invalid = 'Force your team to wait.', 'Oblígate a que tu equipo espere.'
        translator, session, output = self.run_mocked([response({'1': source}), response({'1': invalid}),
            response({'1': invalid})], {1: [source]})
        result = output['1']
        self.assertEqual(session.post.call_count, 3)
        self.assertEqual(translator.metrics['retry_count'], 1)
        self.assertEqual(result['spanish'], source)
        self.assertEqual(result['translation']['status'], 'ENGLISH_FALLBACK')
        self.assertIn('CAUSATIVE_REFLEXIVE_FORM', result['translation']['failure']['message'])
        self.assertIn('naturalidad', result['translation']['failure']['message'])
        self.assertFalse(result['translation']['semantic_equivalence_verified'])

    def test_ambiguous_comparative_heading_is_literal_while_body_is_translated(self):
        heading = '7. Show somebody stronger.'
        source = '__KEEP_EN_0__\nGive a concrete example.'
        spanish = '__KEEP_EN_0__\nDa un ejemplo concreto.'
        _, session, output = self.run_mocked([response({'1': source}), response({'1': {
            'uncertain_spans': [], 'spanish': spanish}})], {1: [
                {'text': heading, 'confidence': .99},
                {'text': 'Give a concrete example.', 'confidence': .98}]})
        result = output['1']
        self.assertEqual(result['spanish'], heading + '\nDa un ejemplo concreto.')
        self.assertEqual(result['translation']['status'], 'PARTIAL_ENGLISH_FALLBACK')
        self.assertEqual(result['uncertain_fragments'][0]['reason'], 'AMBIGUOUS_COMPARATIVE_ATTACHMENT')
        self.assertEqual(result['reconstruction']['confidence_level'], 'HIGH')
        self.assertEqual(result['translation']['confidence_level'], 'LOW')
        self.assertEqual(session.post.call_count, 2)

    def test_protected_heading_marker_cannot_receive_a_small_ocr_style_edit(self):
        original = '__KEEP_EN_0__\nGive a concrete example.'
        invalid = original.replace('KEEP_EN', 'KEEP_EP')
        _, session, output = self.run_mocked([response({'1': invalid}), response({'1': original}),
            response({'1': {'uncertain_spans': [], 'spanish': '__KEEP_EN_0__\nDa un ejemplo concreto.'}})], {
                1: [{'text': 'Show somebody stronger.', 'confidence': .99},
                    {'text': 'Give a concrete example.', 'confidence': .98}]})
        self.assertEqual(session.post.call_count, 3)
        self.assertIn('marcador literal protegido', session.post.call_args_list[1].kwargs['json']['messages'][-1]['content'])
        self.assertEqual(output['1']['spanish'], 'Show somebody stronger.\nDa un ejemplo concreto.')

    def test_bare_communication_comparatives_do_not_claim_low_ocr(self):
        for heading in ['Tell anyone better!', 'Explain anybody easier.', 'Teach somebody clearer.']:
            with self.subTest(heading=heading), patch('src.translation.ollama.requests.Session') as factory:
                translator = OllamaTranslator(Settings())
                result = translator.translate({1: [{'text': heading, 'confidence': .99}]})['1']
                factory.return_value.__enter__.return_value.post.assert_not_called()
                self.assertEqual(result['spanish'], heading)
                self.assertEqual(result['reconstruction']['confidence_level'], 'HIGH')
                self.assertIsNone(result['failure'])

    def test_communication_headings_with_explicit_content_are_not_protected(self):
        cases = [('Teach somebody simpler words.', 'Enseña a alguien palabras más sencillas.'),
                 ('Show anybody a clearer diagram.', 'Muestra a alguien un diagrama más claro.'),
                 ('Tell anyone shorter stories.', 'Cuenta a alguien historias más cortas.')]
        for source, spanish in cases:
            with self.subTest(source=source):
                _, _, output = self.run_mocked([response({'1': source}), response({'1': {
                    'uncertain_spans': [], 'spanish': spanish}})], {1: [{'text': source, 'confidence': .99}]})
                self.assertFalse(output['1']['fallback_to_english'])
                self.assertEqual(output['1']['uncertain_fragments'], [])

    def test_preposition_or_explicit_content_after_comparative_resolves_the_pattern(self):
        cases = [('Teach somebody in a simpler way.', 'Enseña a alguien de una forma más sencilla.'),
                 ('Show someone stronger evidence.', 'Muestra a alguien pruebas más sólidas.'),
                 ('Tell anyone about a clearer solution.', 'Habla a alguien de una solución más clara.')]
        for source, spanish in cases:
            with self.subTest(source=source):
                _, _, output = self.run_mocked([response({'1': source}), response({'1': {
                    'uncertain_spans': [], 'spanish': spanish}})], {1: [{'text': source, 'confidence': .99}]})
                self.assertFalse(output['1']['fallback_to_english'])
                self.assertEqual(output['1']['spanish'], spanish)

    def test_pure_quoted_literal_is_preserved_without_claiming_low_ocr(self):
        translator = OllamaTranslator(Settings())
        with patch('src.translation.ollama.requests.Session') as factory:
            result = translator.translate({1: [{'text': '"glorpz"', 'confidence': .99}]})['1']
        factory.return_value.__enter__.return_value.post.assert_not_called()
        self.assertEqual(result['spanish'], '"glorpz"')
        self.assertEqual(result['reconstruction']['confidence_level'], 'HIGH')
        self.assertEqual(result['translation']['status'], 'ENGLISH_FALLBACK')
        self.assertIsNone(result['failure'])

    def test_ocr_high_confidence_does_not_certify_spanish_adequacy(self):
        original = 'You are kind.'
        _, _, output = self.run_mocked([response({'1': original}), response({'1': {
            'uncertain_spans': [], 'spanish': 'Eres amable.'}})], {1: [{'text': original, 'confidence': .99}]})
        result = output['1']
        self.assertEqual(result['reconstruction']['confidence_level'], 'HIGH')
        self.assertEqual(result['translation']['confidence_level'], 'MEDIUM')
        self.assertEqual(result['translation']['quality_status'], 'UNVERIFIED')
        self.assertIsNone(result['translation']['confidence'])

    def test_bad_slide_does_not_prevent_other_slide_translation(self):
        good_lines = ['Socrates', 'They killed him for it.']
        original = 'Socrates. They killed him for it.'
        translator, session, output = self.run_mocked([
            response({'2': 'Hegel.', '5': original}), response({'2': 'Hegel.'}),
            response({'5': 'Sócrates. Lo mataron por ello.'}),
        ], {2: HEGEL_LINES, 5: good_lines})
        self.assertEqual(session.post.call_count, 3)
        self.assertTrue(output['2']['fallback_to_english'])
        self.assertFalse(output['5']['fallback_to_english'])
        self.assertEqual(output['5']['spanish'], 'Sócrates. Lo mataron por ello.')
        self.assertEqual(translator.metrics['fallback_slides'], ['2'])

    def test_all_uncertain_ocr_is_kept_without_a_model_request(self):
        translator = OllamaTranslator(Settings())
        with patch('src.translation.ollama.requests.Session') as factory:
            output = translator.translate({1: [{'text': 'unreadable word', 'confidence': .2}]})
        factory.return_value.__enter__.return_value.post.assert_not_called()
        self.assertEqual(output['1']['spanish'], 'unreadable word')
        self.assertEqual(output['1']['confidence_level'], 'LOW')

    def test_short_readable_greeting_cannot_be_accepted_as_empty(self):
        _, session, output = self.run_mocked([response({'1': ''}), response({'1': ''}),
                                             response({'1': ''}), response({'1': ''})], {
            1: [{'text': 'Hi', 'confidence': .99}],
        })
        self.assertEqual(session.post.call_count, 4)
        self.assertEqual(output['1']['spanish'], 'Hi')
        self.assertTrue(output['1']['fallback_to_english'])

    def test_disconnected_ollama_keeps_all_slides_and_marks_a_later_retry(self):
        import requests
        translator, session, output = self.run_mocked([requests.ConnectionError('service unavailable')], {
            1: ['You are brave.'], 2: ['You are kind.'],
        })
        self.assertEqual(session.post.call_count, 1)
        self.assertEqual(set(output), {'1', '2'})
        self.assertTrue(all(item['retry_recommended'] for item in output.values()))
        self.assertTrue(all(item['failure']['code'] == 'CONNECTION_ERROR' for item in output.values()))
        self.assertEqual(translator.metrics['transient_failure_slides'], ['1', '2'])

    def test_rejected_model_number_can_use_high_confidence_literal_ocr_for_spanish(self):
        source, rejected, target = '7. Observe two leaves.', '2. Observe two leaves.', '7. Observa dos hojas.'
        translator, session, output = self.run_mocked([
            response({'2': rejected}), response({'2': rejected}),
            response({'2': {'uncertain_spans': [], 'spanish': target}})],
            {2: [{'text': source, 'confidence': .99}]})
        result = output['2']
        import json
        data = json.loads(session.post.call_args_list[-1].kwargs['json']['messages'][-1]['content'].split('\nDATA / DATOS:\n')[1])
        self.assertEqual(data, {'2': source})
        self.assertEqual(result['original'], source)
        self.assertEqual(result['spanish'], target)
        self.assertEqual(result['reconstruction']['source_mode'], 'OCR_LITERAL_AFTER_MODEL_REJECTION')
        self.assertEqual(result['reconstruction']['failure']['code'], 'OUTPUT_VALIDATION_FAILED')
        self.assertFalse(result['reconstruction']['audit']['accepted'])
        self.assertTrue(result['reconstruction']['literal_source_audit']['accepted'])
        self.assertEqual(result['reconstruction']['source_ocr_confidence'], .99)
        self.assertIsNone(result['translation']['failure'])
        self.assertTrue(result['translation']['integrity_validated'])
        self.assertFalse(result['translation']['semantic_equivalence_verified'])
        self.assertEqual(result['translation']['quality_status'], 'UNVERIFIED')
        self.assertFalse(result['fallback_to_english'])
        self.assertFalse(result['retry_recommended'])
        self.assertTrue(result['warnings'])
        self.assertEqual(translator.metrics['literal_source_fallback_slides'], ['2'])
        self.assertEqual(session.post.call_count, 3)

    def test_literal_ocr_continuation_needs_known_high_confidence_on_each_unprotected_line(self):
        source, rejected = '7. Observe two leaves.', '2. Observe two leaves.'
        for confidence in [.899, None]:
            with self.subTest(confidence=confidence):
                translator, session, output = self.run_mocked([
                    response({'2': rejected}), response({'2': rejected})],
                    {2: [{'text': source, 'confidence': confidence}]})
                self.assertEqual(output['2']['spanish'], source)
                self.assertTrue(output['2']['fallback_to_english'])
                self.assertEqual(session.post.call_count, 2)
                self.assertEqual(translator.metrics['literal_source_fallback_slides'], [])

    def test_literal_ocr_continuation_preserves_low_confidence_markers_after_model_omission(self):
        source = '7. Observe two leaves.'
        translator, session, output = self.run_mocked([
            response({'2': source}), response({'2': source}),
            response({'2': {'uncertain_spans': [], 'spanish': '7. Observa dos hojas.\n__KEEP_EN_0__'}})],
            {2: [{'text': source, 'confidence': .99}, {'text': 'blimpor', 'confidence': .2}]})
        result = output['2']
        self.assertEqual(session.post.call_count, 3)
        self.assertIn('blimpor', result['original'])
        self.assertIn('blimpor', result['spanish'])
        self.assertNotIn('__KEEP_EN_', result['spanish'])
        self.assertEqual(result['translation']['status'], 'PARTIAL_ENGLISH_FALLBACK')
        self.assertTrue(result['translation']['integrity_validated'])
        self.assertEqual(result['reconstruction']['failure']['code'], 'OUTPUT_VALIDATION_FAILED')
        self.assertEqual(translator.metrics['literal_source_fallback_slides'], ['2'])

    def test_literal_ocr_continuation_still_rejects_spanish_numeric_drift(self):
        source, rejected, bad_target = '7. Observe two leaves.', '2. Observe two leaves.', '2. Observa dos hojas.'
        _, session, output = self.run_mocked([
            response({'2': rejected}), response({'2': rejected}),
            response({'2': bad_target}), response({'2': bad_target})],
            {2: [{'text': source, 'confidence': .99}]})
        result = output['2']
        self.assertEqual(result['spanish'], source)
        self.assertEqual(session.post.call_count, 4)
        self.assertEqual(result['failure_stage'], 'translation')
        self.assertIsNotNone(result['reconstruction']['failure'])
        self.assertIsNotNone(result['translation']['failure'])
        self.assertFalse(result['translation']['integrity_validated'])

    def test_declared_english_is_rendered_even_when_model_reconstruction_was_rejected(self):
        source = 'The label says blimpor.\nKeep the note.'
        _, _, output = self.run_mocked([
            response({'2': 'Keep the note.'}), response({'2': 'Keep the note.'}),
            response({'2': {'uncertain_spans': [{'source': 'blimpor', 'reason': 'UNKNOWN_TERM'}],
                            'spanish': 'La etiqueta dice blimpor.\nConserva la nota.'}})],
            {2: [{'text': line, 'confidence': .99} for line in source.splitlines()]})
        result = output['2']
        self.assertEqual(result['translation']['status'], 'PARTIAL_ENGLISH_FALLBACK')
        self.assertNotIn('<KEEP_EN>', result['spanish'])
        self.assertIn('blimpor', result['spanish'])
        self.assertIsNone(result['translation']['failure'])
        self.assertTrue(any(item.get('uncertainty_kind') == 'UNKNOWN_TERM' for item in result['uncertain_fragments']))

    def test_malformed_or_truncated_batch_cannot_become_content_only_after_retry(self):
        source, rejected = '7. Observe two leaves.', '2. Observe two leaves.'
        malformed = response({})
        malformed.json.return_value['message']['content'] = '{invalid JSON'
        initial_responses = [malformed, response({'2': rejected}, done_reason='length'), response({'2': {'wrong': source}})]
        for index, initial in enumerate(initial_responses):
            with self.subTest(initial=index):
                translator, session, output = self.run_mocked([initial, response({'2': rejected})],
                    {2: [{'text': source, 'confidence': .99}]})
                self.assertEqual(session.post.call_count, 2)
                self.assertEqual(output['2']['spanish'], source)
                self.assertFalse(output['2']['reconstruction']['failure']['content_validation_only'])
                self.assertEqual(translator.metrics['literal_source_fallback_slides'], [])
                if index < 2:
                    self.assertTrue(output['2']['retry_recommended'])

    def test_high_confidence_source_does_not_continue_after_service_failure(self):
        import requests
        for failure in [requests.ConnectionError('offline'), requests.Timeout('late')]:
            with self.subTest(failure=type(failure).__name__):
                _, session, output = self.run_mocked([failure],
                    {2: [{'text': '7. Observe two leaves.', 'confidence': .99}]})
                self.assertEqual(session.post.call_count, 1)
                self.assertTrue(output['2']['fallback_to_english'])
                self.assertTrue(output['2']['retry_recommended'])
                self.assertNotIn('source_mode', output['2']['reconstruction'])

    def test_invalid_json_retry_is_excluded_and_remains_retryable_for_cache(self):
        source, rejected = '7. Observe two leaves.', '2. Observe two leaves.'
        invalid = response({})
        invalid.json.return_value['message']['content'] = 'not JSON'
        _, session, output = self.run_mocked([response({'2': rejected}), invalid],
            {2: [{'text': source, 'confidence': .99}]})
        self.assertEqual(session.post.call_count, 2)
        self.assertEqual(output['2']['failure']['code'], 'INVALID_MODEL_RESPONSE')
        self.assertTrue(output['2']['retry_recommended'])

    def test_exhausted_original_budget_prevents_http_and_keeps_top_level_cache_retry(self):
        source, rejected = '7. Observe two leaves.', '2. Observe two leaves.'
        translator = OllamaTranslator(Settings(ollama_model='gemma3:4b', ollama_translation_model='', translation_budget_seconds=30))
        elapsed = [0.0]
        replies = iter([response({'2': rejected}), response({'2': rejected})])
        request_count = [0]

        def reply(*args, **kwargs):
            request_count[0] += 1
            result = next(replies)
            if request_count[0] == 2:
                elapsed[0] = 31.0
            return result

        with patch('src.translation.ollama.time.perf_counter', side_effect=lambda: elapsed[0]), \
                patch('src.translation.ollama.requests.Session') as factory:
            session = factory.return_value.__enter__.return_value
            session.post.side_effect = reply
            output = translator.translate({2: [{'text': source, 'confidence': .99}]})
        result = output['2']
        self.assertEqual(session.post.call_count, 2)
        self.assertEqual(result['spanish'], source)
        self.assertEqual(result['reconstruction']['failure']['code'], 'OUTPUT_VALIDATION_FAILED')
        self.assertEqual(result['translation']['failure']['code'], 'TIME_BUDGET_EXHAUSTED')
        self.assertEqual(result['failure']['code'], 'TIME_BUDGET_EXHAUSTED')
        self.assertTrue(result['retry_recommended'])
        self.assertTrue(translator.metrics['budget_exhausted'])
        from src.pipeline import translation_retry_due
        with patch('src.pipeline.time.time', return_value=100):
            self.assertTrue(translation_retry_due({'translations': output, 'translation_generated_at': 0}, translator.settings))

    def test_whole_source_declared_uncertain_is_full_english_not_partial(self):
        source = 'The label says blimpor.'
        _, _, output = self.run_mocked([response({'2': source}), response({'2': {
            'uncertain_spans': [{'source': source, 'reason': 'AMBIGUOUS_SOURCE'}], 'spanish': source}})],
            {2: [{'text': source, 'confidence': .99}]})
        result = output['2']
        self.assertEqual(result['spanish'], source)
        self.assertEqual(result['translation']['status'], 'ENGLISH_FALLBACK')
        self.assertTrue(result['fallback_to_english'])
        self.assertFalse(result['translation']['semantic_equivalence_verified'])

    def test_partial_declared_source_with_real_spanish_remains_partial(self):
        source = 'The label says blimpor.'
        _, _, output = self.run_mocked([response({'2': source}), response({'2': {
            'uncertain_spans': [{'source': 'blimpor', 'reason': 'UNKNOWN_TERM'}], 'spanish': 'La etiqueta dice blimpor.'}})],
            {2: [{'text': source, 'confidence': .99}]})
        self.assertEqual(output['2']['translation']['status'], 'PARTIAL_ENGLISH_FALLBACK')
        self.assertEqual(output['2']['spanish'], 'La etiqueta dice blimpor.')

    def test_empty_ocr_requires_no_model_request(self):
        translator = OllamaTranslator(Settings())
        with patch("src.translation.ollama.requests.Session") as factory:
            self.assertEqual(translator.translate({1: []}), {})
            factory.assert_not_called()


class TextFallbackChainTests(unittest.TestCase):
    def translator(self):
        from src.translation.text import TextTranslator
        return TextTranslator(Settings(ollama_url='http://localhost:11434'))

    def test_readable_low_confidence_ocr_is_translated_without_reconstruction(self):
        translator = self.translator()
        with patch.object(translator, '_local', return_value='Eres valiente.') as primary, \
                patch.object(translator, '_google') as google, patch.object(translator, '_ollama') as ollama:
            result = translator.translate({1: [{'text': 'You are brave.', 'confidence': .4}]})['1']
        primary.assert_called_once()
        google.assert_not_called()
        ollama.assert_not_called()
        self.assertEqual(result['spanish'], 'Eres valiente.')
        self.assertFalse(result['translation']['fallback_used'])
        self.assertEqual(result['reconstruction']['status'], 'OCR_MINIMAL_REFLOW')

    def test_empty_primary_goes_to_google_and_retains_real_engine(self):
        translator = self.translator()
        with patch.object(translator, '_local', return_value=''), \
                patch.object(translator, '_google', return_value='Eres valiente.'), \
                patch.object(translator, '_ollama') as ollama:
            result = translator.translate({1: ['You are brave.']})['1']
        ollama.assert_not_called()
        self.assertEqual(result['translation']['engine'], 'google_translate_web')
        self.assertTrue(result['translation']['fallback_used'])
        self.assertEqual(result['translation']['text'], 'Eres valiente.')

    def test_upstream_failures_still_try_ollama(self):
        translator = self.translator()
        with patch.object(translator, '_local', side_effect=ImportError('not installed')), \
                patch.object(translator, '_google', side_effect=RuntimeError('HTTP 429')), \
                patch.object(translator, '_ollama', return_value='Eres valiente.') as ollama:
            result = translator.translate({1: ['You are brave.']})['1']
        ollama.assert_called_once()
        self.assertTrue(result['translation']['engine'].startswith('ollama:'))
        self.assertEqual(result['spanish'], 'Eres valiente.')

    def test_all_engines_fail_never_returns_empty_translation_or_omits_slide(self):
        translator = self.translator()
        with patch.object(translator, '_local', side_effect=RuntimeError('local failed')), \
                patch.object(translator, '_google', side_effect=RuntimeError('google failed')), \
                patch.object(translator, '_ollama', side_effect=RuntimeError('ollama failed')):
            output = translator.translate({1: ['You are brave.'], 2: []})
        self.assertEqual(set(output), {'1', '2'})
        self.assertIn('[No se pudo traducir automáticamente]', output['1']['spanish'])
        self.assertIn('You are brave.', output['1']['spanish'])
        self.assertEqual(output['1']['translation']['status'], 'ENGLISH_FALLBACK')
        self.assertTrue(output['2']['spanish'])

    def test_unsupported_named_caption_goes_to_next_engine(self):
        translator = self.translator()
        lines = ['Nietzsche', 'Lived it.', 'Wrote about it.', 'Changed nothing.']
        with patch.object(translator, '_local', return_value='Nietzsche.\nLo viví. Lo escribí. No cambié nada.'), \
                patch.object(translator, '_google', return_value='Nietzsche.\nLo vivió. Escribió sobre ello. No cambió nada.'):
            result = translator.translate({3: lines})['3']
        self.assertEqual(result['translation']['engine'], 'google_translate_web')
        self.assertIn('Lo vivió.', result['spanish'])

    def test_ollama_request_is_text_only_deterministic_and_keeps_model_alive(self):
        translator = self.translator()
        session = Mock()
        session.post.return_value.json.return_value = {'done': True, 'message': {'content': 'Eres valiente.'}}
        self.assertEqual(translator._ollama('You are brave.', session), 'Eres valiente.')
        payload = session.post.call_args.kwargs['json']
        self.assertEqual(payload['options']['temperature'], 0)
        self.assertEqual(payload['keep_alive'], translator.settings.ollama_keep_alive)
        self.assertTrue(all('images' not in message for message in payload['messages']))

    def test_google_parses_all_segments_and_sends_complete_slide(self):
        translator = self.translator()
        session = Mock()
        session.get.return_value.json.return_value = [[['Eres valiente.', 'You are brave.', None],
                                                      ['\nNo estás solo.', 'You are not alone.', None]]]
        source = 'You are brave.\nYou are not alone.'
        self.assertEqual(translator._google(source, session), 'Eres valiente.\nNo estás solo.')
        self.assertEqual(session.get.call_args.kwargs['params']['q'], source)

    def test_named_artwork_label_can_stay_literal_while_rest_is_spanish(self):
        from src.translation.text import _validate_output
        lines = ['Which one are you?', 'ANAAN GIVING THE SWORD TO LUGh.']
        source = '\n'.join(lines)
        spanish = '¿Cuál eres tú?\nANAAN DANDO LA ESPADA A LUGh.'
        self.assertEqual(_validate_output('7', lines, source, spanish), spanish)

    def test_english_negation_is_not_exempted_as_an_uppercase_name(self):
        from src.translation.text import _validate_output
        with self.assertRaises(RuntimeError):
            _validate_output('1', ['NEVER say you are alone.'], 'NEVER say you are alone.', 'NEVER di que estás solo.')

    def test_only_rejected_ollama_content_receives_one_bounded_repair(self):
        translator = self.translator()
        lines = ['Nietzsche', 'Lived it.', 'Wrote about it.', 'Changed nothing.']
        with patch.object(translator, '_local', side_effect=ImportError('not installed')), \
                patch.object(translator, '_google', side_effect=RuntimeError('google failed')), \
                patch.object(translator, '_ollama', side_effect=[
                    'Nietzsche.\nLo vivió. Lo escribió. No cambió nada.',
                    'Nietzsche.\nLo vivió. Escribió sobre ello. No cambió nada.',
                ]) as ollama:
            result = translator.translate({3: lines})['3']
        self.assertEqual(ollama.call_count, 2)
        self.assertIn('WRITE_ABOUT_TOPIC', ollama.call_args.args[2])
        self.assertEqual(result['translation']['status'], 'TRANSLATED')
        self.assertEqual(translator.metrics['retry_count'], 1)

    def test_elliptical_caption_uses_the_detected_name_without_altering_ocr(self):
        translator = self.translator()
        lines = ['Clara', 'Wrote about music.', 'Changed nothing.']
        with patch.object(translator, '_local', return_value='Clara.\nClara escribió sobre música.\nNo cambió nada.') as primary:
            item = translator.translate({1: lines})['1']
        self.assertIn('Clara wrote about music.', primary.call_args.args[0])
        self.assertEqual(item['ocr_text'], '\n'.join(lines))
        self.assertNotIn('Clara wrote', item['original'])
        self.assertEqual(item['spanish'], 'Clara.\nEscribió sobre música.\nNo cambió nada.')

    def test_caption_context_does_not_replace_an_explicit_subject(self):
        from src.translation.text import _caption_context
        source = 'Clara.\nShe wrote about music.'
        self.assertEqual(_caption_context(source, ['Clara', 'She wrote about music.']), source)

    def test_missing_caption_sentence_tries_the_next_translator(self):
        translator = self.translator()
        lines = ['Clara', 'Lived it.', 'Wrote about it.', 'Changed nothing.']
        with patch.object(translator, '_local', return_value='Clara.\nLo vivió. No cambió nada.'), \
                patch.object(translator, '_google', return_value='Clara.\nLo vivió, escribió sobre ello, no cambió nada.'):
            item = translator.translate({1: lines})['1']
        self.assertEqual(item['translation']['engine'], 'google_translate_web')
        self.assertIn('escribió sobre ello', item['spanish'])

    def test_provider_timeouts_are_not_repeated_for_every_slide(self):
        import requests
        translator = self.translator()
        with patch.object(translator, '_local', side_effect=ImportError('local unavailable')) as local, \
                patch.object(translator, '_google', side_effect=requests.Timeout('Google timed out')) as google, \
                patch.object(translator, '_ollama', side_effect=requests.Timeout('Ollama timed out')) as ollama:
            results = translator.translate({index: ['You are brave.'] for index in range(1, 8)})
        self.assertEqual(len(results), 7)
        self.assertEqual((local.call_count, google.call_count, ollama.call_count), (1, 1, 1))
        for item in results.values():
            self.assertIn('[No se pudo traducir automáticamente]', item['spanish'])
            self.assertIn('You are brave.', item['spanish'])

    def test_exhausted_batch_budget_keeps_every_source_and_stops_new_requests(self):
        clock = [100.0]
        translator = self.translator()
        def slow_local(source, session):
            clock[0] += translator.settings.translation_budget_seconds + 1
            raise RuntimeError('local failed after using the time budget')
        with patch('src.translation.text.time.perf_counter', side_effect=lambda: clock[0]), \
                patch.object(translator, '_local', side_effect=slow_local) as local, \
                patch.object(translator, '_google') as google, patch.object(translator, '_ollama') as ollama:
            results = translator.translate({index: ['You are brave.'] for index in range(1, 8)})
        local.assert_called_once()
        google.assert_not_called()
        ollama.assert_not_called()
        self.assertTrue(translator.metrics['budget_exhausted'])
        self.assertEqual(len(results), 7)
        self.assertTrue(all('You are brave.' in item['spanish'] for item in results.values()))

    def test_http_timeout_is_capped_to_remaining_batch_budget(self):
        translator = self.translator()
        translator._deadline = 108
        session = Mock()
        session.post.return_value.json.return_value = {'done': True, 'message': {'content': 'Eres valiente.'}}
        with patch('src.translation.text.time.perf_counter', return_value=100):
            self.assertEqual(translator._ollama('You are brave.', session), 'Eres valiente.')
        self.assertEqual(session.post.call_args.kwargs['timeout'], (8, 8))

    def test_push_people_away_calque_tries_another_translator_without_rewriting_source(self):
        translator = self.translator()
        source = "You push everyone away then wonder why you're alone."
        with patch.object(translator, '_local', return_value='Empujas a todos y luego te preguntas por qué estás solo.'), \
                patch.object(translator, '_google', return_value='Alejas a todo el mundo y luego te preguntas por qué estás solo.') as google:
            item = translator.translate({1: [source]})['1']
        self.assertEqual(item['translation']['engine'], 'google_translate_web')
        self.assertEqual(item['ocr_text'], source)
        self.assertEqual(google.call_args.args[0], source)
        self.assertIn('Alejas a todo el mundo', item['spanish'])

    def test_literal_physical_object_push_away_translation_is_allowed(self):
        from src.translation.text import _validate_output
        source, spanish = 'You pushed the chair away.', 'Empujaste la silla.'
        self.assertEqual(_validate_output('1', [source], source, spanish), spanish)

    def test_quoted_push_people_away_phrase_is_not_treated_as_emotional_instruction(self):
        from src.translation.text import _validate_output
        source = 'He said "push someone away".'
        spanish = 'Dijo "empuja a alguien".'
        self.assertEqual(_validate_output('1', [source], source, spanish), spanish)

    def test_push_away_repairs_only_bad_paragraph_and_keeps_valid_named_caption(self):
        from src.translation.text import MODEL_NAME
        translator = self.translator()
        lines = ["You push everyone away then wonder why you're alone.", 'Nietzsche',
                 'Lived it.', 'Wrote about it.', 'Changed nothing.']
        primary = 'Empujas a todos y luego te preguntas por qué estás solo.\nNietzsche.\nLo vivió, escribió sobre ello, no cambió nada.'
        with patch.object(translator, '_local', return_value=primary), \
                patch.object(translator, '_google', return_value='Alejas a todos y luego te preguntas por qué estás solo.') as google:
            item = translator.translate({3: lines})['3']
        self.assertEqual(google.call_args.args[0], lines[0])
        self.assertIn('Lo vivió, escribió sobre ello, no cambió nada.', item['spanish'])
        self.assertEqual(item['translation']['engine'], MODEL_NAME + ' + google_translate_web')
        self.assertTrue(item['translation']['fallback_used'])
        self.assertEqual(item['translation']['paragraph_engines']['1'], 'google_translate_web')
        self.assertEqual(item['translation']['paragraph_engines']['3'], MODEL_NAME)
        self.assertEqual(translator.metrics['calls'][1]['source'], lines[0])

    def test_unaligned_primary_is_not_mixed_and_first_person_full_fallback_is_rejected(self):
        translator = self.translator()
        lines = ["You push everyone away then wonder why you're alone.", 'Nietzsche',
                 'Lived it.', 'Wrote about it.', 'Changed nothing.']
        primary = 'Empujas a todos y luego te preguntas por qué estás solo.\nNietzsche. Lo vivió, escribió sobre ello, no cambió nada.'
        bad_google = 'Alejas a todos y luego te preguntas por qué estás solo.\nNietzsche.\nLo vivió. Escribí sobre eso. No cambió nada.'
        good_ollama = 'Alejas a todos y luego te preguntas por qué estás solo.\nNietzsche.\nLo vivió. Escribió sobre ello. No cambió nada.'
        with patch.object(translator, '_local', return_value=primary), \
                patch.object(translator, '_google', return_value=bad_google) as google, \
                patch.object(translator, '_ollama', return_value=good_ollama):
            item = translator.translate({3: lines})['3']
        self.assertIn('Nietzsche lived it.', google.call_args.args[0])
        self.assertTrue(item['translation']['engine'].startswith('ollama:'))
        self.assertNotIn('paragraph_engines', item['translation'])
        self.assertIn('Escribió sobre ello.', item['spanish'])


if __name__ == "__main__":
    unittest.main()
