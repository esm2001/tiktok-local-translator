"""Measured low-confidence crop retry, with real pixels and mocked predictions."""

import copy
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'translator'))
from src.config import Settings
from src.ocr.paddle import OCR_REFINEMENT_THRESHOLD, OCR_REFINEMENT_VERSION, ocr_image, refine_recognized_lines

REAL_POST = Path(os.environ.get('TIKTOK_TEST_POST', str(PROJECT.parent / 'TikTokTraducidos' / '7659483315203804449')))
ORIGINAL = REAL_POST / 'original' / '07.jpg'
POLYGON = [[146, 776], [918, 780], [918, 901], [145, 897]]
CROP_POLYGON = [[18, 18], [780, 26], [778, 132], [16, 124]]


def prediction(text='Which one are you?', confidence=.953, polygons=None):
    polygons = polygons if polygons is not None else [CROP_POLYGON]
    texts = [text] if isinstance(text, str) else text
    scores = [confidence] * len(texts) if isinstance(confidence, (int, float)) else confidence
    return [{'dt_polys': polygons, 'rec_polys': polygons, 'rec_texts': texts, 'rec_scores': scores}]


@unittest.skipUnless(ORIGINAL.exists(), 'Local real slide 07 fixture is unavailable')
class NativeCropRefinementTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings()
        self.line = {'text': 'Which one are your', 'confidence': .899, 'polygon': copy.deepcopy(POLYGON)}
        self.engine = Mock()
        self.engine.predict.return_value = prediction()
        patcher = patch('src.ocr.paddle.get_ocr', return_value=self.engine)
        self.model = patcher.start()
        self.addCleanup(patcher.stop)

    def refine(self, lines=None):
        return refine_recognized_lines(ORIGINAL, [self.line] if lines is None else lines, self.settings)

    def test_measured_retry_corrects_text_and_keeps_cached_input_and_geometry(self):
        cached = copy.deepcopy(self.line)
        result = self.refine()
        self.assertEqual(result[0]['text'], 'Which one are you?')
        self.assertEqual(result[0]['confidence'], .953)
        self.assertEqual(result[0]['polygon'], POLYGON)
        self.assertEqual(self.line, cached)
        self.assertEqual(result[0]['refinement']['version'], OCR_REFINEMENT_VERSION)
        self.engine.predict.assert_called_once()
        crop = self.engine.predict.call_args.args[0]
        self.assertIsInstance(crop, np.ndarray)
        self.assertEqual(crop.shape, (145, 793, 3))

    def test_high_confidence_lines_are_unchanged_without_loading_model(self):
        line = {**self.line, 'confidence': OCR_REFINEMENT_THRESHOLD}
        self.assertEqual(self.refine([line]), [line])
        self.model.assert_not_called()
        self.engine.predict.assert_not_called()

    def test_confidence_gain_below_threshold_does_not_replace_reading(self):
        self.engine.predict.return_value = prediction(confidence=.905)
        self.assertEqual(self.refine(), [self.line])
        self.engine.predict.assert_called_once()

    def test_gain_exactly_point_zero_two_is_accepted(self):
        self.engine.predict.return_value = prediction(confidence=.919)
        self.assertEqual(self.refine()[0]['text'], 'Which one are you?')

    def test_unrelated_high_confidence_text_does_not_replace_reading(self):
        self.engine.predict.return_value = prediction('Download your free guide today!', .999)
        self.assertEqual(self.refine(), [self.line])

    def test_multiple_recognized_lines_do_not_replace_initial_line(self):
        self.engine.predict.return_value = prediction(
            ['Which one are you?', 'Unexpected extra text'], [.99, .99], [CROP_POLYGON, CROP_POLYGON]
        )
        self.assertEqual(self.refine(), [self.line])
        self.engine.predict.assert_called_once()

    def test_extra_unrecognized_detector_fragment_does_not_block_one_reliable_line(self):
        other_polygon = [[10, 2], [45, 2], [45, 12], [10, 12]]
        self.engine.predict.return_value = prediction(polygons=[CROP_POLYGON, other_polygon])
        result = self.refine()
        self.assertEqual(result[0]['text'], 'Which one are you?')
        self.assertEqual(result[0]['polygon'], POLYGON)

    def test_footer_below_point_nine_three_ignores_measured_unreadable_fragment(self):
        polygon = [[154, 1070], [601, 1088], [597, 1197], [150, 1179]]
        footer = {'text': 'differentlyo', 'confidence': .922, 'polygon': polygon}
        noise = [[189, 0], [365, 0], [365, 17], [189, 17]]
        target = [[23, 21], [448, 38], [443, 131], [18, 114]]
        self.engine.predict.return_value = prediction(['OO', 'differently.'], [.17, .952], [noise, target])
        result = self.refine([footer])
        self.assertEqual(result[0]['text'], 'differently.')
        self.assertEqual(result[0]['confidence'], .952)
        self.assertEqual(result[0]['polygon'], polygon)
        self.assertEqual(footer['text'], 'differentlyo')
        self.engine.predict.assert_called_once()
        self.assertEqual(self.engine.predict.call_args.args[0].shape, (147, 471, 3))

    def test_truncated_single_line_is_rejected_even_with_good_character_similarity(self):
        self.engine.predict.return_value = prediction('Which one are', .999)
        self.assertEqual(self.refine(), [self.line])

    def test_missing_malformed_or_out_of_image_polygons_do_not_load_model(self):
        for polygon in (None, [], [[1, 2]], [[float('nan'), 4]] * 4, [[-100, -100]] * 4):
            with self.subTest(polygon=polygon):
                line = {**self.line, 'polygon': polygon}
                self.assertEqual(self.refine([line]), [line])
        self.model.assert_not_called()

    def test_retry_failure_retains_initial_reading(self):
        self.engine.predict.side_effect = RuntimeError('Optional crop inference failed')
        with patch('src.ocr.paddle.logger') as logger:
            self.assertEqual(self.refine(), [self.line])
        logger.warning.assert_called_once()
        self.engine.predict.assert_called_once()

    def test_empty_crop_prediction_retains_initial_reading(self):
        self.engine.predict.return_value = []
        self.assertEqual(self.refine(), [self.line])

    def test_fresh_ocr_refines_after_full_prediction_without_changing_detection(self):
        full = prediction('Which one are your', .899, [POLYGON])
        self.engine.predict.side_effect = [full, prediction()]
        result = ocr_image(ORIGINAL, self.settings)
        self.assertEqual(result['detected_polygons'], [POLYGON])
        self.assertEqual(result['recognized_lines'][0]['polygon'], POLYGON)
        self.assertEqual(result['recognized_lines'][0]['text'], 'Which one are you?')
        self.assertEqual(self.engine.predict.call_count, 2)
        self.assertIsInstance(self.engine.predict.call_args_list[1].args[0], np.ndarray)


if __name__ == '__main__':
    unittest.main(verbosity=2)
