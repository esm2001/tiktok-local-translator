"""Export invariants: web bytes stay RAW, and every slide has a TXT section."""
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'translator'))
from src.output.manifest import write_translations_txt
from src.pipeline import validate_outputs, reusable_source, pipeline_signature
from src.config import Settings


class OutputContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        for folder in ('original', 'clean', 'recovered'):
            (self.base / folder).mkdir()

    def slide(self, index, recovered=True):
        original = f'original/{index:02d}.jpg'
        clean = f'clean/{index:02d}_CLEAN.jpg'
        source = f'recovered/{index:02d}_SOURCE.png'
        Image.new('RGB', (80, 100), (index * 20, 70, 90)).save(self.base / original)
        Image.new('RGB', (80, 100), (index * 20, 70, 90)).save(self.base / clean)
        if recovered:
            Image.new('RGB', (80, 100), (index * 20, 100, 110)).save(self.base / source)
        digest = lambda path: hashlib.sha256((self.base / path).read_bytes()).hexdigest()
        return {'position': index, 'original': original, 'clean': clean,
                'original_sha256': digest(original), 'clean_sha256': digest(clean),
                'ocr': {'mask_sha256': 'intact'},
                'reverse_search': {'downloaded': source if recovered else None,
                    'status': 'MATCH' if recovered else 'NOT_FOUND',
                    'sha256': digest(source) if recovered else None,
                    'source': 'web_download', 'raw_download': True, 'modified': False,
                    'source_url': 'https://example.org/photo.png', 'score': .03,
                    'resolution': {'width': 80, 'height': 100}}}

    def test_complete_checks_all_seven_slides_and_preserves_png_bytes(self):
        slides = [self.slide(index) for index in range(1, 8)]
        before = {p.name: p.read_bytes() for p in (self.base / 'recovered').iterdir()}
        write_translations_txt(self.base / 'traducciones.txt', 7,
                               {str(i): {'original': 'You are alone.', 'spanish': 'Estás solo.'} for i in range(1, 8)})
        result = validate_outputs(self.base, slides)
        self.assertTrue(result['complete'], result)
        self.assertEqual([result[k] for k in ('original_count', 'clean_count', 'recovered_count', 'translation_sections')], [7] * 4)
        self.assertEqual(before, {p.name: p.read_bytes() for p in (self.base / 'recovered').iterdir()})

    def test_missing_source_is_incomplete_and_identifies_slide(self):
        slides = [self.slide(1), self.slide(2, recovered=False)]
        write_translations_txt(self.base / 'traducciones.txt', 2,
                               {'1': {'spanish': 'Hola'}, '2': {'spanish': 'Adiós'}})
        result = validate_outputs(self.base, slides)
        self.assertFalse(result['complete'])
        self.assertTrue(any('Diapositiva 2: falta recovered' in message for message in result['missing']))

    def test_empty_translation_preserves_original_with_explicit_marker(self):
        slide = self.slide(1)
        write_translations_txt(self.base / 'traducciones.txt', 1,
                               {'1': {'original': 'You push everyone away.', 'spanish': ''}})
        output = (self.base / 'traducciones.txt').read_text(encoding='utf-8')
        self.assertIn('TRADUCCIÓN:\n[No se pudo traducir automáticamente]\nYou push everyone away.', output)
        self.assertTrue(validate_outputs(self.base, [slide])['complete'])

    def test_low_score_small_raw_png_is_reused_without_threshold(self):
        slide = self.slide(1)
        settings = Settings(output_base=self.base)
        signature = pipeline_signature(settings)
        cached = reusable_source(self.base, slide, slide, signature, signature, settings)
        self.assertIsNotNone(cached)
        self.assertTrue(cached['cache_hit'])
        self.assertEqual(cached['downloaded'], 'recovered/01_SOURCE.png')

    def test_clean_without_raw_provenance_cannot_be_reused_as_recovered(self):
        slide = self.slide(1)
        slide['reverse_search']['raw_download'] = False
        settings = Settings(output_base=self.base)
        signature = pipeline_signature(settings)
        self.assertIsNone(reusable_source(self.base, slide, slide, signature, signature, settings))


if __name__ == '__main__':
    unittest.main()
