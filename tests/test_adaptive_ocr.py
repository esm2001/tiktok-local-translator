"""Regressions for lost duplicate regions and bounded missing-text recovery."""
import copy
from pathlib import Path
import sys
from unittest.mock import Mock,patch

import numpy as np

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'translator'))
from src.config import Settings
from src.image.cleaning import write_image
from src.ocr.paddle import parse_results,ocr_image
from src.ocr.editorial import editorial_lines


def prediction(texts,polygons,scores):
    return [{'dt_polys':polygons,'rec_polys':polygons,'rec_texts':texts,'rec_scores':scores}]


def test_repeated_no_in_two_regions_survives_parser():
    regions=[[[10,10],[90,10],[90,30],[10,30]],[[10,70],[90,70],[90,90],[10,90]]]
    result=parse_results(prediction(['No','No'],regions,[.99,.99]))
    assert len(result['recognized_lines'])==2
    assert len(result['detected_polygons'])==2


def test_split_full_guide_is_ordered_left_to_right_despite_detector_y_jitter():
    right=[[120,20],[230,20],[230,50],[120,50]]
    left=[[10,23],[110,23],[110,53],[10,53]]
    result=parse_results(prediction(['Guide)','(Full'],[right,left],[.99,.99]))
    assert [l['text'] for l in result['recognized_lines']]==['(Full','Guide)']


def test_empty_primary_gets_one_fallback_and_retains_new_text_geometry(tmp_path):
    path=tmp_path/'real-pixels.png';write_image(path,np.zeros((160,320,3),dtype=np.uint8))
    region=[[20,50],[260,50],[260,90],[20,90]]
    engine=Mock();engine.predict.side_effect=[prediction([],[],[]),prediction(['Clearly visible text'],[region],[.99])]
    with patch('src.ocr.paddle.get_ocr',return_value=engine): result=ocr_image(path,Settings())
    assert engine.predict.call_count==2
    assert result['recognized_lines'][0]['text']=='Clearly visible text'
    assert result['detected_polygons']==[region]
    assert result['diagnostics']['fallback_used']
    assert result['diagnostics']['primary_regions']==0
    assert result['diagnostics']['detected_regions']==1
    assert not result['diagnostics']['unresolved_signals']


def test_reliable_short_word_fast_path_does_not_assume_text_is_missing(tmp_path):
    path=tmp_path/'pixels.png';write_image(path,np.zeros((160,320,3),dtype=np.uint8))
    region=[[20,50],[160,50],[160,90],[20,90]]
    engine=Mock();engine.predict.return_value=prediction(['No'],[region],[.99])
    with patch('src.ocr.paddle.get_ocr',return_value=engine): result=ocr_image(path,Settings())
    assert engine.predict.call_count==1
    assert result['recognized_lines'][0]['text']=='No'
    assert not result['diagnostics']['fallback_used']


def test_gray_failure_is_recorded_and_preserves_primary_evidence(tmp_path):
    path=tmp_path/'pixels.png';write_image(path,np.zeros((160,320,3),dtype=np.uint8))
    engine=Mock();engine.predict.side_effect=[prediction([],[],[]),RuntimeError('gray engine failed')]
    with patch('src.ocr.paddle.get_ocr',return_value=engine): result=ocr_image(path,Settings())
    assert result['diagnostics']['fallback_error']=='gray engine failed'
    assert result['diagnostics']['unresolved_signals']==['NO_READABLE_TEXT']
    assert result['recognized_lines']==[]


def test_weak_repetitive_coil_reading_does_not_trigger_extra_passes(tmp_path):
    path=tmp_path/'pixels.png';write_image(path,np.zeros((240,400,3),dtype=np.uint8))
    diagonal=[[180,80],[340,15],[355,45],[195,110]]
    caption1=[[30,140],[350,140],[350,170],[30,170]]
    caption2=[[30,185],[350,185],[350,215],[30,215]]
    engine=Mock();engine.predict.return_value=prediction(
        ['mmmmmmmmmmmmmmm','A clearly readable heading','Another readable caption'],
        [diagonal,caption1,caption2],[.545,.99,.99])
    with patch('src.ocr.paddle.get_ocr',return_value=engine): result=ocr_image(path,Settings())
    assert engine.predict.call_count==1
    assert len(result['recognized_lines'])==3  # Keep the original weak evidence.
    kept,ignored=editorial_lines(result['recognized_lines'],(240,400,3))
    assert len(kept)==2
    assert ignored[0]['text']=='mmmmmmmmmmmmmmm'
    assert ignored[0]['reason']=='LOW_CONFIDENCE_DIAGONAL_SYMBOLS'
    assert not result['diagnostics']['fallback_signals']


def test_confident_detached_letter_near_equation_stays_in_raw_background():
    def line(text, confidence, x, y, w, h):
        return {'text':text,'confidence':confidence,'polygon':[[x,y],[x+w,y],[x+w,y+h],[x,y+h]]}
    evidence=[line('R',.99,100,20,12,12),line('x^2=3',.65,90,45,70,28),
              line('A readable full heading',.99,40,130,240,30),
              line('Another readable sentence',.99,40,170,240,30)]
    kept,ignored=editorial_lines(evidence,(240,320,3))
    assert [entry['text'] for entry in kept]==['A readable full heading','Another readable sentence']
    assert next(entry for entry in ignored if entry['text']=='R')['reason']=='DETACHED_EQUATION_LETTER'
    assert evidence[0]['confidence']==.99  # Original evidence is unchanged.


def test_single_letter_without_nearby_equation_is_preserved():
    lines=[{'text':text,'confidence':.99,'polygon':polygon} for text,polygon in [
        ('I',[[100,20],[112,20],[112,32],[100,32]]),
        ('A readable full heading',[[40,130],[280,130],[280,160],[40,160]]),
        ('Another readable sentence',[[40,170],[280,170],[280,200],[40,200]])]]
    kept,ignored=editorial_lines(lines,(240,320,3))
    assert kept==lines
    assert not ignored


def test_full_size_letter_next_to_equation_is_preserved():
    lines=[{'text':text,'confidence':confidence,'polygon':polygon} for text,confidence,polygon in [
        ('I',.99,[[100,20],[115,20],[115,50],[100,50]]),
        ('x^2=3',.65,[[90,55],[160,55],[160,80],[90,80]]),
        ('A readable full heading',.99,[[40,130],[280,130],[280,160],[40,160]]),
        ('Another readable sentence',.99,[[40,170],[280,170],[280,200],[40,200]])]]
    kept,ignored=editorial_lines(lines,(240,320,3))
    assert any(entry['text']=='I' for entry in kept)


def test_isolated_letter_without_prose_block_is_preserved():
    lines=[{'text':'R','confidence':.99,'polygon':[[100,20],[112,20],[112,32],[100,32]]}]
    assert editorial_lines(lines,(240,320,3))==(lines,[])
