"""Keep background print and detector noise as evidence, outside the overlay.

Decisions use geometry and observed character shape, never a list of post IDs,
authors or expected regression phrases. Small isolated text is retained when
there is no larger, coherent block against which to distinguish it.
"""
import math
import re
import statistics


def bounds(polygon):
    try:
        xs,ys=zip(*polygon)
        if len(xs)<3 or not all(math.isfinite(float(v)) for v in (*xs,*ys)): return None
        return min(xs),min(ys),max(xs),max(ys)
    except (TypeError,ValueError): return None


def unreliable_diagonal(line):
    """A weak short/repetitive reading aligned with a diagonal may be line art.

    Ordinary words and high-confidence short labels remain text evidence.
    Callers retain discarded readings and the reason in their diagnostics.
    """
    try:
        polygon=line.get('polygon',[])
        dx=polygon[1][0]-polygon[0][0]; dy=polygon[1][1]-polygon[0][1]
        angle=abs(math.degrees(math.atan2(dy,dx)))
        angle=min(angle,abs(180-angle))
        confidence=float(line.get('confidence',0))
        characters=[c.casefold() for c in str(line.get('text','')) if c.isalnum()]
        alphabetic=sum(c.isalpha() for c in characters)
        repeated=max((characters.count(c) for c in set(characters)),default=0)/max(1,len(characters))
        return ((angle>25 and confidence<.90 and (alphabetic<=3 or repeated>=.75))
                or (angle>15 and confidence<.75 and len(characters)>=6 and repeated>=.75))
    except (TypeError,ValueError,IndexError):
        return False


def editorial_lines(lines, shape):
    height,width=shape[:2]
    reliable=[]
    for line in lines:
        box=bounds(line.get('polygon',[]))
        if box and float(line.get('confidence',1)) >= .9 and len(str(line.get('text','')).strip()) >= 8:
            reliable.append((line,box))
    if len(reliable)<2:
        return list(lines), []
    typical=statistics.median(b[3]-b[1] for _,b in reliable)
    centre=statistics.median((b[0]+b[2])/2 for _,b in reliable)
    prose_top=min(b[1] for _,b in reliable)
    prose_bottom=max(b[3] for _,b in reliable)
    # A high-confidence isolated letter can be part of a background equation.
    # Confidence alone cannot distinguish that lettering from an overlay.
    equations=[]
    for line in lines:
        text=str(line.get('text','')).strip()
        box=bounds(line.get('polygon',[]))
        if box and float(line.get('confidence',1))<.9 and len(text)<=12 and re.search(r'[\^=+]',text):
            equations.append(box)
    kept,ignored=[],[]
    for line in lines:
        text=str(line.get('text','')).strip(); box=bounds(line.get('polygon',[]))
        reason=None
        if unreliable_diagonal(line):
            reason='LOW_CONFIDENCE_DIAGONAL_SYMBOLS'
        elif box:
            line_height=box[3]-box[1]; cx=(box[0]+box[2])/2
            if line_height < typical*.50 and box[3] >= height*.94 and len(text.split()) >= 5:
                reason='SMALL_BORDER_BACKGROUND_PRINT'
            elif float(line.get('confidence',1)) < .9 and len(text)<=12 and (
                    re.search(r'[\^=+]',text) or len(re.sub(r'[^A-Za-z]','',text)) <= 1):
                # A short low-confidence equation detached from the prose block.
                if abs(cx-centre)>width*.15 or re.search(r'[\^=]',text) or line_height<typical*.7:
                    reason='BACKGROUND_SYMBOLS'
            elif len(text)==1 and text.isalpha() and line_height<typical*.7:
                cy=(box[1]+box[3])/2
                detached=(prose_top-box[3]>typical*1.5 or box[1]-prose_bottom>typical*1.5)
                near_equation=any(b[0]-typical<=cx<=b[2]+typical and
                                  b[1]-typical*1.5<=cy<=b[3]+typical*1.5 for b in equations)
                if detached and near_equation:
                    reason='DETACHED_EQUATION_LETTER'
        if reason:
            ignored.append({**line,'reason':reason})
        else:
            kept.append(line)
    return kept,ignored


def overlay_polygons(ocr, shape):
    """Recognised overlay plus horizontal unrecognised detector evidence.

    Long diagonal, unreadable line detections in the Tesla coil were previously
    erased by TELEA and mistaken for text still present on the correct source.
    Detector polygons remain in the manifest; only the cleaning mask is filtered.
    """
    kept,ignored=editorial_lines(ocr.get('recognized_lines',[]),shape)
    excluded={tuple(map(tuple,l.get('polygon',[]))) for l in ignored}
    polygons=[]
    for polygon in ocr.get('detected_polygons',[]):
        key=tuple(map(tuple,polygon))
        if key in excluded: continue
        box=bounds(polygon)
        if not box: continue
        try:
            dx=polygon[1][0]-polygon[0][0]; dy=polygon[1][1]-polygon[0][1]
            angle=abs(math.degrees(math.atan2(dy,dx)))
            angle=min(angle,abs(180-angle))
        except (IndexError,TypeError): angle=0
        known=any(tuple(map(tuple,l.get('polygon',[])))==key for l in kept)
        if angle > 25 and not known: continue
        polygons.append(polygon)
    return polygons,kept,ignored
