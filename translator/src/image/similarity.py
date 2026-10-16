"""Visual ranking using OpenCV. Candidate pixels are never written or inpainted.

ORB/RANSAC improves the score when geometry aligns. Every readable image also
gets a coarse score, so weak geometry and low resolution cannot prevent recovery.
The accepted flag is diagnostic only; the downloader always selects the best.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
import threading

import cv2
import numpy as np

from .cleaning import read_image

_ransac_lock = threading.Lock()


def _phash(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    low = cv2.dct(gray.astype(np.float32))[:8, :8].ravel()[1:]
    return low > np.median(low)


@dataclass
class CandidateComparison:
    accepted: bool = False
    score: float = 0.0
    width: int = 0
    height: int = 0
    reason: str = "UNVERIFIED"
    metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _resize(image: np.ndarray, maximum: int = 1000) -> np.ndarray:
    scale = min(1.0, maximum / max(image.shape[:2]))
    if scale == 1.0:
        return image
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def _ssim_map(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = first.astype(np.float32) / 255.0
    second = second.astype(np.float32) / 255.0
    mean_a = cv2.GaussianBlur(first, (7, 7), 1.5)
    mean_b = cv2.GaussianBlur(second, (7, 7), 1.5)
    variance_a = np.maximum(cv2.GaussianBlur(first * first, (7, 7), 1.5) - mean_a * mean_a, 0)
    variance_b = np.maximum(cv2.GaussianBlur(second * second, (7, 7), 1.5) - mean_b * mean_b, 0)
    covariance = cv2.GaussianBlur(first * second, (7, 7), 1.5) - mean_a * mean_b
    return ((2 * mean_a * mean_b + 0.01**2) * (2 * covariance + 0.03**2)) / (
        (mean_a * mean_a + mean_b * mean_b + 0.01**2) * (variance_a + variance_b + 0.03**2)
    )


def _correlation(first: np.ndarray, second: np.ndarray, selected: np.ndarray) -> float:
    a = first[selected].astype(np.float64)
    b = second[selected].astype(np.float64)
    if a.size < 200 or a.std() < 2 or b.std() < 2:
        return 0.0
    return float(np.clip(np.corrcoef(a, b)[0, 1], 0, 1))


def _colour_normalise(reference: np.ndarray, aligned: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Fit only a small global colour correction outside OCR regions."""
    output = aligned.astype(np.float32)
    for channel in range(3):
        a = reference[:, :, channel][valid].astype(np.float32)
        b = aligned[:, :, channel][valid].astype(np.float32)
        if len(a) < 200:
            continue
        gain = float(np.clip(a.std() / max(float(b.std()), 1.0), 0.80, 1.25))
        offset = float(np.clip(a.mean() - gain * b.mean(), -24.0, 24.0))
        output[:, :, channel] = output[:, :, channel] * gain + offset
    return np.clip(output, 0, 255).astype(np.uint8)


def _coarse_score(reference: np.ndarray, candidate: np.ndarray, mask: np.ndarray) -> tuple[float, dict]:
    """Rank even unrelated/low-detail candidates without a match threshold."""
    width, height = 192, 192
    a = cv2.resize(reference, (width, height), interpolation=cv2.INTER_AREA)
    b = cv2.resize(candidate, (width, height), interpolation=cv2.INTER_AREA)
    excluded = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST) > 0
    valid = ~excluded
    if np.count_nonzero(valid) < 200:
        valid = np.ones((height, width), dtype=bool)
    gray_a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    correlation = _correlation(gray_a, gray_b, valid)
    ssim = float(np.clip(np.mean(_ssim_map(gray_a, gray_b)[valid]), 0, 1))
    phash_similarity = 1 - np.count_nonzero(_phash(reference) != _phash(candidate)) / 63
    hist_a = cv2.calcHist([a], [0, 1, 2], valid.astype(np.uint8), [8, 8, 8], [0, 256] * 3)
    hist_b = cv2.calcHist([b], [0, 1, 2], valid.astype(np.uint8), [8, 8, 8], [0, 256] * 3)
    cv2.normalize(hist_a, hist_a)
    cv2.normalize(hist_b, hist_b)
    colour = max(0., cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL))
    score = min(.79, .35 * correlation + .25 * ssim + .25 * phash_similarity + .15 * colour)
    return round(float(score), 4), {'coarse_correlation': round(correlation, 4),
        'coarse_ssim': round(ssim, 4), 'coarse_phash_similarity': round(float(phash_similarity), 4),
        'coarse_colour_similarity': round(colour, 4)}


def compare_candidate(
    original: Path,
    candidate: Path,
    mask: Path,
    min_dimension: int = 400,
    threshold: float = 0.84,
    clean: Path | None = None,
) -> dict:
    result = CandidateComparison()
    try:
        reference_full = read_image(original)
        candidate_full = read_image(candidate)
        text_mask_full = read_image(mask, cv2.IMREAD_GRAYSCALE)
    except (RuntimeError, OSError) as exc:
        result.reason = "UNREADABLE_IMAGE"
        result.metrics["error"] = str(exc)
        return result.to_dict()

    result.height, result.width = map(int, candidate_full.shape[:2])
    if text_mask_full.shape != reference_full.shape[:2]:
        result.reason = "MASK_SIZE_MISMATCH"
        return result.to_dict()
    coarse_reference = read_image(clean) if clean is not None and clean.is_file() else reference_full
    result.score, coarse_metrics = _coarse_score(coarse_reference, candidate_full, text_mask_full)
    result.metrics.update(coarse_metrics)
    result.metrics['thumbnail'] = min(result.width, result.height) < min_dimension

    reference = _resize(reference_full)
    candidate_image = _resize(candidate_full, 1400)
    result.metrics['phash_distance'] = int(np.count_nonzero(_phash(reference) != _phash(candidate_image)))
    height, width = reference.shape[:2]
    reference_scale = width / reference_full.shape[1]
    candidate_scale = candidate_image.shape[1] / candidate_full.shape[1]
    result.metrics.update(reference_scale=round(reference_scale, 8), candidate_scale=round(candidate_scale, 8))
    text_mask = cv2.resize(text_mask_full, (width, height), interpolation=cv2.INTER_NEAREST)
    # Exclude mask borders from feature matching and SSIM windows as well.
    excluded = cv2.dilate(text_mask, np.ones((9, 9), dtype=np.uint8)) > 0
    background_mask = (~excluded).astype(np.uint8) * 255
    gray_reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    gray_candidate = cv2.cvtColor(candidate_image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=6000, scaleFactor=1.2, nlevels=8, fastThreshold=10)
    keys_reference, descriptors_reference = orb.detectAndCompute(gray_reference, background_mask)
    keys_candidate, descriptors_candidate = orb.detectAndCompute(gray_candidate, None)
    if descriptors_reference is None or descriptors_candidate is None:
        result.reason = "INSUFFICIENT_FEATURES"
        return result.to_dict()

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(descriptors_candidate, descriptors_reference, k=2)
    matches = [a for pair in pairs if len(pair) == 2 for a, b in [pair] if a.distance < 0.75 * b.distance]
    # Repeated structures cannot count the same target point multiple times.
    unique = {}
    for match in matches:
        if match.trainIdx not in unique or match.distance < unique[match.trainIdx].distance:
            unique[match.trainIdx] = match
    matches = list(unique.values())
    result.metrics["good_matches"] = len(matches)
    if len(matches) < 20:
        result.reason = "INSUFFICIENT_MATCHES"
        return result.to_dict()

    source_points = np.float32([keys_candidate[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    target_points = np.float32([keys_reference[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    # Candidate order and previous matches must not alter acceptance at a boundary.
    with _ransac_lock:
        cv2.setRNGSeed(0)
        homography, inlier_mask = cv2.findHomography(source_points, target_points, cv2.RANSAC, 3.0)
    if homography is None or inlier_mask is None or not np.isfinite(homography).all():
        result.reason = "ALIGNMENT_FAILED"
        return result.to_dict()
    inliers = inlier_mask.ravel().astype(bool)
    count = int(inliers.sum())
    ratio = count / len(matches)
    hull_area = cv2.contourArea(cv2.convexHull(target_points[inliers])) if count >= 3 else 0.0
    spatial_coverage = float(hull_area / max(np.count_nonzero(~excluded), 1))
    result.metrics.update(inliers=count, inlier_ratio=round(ratio, 4), spatial_coverage=round(spatial_coverage, 4))
    if count < 20 or ratio < 0.45 or spatial_coverage < 0.06:
        result.reason = "WEAK_GEOMETRIC_EVIDENCE"
        return result.to_dict()

    projected = cv2.perspectiveTransform(source_points[inliers], homography)
    reprojection_error = float(np.median(np.linalg.norm(projected - target_points[inliers], axis=2)))
    if reprojection_error > 2.5:
        result.reason = "ALIGNMENT_IMPRECISE"
        return result.to_dict()
    aligned = cv2.warpPerspective(candidate_image, homography, (width, height))
    visibility = cv2.warpPerspective(
        np.full(candidate_image.shape[:2], 255, dtype=np.uint8), homography, (width, height), flags=cv2.INTER_NEAREST
    )
    visible = cv2.erode(visibility, np.ones((9, 9), dtype=np.uint8)) > 0
    valid = visible & ~excluded
    coverage = float(np.count_nonzero(valid) / max(np.count_nonzero(~excluded), 1))
    result.metrics.update(background_coverage=round(coverage, 4), reprojection_error=round(reprojection_error, 4))
    if coverage < 0.45 or np.count_nonzero(valid) < 20000:
        result.reason = "INSUFFICIENT_BACKGROUND_COVERAGE"
        return result.to_dict()

    corrected = _colour_normalise(reference, aligned, valid)
    # Slight blur discounts compression/resampling noise without erasing structure.
    reference_blur = cv2.GaussianBlur(reference, (3, 3), 0.7)
    candidate_blur = cv2.GaussianBlur(corrected, (3, 3), 0.7)
    gray_a = cv2.cvtColor(reference_blur, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(candidate_blur, cv2.COLOR_BGR2GRAY)
    ssim = float(np.clip(np.mean(_ssim_map(gray_a, gray_b)[valid]), 0, 1))
    correlation = _correlation(gray_a, gray_b, valid)
    difference = np.mean(np.abs(reference_blur.astype(np.float32) - candidate_blur.astype(np.float32)), axis=2) / 255.0
    mae = float(np.mean(difference[valid]))
    pixel_similarity = float(np.clip(1 - mae / 0.18, 0, 1))
    geometry = min(1.0, count / 80) * 0.5 + min(1.0, ratio / 0.8) * 0.5
    result.score = round(float(0.45 * ssim + 0.25 * correlation + 0.20 * pixel_similarity + 0.10 * geometry), 4)
    result.metrics.update(masked_ssim=round(ssim, 4), background_correlation=round(correlation, 4), normalised_mae=round(mae, 4))
    if ssim < 0.70 or correlation < 0.82 or mae > 0.10 or result.score < threshold:
        result.reason = "BACKGROUND_DIFFERS"
        return result.to_dict()

    masked_visible = visible & (text_mask > 0)
    masked_pixels = np.count_nonzero(masked_visible)
    if masked_pixels >= 500:
        inside_mae = float(np.mean(difference[masked_visible]))
        changed = float(np.mean(difference[masked_visible] > max(0.10, mae * 3)))
        mask_similarity = float(np.clip(np.mean(_ssim_map(gray_a, gray_b)[masked_visible]), 0, 1))
        result.metrics.update(
            mask_region_similarity=round(mask_similarity, 4), mask_region_mae=round(inside_mae, 4),
            mask_changed_fraction=round(changed, 4),
        )
        if changed < 0.08 or (mask_similarity > 0.80 and inside_mae < max(0.04, mae * 1.5)):
            result.reason = "ORIGINAL_TEXT_STILL_PRESENT"
            return result.to_dict()

        if clean is not None and clean.is_file():
            clean_image = _resize(read_image(clean))
            clean_blur = cv2.GaussianBlur(clean_image, (3, 3), 0.7)
            clean_gray = cv2.cvtColor(clean_blur, cv2.COLOR_BGR2GRAY)
            clean_similarity = float(np.clip(np.mean(_ssim_map(clean_gray, gray_b)[masked_visible]), 0, 1))
            clean_mae = float(np.mean(np.abs(clean_blur.astype(np.float32) - candidate_blur.astype(np.float32))[masked_visible]) / 255.0)
            result.metrics.update(clean_region_similarity=round(clean_similarity, 4), clean_region_mae=round(clean_mae, 4))
            if clean_similarity > 0.92 and clean_mae < 0.025:
                result.reason = "INPAINTED_FALLBACK_COPY"
                return result.to_dict()
    else:
        result.metrics["mask_region_check"] = "NO_OVERLAP_WITH_CROP"

    result.accepted = True
    result.reason = "MATCH"
    result.metrics["homography"] = [[round(float(v), 6) for v in row] for row in homography]
    to_full_reference = np.diag([1 / reference_scale, 1 / reference_scale, 1.0])
    from_full_candidate = np.diag([candidate_scale, candidate_scale, 1.0])
    full_homography = to_full_reference @ homography @ from_full_candidate
    result.metrics["homography_full_resolution"] = [[round(float(v), 8) for v in row] for row in full_homography]
    return result.to_dict()
