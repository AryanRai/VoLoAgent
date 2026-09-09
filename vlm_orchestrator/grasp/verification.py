"""Opt-in visual gates for recovery, never benchmark success predicates.

Uses the orchestrator's existing VLM callable; no separate model or transport.
All positive assessments remain fallible visual judgments, recorded for audit.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import uuid
import numpy as np
from vlm_orchestrator.vlm import encode_image_b64, parse_json


def target_queries(target):
    # Preserve the full relation for verification. Do not delete it from the
    # goal or assume the first same-colour instance is the intended object.
    noun = re.split(r'\s+(?:beside|next to|near|in front of|behind|between|under|above|below|to the (?:left|right) of)\s+',
                    target, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    return list(dict.fromkeys([noun, target.strip()]))[:2]


def image_content(image, label):
    return [{'type': 'text', 'text': label}, {'type': 'image_url', 'image_url': {
        'url': 'data:image/jpeg;base64,' + encode_image_b64(image)}}]


def assessment(call, system, content, required):
    try:
        raw = call(system, content)
        data = parse_json(raw)
        valid = (isinstance(data, dict) and all(data.get(k) is True for k in required)
                 and data.get('ambiguous') is False
                 and isinstance(data.get('evidence'), str) and len(data['evidence'].strip()) >= 10)
        return valid, {'accepted': valid, 'response': data}
    except Exception as error:
        return False, {'accepted': False, 'error': type(error).__name__}


def verify_mask(call, image, mask, target):
    image, mask = np.asarray(image), np.asarray(mask)
    if (image.ndim != 3 or image.shape[2] != 3 or mask.shape != image.shape[:2]
            or not np.isfinite(mask).all() or not np.isin(mask, [0, 1]).all() or not mask.any()):
        return False, {'accepted': False, 'error': 'invalid_or_empty_mask'}
    mask = mask.astype(bool)
    yy, xx = np.where(mask)
    overlay = image.copy()
    overlay[~mask] = (overlay[~mask] * .2).astype(overlay.dtype)
    crop = image[yy.min():yy.max()+1, xx.min():xx.max()+1].copy()
    system = '''Verify a robot recovery target BEFORE motion. You receive the
original scene, a mask spotlight (bright pixels selected; surrounding scene dim),
and its bounding crop. The selection must correspond to the requested object,
not a referenced container or neighbour. A high segmentation score is NOT
identity evidence. Check the full relational description and whether multiple
objects make identity ambiguous. Reject a mask selecting the destination container instead of the
requested movable object, a partial/occluded selection whose identity cannot be established,
or an inseparable mix of target and neighbour. Do not guess.
Return JSON: {"target_matches": bool, "mask_excludes_other_objects": bool,
"ambiguous": bool, "evidence": "specific visible evidence"}.'''
    content = [{'type': 'text', 'text': 'Intended target (including context): ' + target}]
    content += image_content(image, 'Original scene')
    content += image_content(overlay, 'Selected mask spotlight')
    content += image_content(crop, 'Bounding crop (may contain background)')
    accepted, audit = assessment(call, system, content, ('target_matches', 'mask_excludes_other_objects'))
    audit.update(mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),
                 image_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                 mask_pixels=int(mask.sum()), image_shape=list(image.shape))
    capture_root = os.environ.get('VOLO_RECOVERY_CAPTURE_DIR')
    if not capture_root and os.environ.get('RECORD_LOG_DIR'):
        capture_root = Path(os.environ['RECORD_LOG_DIR']) / 'recovery-masks'
    if capture_root:
        from vlm_orchestrator.policy_capture import save_capture
        audit['capture'] = save_capture(capture_root, uuid.uuid4().hex, 'mask', {
            'image': image, 'mask': mask, 'spotlight': overlay,
            'target': target, 'assessment': audit.copy()})
    return accepted, audit


def verified_segment(segment, call, image, target, record):
    """At most two perception calls, independent of model confidence scores."""
    for query in target_queries(target):
        try:
            mask, score, bbox = segment(image, query)
            accepted, audit = verify_mask(call, image, mask, target)
            audit.update(query=query, target=target, detector_score=float(score))
        except Exception as error:
            accepted, audit = False, {'accepted': False, 'query': query,
                                      'target': target, 'error': type(error).__name__}
        record(audit)
        if accepted:
            return mask, score, bbox
    raise RuntimeError('unresolved_recovery_target_identity')


def verify_grasp_outcome(call, before, after, wrist, target, proprio):
    system = '''Verify the outcome of a completed recovery trajectory. Trajectory
completion is NOT grasp success. Compare before/after scenes and the current
wrist image if supplied. Require the intended target visibly retained by the
gripper and lifted clear of its support, not just closed fingers, movement of
the bowl, proximity, or a plausible end-effector pose. If occluded or unclear,
return ambiguous=true. This is a grasp gate, not task or placement scoring.
Return JSON: {"correct_target_held": bool, "lifted_clear": bool,
"ambiguous": bool, "evidence": "specific visible evidence"}.'''
    content = [{'type': 'text', 'text': 'Target: ' + target + '\nRobot state: ' + json.dumps(proprio)}]
    content += image_content(before, 'Before recovery') + image_content(after, 'After recovery')
    if wrist is not None:
        content += image_content(wrist, 'Current wrist camera')
    return assessment(call, system, content, ('correct_target_held', 'lifted_clear'))
