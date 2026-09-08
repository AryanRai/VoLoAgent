"""N1.7 DROID backend for the official GR00T sim-policy wrapper.

Explicit opt-in; legacy gr00t-zmq and OpenPI behaviour are unchanged.
The server processor undoes normalization and relative actions. Returned
joint positions are absolute; this adapter must NOT add current joints again.
"""
from __future__ import annotations

import asyncio
import functools
import os
import threading
import time

import cv2
import msgpack
import msgpack_numpy as mnp
import numpy as np
import zmq

from .base import Backend, BackendConnection
from .gr00t_zmq import compute_eef_9d

MODEL = "nvidia/GR00T-N1.7-DROID"
MODEL_REVISION = "05e7cc97e40dbd33b0890c35cc0214fcb0547ab5"
EMBODIMENT = "OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT"


def _encode(value, chain=None):
    if isinstance(value, np.ndarray) and value.dtype.hasobject:
        raise TypeError("Object arrays are not valid GR00T observations")
    encoded = mnp.encode(value, chain=chain)
    if isinstance(value, (np.ndarray, np.generic)) and isinstance(encoded, dict):
        # Explicit bytes also work with older msgpack's Python fallback,
        # whose memoryview length handling can truncate multidimensional data.
        encoded[b'data'] = value.tobytes()
    return encoded


def _decode(value, chain=None):
    if isinstance(value, dict):
        if value.get(b"nd", value.get("nd")) and value.get(b"kind", value.get("kind")) in (b"O", "O"):
            raise ValueError("Object arrays are not valid GR00T responses")
    return mnp.decode(value, chain=chain)


def pack(value):
    return msgpack.packb(value, default=functools.partial(_encode, chain=lambda x: x))


def unpack(value):
    return msgpack.unpackb(value, object_hook=functools.partial(_decode, chain=lambda x: x), raw=False)


def _vector(value, size, name):
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape ({size},), got {result.shape}")
    return result


def canonical_to_n17(obs):
    """Pinned RoboLab base-link pose and RAW cameras -> N1.7 native request."""
    if obs.get("observation/ee_frame") != "robot_base":
        raise ValueError("GR00T N1.7 requires an explicitly robot_base EEF pose")
    images = {}
    for camera in ("exterior_image_1_left", "wrist_image_left"):
        raw = obs.get(f"observation/{camera}_raw")
        # VoLo strips *_raw before calling its backend. The N1.7 client
        # already supplies native 180x320 images on the standard keys.
        image = np.asarray(raw if raw is not None else obs[f"observation/{camera}"])
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"{camera} must be raw HWC uint8 RGB")
        if raw is None and image.shape != (180, 320, 3):
            raise ValueError(f'{camera} needs native 180x320 input; padded Pi05 images are not supported')
        # Match pinned RoboLab resize_no_pad: INTER_AREA, no letterboxing.
        image = cv2.resize(image, (320, 180), interpolation=cv2.INTER_AREA)
        images[f"video.{camera}"] = image[None, None]
    xyz = _vector(obs["observation/ee_pos"], 3, "EEF position")
    quat = _vector(obs["observation/ee_quat"], 4, "EEF quaternion")
    if np.linalg.norm(quat) == 0:
        raise ValueError("EEF quaternion must be nonzero")
    goal = obs["prompt"]
    if not isinstance(goal, str):
        raise TypeError("GR00T instruction must be a string")
    return dict(images, **{
        "state.eef_9d": compute_eef_9d(xyz, quat)[None, None],
        "state.joint_position": _vector(obs["observation/joint_position"], 7, "joints")[None, None],
        "state.gripper_position": _vector(obs["observation/gripper_position"], 1, "gripper")[None, None],
        "annotation.language.language_instruction": [goal],
    })


def n17_action_to_canonical(action):
    """Official wrapper's already-decoded absolute joints/gripper -> commands."""
    joint = np.asarray(action["action.joint_position"], dtype=np.float32)
    grip = np.asarray(action["action.gripper_position"], dtype=np.float32)
    if joint.ndim != 3 or joint.shape[0] != 1 or joint.shape[1] < 1 or joint.shape[2] != 7:
        raise ValueError(f"Expected one DROID joint chunk [1,T,7], got {joint.shape}")
    if grip.shape != (*joint.shape[:2], 1):
        raise ValueError(f"Gripper chunk mismatch: {grip.shape}")
    chunk = np.concatenate([joint[0], grip[0]], axis=-1)
    if not np.isfinite(chunk).all():
        raise ValueError("GR00T returned non-finite actions")
    return {"actions": chunk}


class Gr00tN17Connection(BackendConnection):
    def __init__(self, host, port, api_token=None, timeout_ms=180000):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(f"tcp://{host}:{port}")
        self._api_token = api_token
        self._lock = threading.Lock()

    async def recv_metadata(self):
        return {"backend": "gr00t_n17_droid", "model": MODEL,
                "configured_revision": MODEL_REVISION, "embodiment": EMBODIMENT}

    async def infer(self, canonical_obs):
        request = {"endpoint": "get_action", "data": {
            "observation": canonical_to_n17(canonical_obs), "options": None}}
        if self._api_token:
            request["api_token"] = self._api_token

        if os.environ.get('POLICY_CAPTURE_DIR'):
            from vlm_orchestrator.policy_capture import save_capture
            save_capture(os.environ['POLICY_CAPTURE_DIR'],canonical_obs['__capture_id'],
                         'request',{'native_request':request['data']['observation']})

        def round_trip():
            with self._lock:
                self._socket.send(pack(request))
                return self._socket.recv()

        start = time.perf_counter()
        response_bytes = await asyncio.to_thread(round_trip)
        elapsed_ms = (time.perf_counter() - start) * 1000
        if response_bytes == b"ERROR":
            raise RuntimeError("GR00T N1.7 server error; inspect its log")
        response = unpack(response_bytes)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"GR00T N1.7: {response['error']}")
        if not isinstance(response, (list, tuple)) or len(response) != 2:
            raise ValueError("Expected GR00T (actions, info) response")
        result = n17_action_to_canonical(response[0])
        capture_root = os.environ.get('POLICY_CAPTURE_DIR')
        if capture_root:
            from pathlib import Path
            from vlm_orchestrator.policy_capture import save_capture, source_revision
            capture_id = canonical_obs.get('__capture_id')
            if not capture_id:
                raise ValueError('Capture enabled but client supplied no capture_id')
            result['policy_capture'] = save_capture(capture_root, capture_id, 'native', {
                'canonical': canonical_obs, 'native_request': request['data']['observation'],
                'native_response': response, 'decoded_chunk': result['actions'],
                'model': MODEL, 'revision': MODEL_REVISION, 'embodiment': EMBODIMENT,
                'volo_revision': source_revision(Path(__file__).parent)})
        result["policy_timing"] = {"round_trip_ms": elapsed_ms}
        return result

    async def close(self):
        self._socket.close(linger=0)
        self._context.term()


class Gr00tN17Backend(Backend):
    def __init__(self, host, port, api_token=None):
        self._host, self._port, self._api_token = host, port, api_token

    async def connect(self):
        return Gr00tN17Connection(self._host, self._port, self._api_token)
