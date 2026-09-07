"""N1.7 opt-in protocol, strict action semantics, real ZMQ / stub model I/O."""
import asyncio
import threading

import msgpack
import msgpack_numpy as mnp
import numpy as np
import pytest
import zmq

from vlm_orchestrator.protocols.gr00t_n17 import Gr00tN17Connection, canonical_to_n17, pack, unpack


def observation():
    return {
        'observation/exterior_image_1_left_raw': np.full((40, 50, 3), 127, np.uint8),
        'observation/wrist_image_left_raw': np.full((60, 45, 3), 64, np.uint8),
        'observation/ee_frame': 'robot_base',
        'observation/ee_pos': np.array([.4, .1, .3], np.float32),
        'observation/ee_quat': np.array([1, 0, 0, 0], np.float32),
        'observation/joint_position': np.ones(7, np.float32),
        'observation/gripper_position': np.ones(1, np.float32),
        'prompt': 'Pick the orange.',
        'gt_state': {'private_to_orchestrator': True},
    }


def test_exact_n17_keys_frame_and_no_padding():
    native = canonical_to_n17(observation())
    assert len(native) == 6 and 'gt_state' not in native
    assert native['video.exterior_image_1_left'].shape == (1, 1, 180, 320, 3)
    assert native['video.exterior_image_1_left'].min() == 127
    assert native['video.wrist_image_left'].min() == 64
    np.testing.assert_allclose(native['state.eef_9d'][0, 0], [.4,.1,.3,0,0,-1,-1,0,0])


def test_current_server_wire_roundtrip_and_absolute_actions():
    ctx = zmq.Context()
    server = ctx.socket(zmq.REP)
    port = server.bind_to_random_port('tcp://127.0.0.1')
    captured = []
    def serve():
        request = msgpack.unpackb(server.recv(), object_hook=mnp.decode, raw=False)
        captured.append(request)
        joint = np.full((1, 16, 7), .25, np.float32)
        grip = np.full((1, 16, 1), .8, np.float32)
        server.send(msgpack.packb(({'action.joint_position': joint, 'action.gripper_position': grip}, {}), default=mnp.encode))
    worker = threading.Thread(target=serve, daemon=True)
    worker.start()
    async def run():
        connection = Gr00tN17Connection('127.0.0.1', port, timeout_ms=2000)
        try:
            assert (await connection.recv_metadata())['backend'] == 'gr00t_n17_droid'
            return await connection.infer(observation())
        finally:
            await connection.close()
    try:
        result = asyncio.run(run())
        worker.join(timeout=3)
        assert not worker.is_alive()
        np.testing.assert_allclose(result['actions'][:, :7], .25)  # never + current joints
        assert result['actions'].shape == (16, 8)
        assert result['policy_timing']['round_trip_ms'] >= 0
        assert captured[0]['data']['observation']['annotation.language.language_instruction'] == ['Pick the orange.']
    finally:
        server.close(linger=0)
        ctx.term()


def test_numpy_codec_refuses_object_arrays():
    with pytest.raises(TypeError, match='Object arrays'):
        pack(np.array([object()], dtype=object))
    payload = msgpack.packb({b'nd': True, b'kind': b'O', b'data': b'not pickle'})
    with pytest.raises(ValueError, match='Object arrays'):
        unpack(payload)


def test_n17_socket_has_bounded_wait():
    ctx = zmq.Context()
    silent = ctx.socket(zmq.REP)
    port = silent.bind_to_random_port('tcp://127.0.0.1')
    async def run():
        connection = Gr00tN17Connection('127.0.0.1', port, timeout_ms=100)
        try:
            with pytest.raises(zmq.Again):
                await connection.infer(observation())
        finally:
            await connection.close()
    try:
        asyncio.run(run())
    finally:
        silent.close(linger=0)
        ctx.term()
