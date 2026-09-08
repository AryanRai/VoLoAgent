"""Physical sessions opt into clean planning state without new connections."""
from unittest.mock import Mock
import json
import numpy as np
import pytest
from vlm_orchestrator.proxy import fresh_goal_state
from vlm_orchestrator.strategies.base import SessionState, StrategyContext
from vlm_orchestrator.strategies.subgoal import SubgoalConfig, SubgoalStrategy
from vlm_orchestrator.vlm import PassthroughVLM


def test_fresh_state_drops_conversation_tools_and_observations():
    tool = Mock()
    old = SessionState(infer_count=40,episode_step=0,episode_id=8,
                       original_instruction='old',subgoals=['old active'],
                       grasp_tool_executor=tool,grasp_tool_active=True,
                       last_grasped_object_pc_world=np.ones((3,3)),
                       initial_image=np.ones((4,4,3)))
    old.supervisor_stop = {'reason':'old abort'}
    new = fresh_goal_state(old)
    tool.reset.assert_called_once()
    assert new.infer_count == 0 and new.subgoals == []
    assert new.original_instruction is None and new.initial_image is None
    assert new.last_grasped_object_pc_world is None
    assert not hasattr(new,'supervisor_stop')
    assert old.subgoals == ['old active']  # finalization can still inspect prior state


def test_fresh_goal_replans_same_text_with_new_scene_not_old_subgoal():
    strategy = SubgoalStrategy(StrategyContext(vlm=PassthroughVLM()),SubgoalConfig())
    strategy._vlm_call = Mock(side_effect=[json.dumps({'subgoals':['first'],'ordered':True}),
                                         json.dumps({'subgoals':['second'],'ordered':True})])
    first = SessionState(episode_id=1)
    obs = {'prompt':'same task','observation/exterior_image_1_left':np.zeros((8,8,3),np.uint8)}
    _,first = strategy.process(obs,first)
    first.infer_count = 3
    second = fresh_goal_state(first)
    second.episode_id = 2
    obs2 = dict(obs, **{'observation/exterior_image_1_left':np.ones((8,8,3),np.uint8)})
    _,second = strategy.process(obs2,second)
    assert strategy._vlm_call.call_count == 2
    assert second.subgoals == ['second']
    assert second.initial_image.min() == 1
    assert first.subgoals == ['first']


def test_tool_reset_failure_does_not_silently_continue():
    old = SessionState(grasp_tool_executor=Mock())
    old.grasp_tool_executor.reset.side_effect = RuntimeError('reset failed')
    with pytest.raises(RuntimeError): fresh_goal_state(old)


def test_proxy_fresh_marker_round_trip_same_prompt():
    import threading
    import time
    import websockets.sync.client as ws
    from test_proxy import start_mock_vla
    from vlm_orchestrator.proxy import OrchestratorProxy, ProxyConfig
    from vlm_orchestrator.utils import codec
    from vlm_orchestrator.strategies.passthrough import PassthroughStrategy
    observations = []
    class Tracking(PassthroughStrategy):
        def process(self, obs, state):
            observations.append((state.infer_count,list(state.subgoals)))
            state.subgoals = ['prior goal state']
            return obs,state
    server = start_mock_vla(19910)
    config = ProxyConfig(vla_host='127.0.0.1',vla_port=19910,host='127.0.0.1',port=19911,
                         vlm=PassthroughVLM(),
                         strategy=Tracking(StrategyContext(vlm=PassthroughVLM())))
    proxy = OrchestratorProxy(config)
    threading.Thread(target=proxy.serve_forever,daemon=True).start()
    time.sleep(.5)
    try:
        with ws.connect('ws://127.0.0.1:19911',compression=None,max_size=None) as client:
            client.recv()
            for marker in ('one','one','two'):
                client.send(codec.Packer().pack({'prompt':'same','__episode_id':marker,
                                                '__fresh_goal':True,'__step':0,
                                                'observation/exterior_image_1_left':np.zeros((8,8,3),np.uint8)}))
                response = codec.unpackb(client.recv())
                assert response['orchestrator_goal_id'] == marker
                assert '__fresh_goal' not in response['received_keys']
        assert observations[0] == (0,[])
        assert observations[1][1] == ['prior goal state']
        assert observations[2] == (0,[])
    finally:
        server.shutdown()
