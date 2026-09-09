import json
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pytest
from vlm_orchestrator.grasp.verification import verified_segment, verify_mask, verify_grasp_outcome, target_queries
from vlm_orchestrator.grasp.tool import GraspToolExecutor, GraspPhase, GraspSegMode


def verdict(ok=True):
    return json.dumps(dict(target_matches=ok, mask_excludes_other_objects=ok,
        ambiguous=not ok, evidence='The selected region contains the wooden bowl, not the requested target.' if not ok
        else 'The selected region is the purple target and excludes the adjacent container.'))


def test_reject_high_score_wrong_mask_retry_then_accept_low_score_without_mutation():
    image=np.arange(12*16*3,dtype=np.uint8).reshape(12,16,3)
    wrong=np.zeros((12,16),bool);wrong[2:10,5:14]=True
    correct=np.zeros_like(wrong);correct[7:9,1:3]=True
    originals=[a.copy() for a in (image,wrong,correct)]
    segment=Mock(side_effect=[(wrong,.96,(5,2,14,10)),(correct,.39,(1,7,3,9))])
    call=Mock(side_effect=[verdict(False),verdict(True)]); records=[]
    mask,score,_=verified_segment(segment,call,image,'purple onion beside the wooden bowl',records.append)
    assert score==.39 and np.array_equal(mask,correct)
    assert [c.args[1] for c in segment.call_args_list]==['purple onion','purple onion beside the wooden bowl']
    assert [r['accepted'] for r in records]==[False,True]
    assert 'beside the wooden bowl' in str(call.call_args_list[0])
    for a,b in zip((image,wrong,correct),originals):np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize('raw',[verdict(False),'{}','not json','{"target_matches": "true"}',
                               json.dumps(dict(target_matches=True,mask_excludes_other_objects=True,ambiguous=False,evidence=''))])
def test_unknown_or_rejected_masks_never_succeed(raw):
    call=Mock(return_value=raw); segment=Mock(return_value=(np.ones((8,8),bool),.99,(0,0,8,8)))
    with pytest.raises(RuntimeError,match='unresolved_recovery_target_identity'):
        verified_segment(segment,call,np.zeros((8,8,3),np.uint8),'object near container',Mock())
    assert segment.call_count==2


@pytest.mark.parametrize('mask',[np.zeros((8,8)), np.ones((4,4)), np.full((8,8),np.nan),np.ones((8,8))*.3])
def test_invalid_mask_does_not_reach_verifier(mask):
    call=Mock()
    assert not verify_mask(call,np.zeros((8,8,3),np.uint8),mask,'object')[0]
    call.assert_not_called()


def test_preflight_rejection_occurs_before_release_or_lift(monkeypatch):
    monkeypatch.setenv('VOLO_VERIFIED_RECOVERY','1');monkeypatch.setenv('VOLO_STALL_SUPERVISOR','1')
    executor=object.__new__(GraspToolExecutor)
    executor._verification_call=Mock(return_value=verdict(False))
    executor._seg_mode=GraspSegMode.SAM3
    executor._extract_image=Mock(return_value=np.zeros((8,8,3),np.uint8))
    executor._grasp_client=Mock()
    executor._grasp_client.detect_and_segment.return_value=(np.ones((8,8),bool),.96,(0,0,8,8))
    executor._extract_gripper=Mock();executor._start_perception=Mock()
    executor.start('onion near bowl',{},SimpleNamespace(episode_step=320,log=Mock()))
    assert executor.phase==GraspPhase.FAILED
    executor._extract_gripper.assert_not_called();executor._start_perception.assert_not_called()
    executor._grasp_client.predict_grasp.assert_not_called()


def test_trajectory_completion_waits_for_fresh_observation_and_failed_outcome_stops():
    executor=object.__new__(GraspToolExecutor)
    executor._verified_recovery=True;executor._phase=GraspPhase.RETREATING
    executor.verify_outcome=Mock(return_value=False);executor._noop_response=Mock(return_value={'actions':np.ones((8,8))})
    executor._advance_phase_after_trajectory({},SimpleNamespace())
    assert executor.phase==GraspPhase.VERIFYING
    executor.verify_outcome.assert_not_called()
    from vlm_orchestrator.proxy import _step_with_supervision
    from vlm_orchestrator.failure_handlers.stall import StallSupervisor
    supervisor=StallSupervisor();supervisor.clock({'step':400,'dt_s':1/15,'stop_supported':True},400)
    supervisor.tool='grasp';supervisor.tool_start=320
    state=SimpleNamespace(grasp_tool_executor=executor,supervisor_stop=None)
    def handle(obs,state,result):state.supervisor_stop={'reason':result.reason};return obs,state
    strategy=SimpleNamespace(_failure_handler=SimpleNamespace(supervisor=supervisor),_execute_handler_result=handle)
    fresh={'fresh_frame':'after_final_chunk'}
    response=_step_with_supervision(strategy,fresh,state,'grasp')
    executor.verify_outcome.assert_called_once_with(fresh,state)
    assert 'orchestrator_stop' in response and 'actions' not in response


def test_completed_trajectory_is_not_a_positive_grasp_assessment():
    image=np.zeros((8,8,3),np.uint8)
    call=Mock(return_value=json.dumps(dict(correct_target_held=False,lifted_clear=False,
        ambiguous=False,evidence='The arm has moved but the target remains on the table.')))
    assert not verify_grasp_outcome(call,image,image,None,'object',{})[0]
    assert target_queries('a green object')==['a green object']


def test_strategy_tool_constructor_arguments_are_supported():
    import ast
    import inspect
    from pathlib import Path
    from vlm_orchestrator.place.tool import PlaceToolExecutor
    from vlm_orchestrator.strategies import subgoal_base
    tree=ast.parse(Path(subgoal_base.__file__).read_text(encoding='utf-8'))
    classes={'GraspToolExecutor':GraspToolExecutor,'PlaceToolExecutor':PlaceToolExecutor}
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in classes:
            params=inspect.signature(classes[node.func.id]).parameters
            assert all(k.arg in params for k in node.keywords), (node.lineno,node.func.id)
