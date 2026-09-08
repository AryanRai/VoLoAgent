import json
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pytest
from vlm_orchestrator.failure_handlers.stall import StallSupervisor, StallConfig
from vlm_orchestrator.failure_handlers.vlm import VLMFailureHandler


def clock(s, step):
    return s.clock({'step':step,'dt_s':1/15,'stop_supported':True},step)


def assessment(phase='approach', held=False):
    return {'phase':phase,'held':held,'target_confirmed':True,'target':'orange fruit',
            'destination':'in the white bowl','evidence':'The orange remains on the table beside an empty gripper.'}


def test_stall_replan_tool_then_absolute_stop():
    s=StallSupervisor();clock(s,0)
    for n in (80,160):
        clock(s,n);s.assess(assessment());assert s.decide() is None
    clock(s,240);assert s.decide().action=='replan'
    # Replanning does not reset the milestone timer or grant another replan.
    clock(s,320);assert s.decide().action=='grasp_tool'
    clock(s,400);assert s.decide() is None
    clock(s,480);assert s.decide().action=='stop'
    assert (s.replans,s.grasps,s.places)==(1,1,0)


def test_milestones_and_one_grasp_one_place():
    s=StallSupervisor();clock(s,0);s.assess(assessment())
    clock(s,240);s.decide();clock(s,320);s.decide()
    state=SimpleNamespace(grasp_tool_executor=SimpleNamespace(phase=SimpleNamespace(value='done')))
    assert s.check_tool(state) is None
    s.assess(assessment('lift',True));assert s.progress_step==320
    clock(s,560);s.assess(assessment('lift',True))
    assert s.decide().action=='place_tool'
    state.place_tool_executor=SimpleNamespace(phase=SimpleNamespace(value='done'))
    s.check_tool(state);clock(s,568)
    assert s.decide().action=='stop'
    assert s.replans==s.grasps==s.places==1


@pytest.mark.parametrize('value',[None,{},[],{'phase':[]},assessment('grasp',False)])
def test_unknown_or_invalid_does_not_make_progress(value):
    s=StallSupervisor();clock(s,0);clock(s,240);s.assess(value)
    assert s.progress_step==0 and s.decide().action=='replan'
    clock(s,320)
    assert s.decide().action=='stop'


def test_failed_tool_and_timeout():
    for phase,expected in [('failed','grasp_tool_failed'),('approaching','tool_control_timeout')]:
        s=StallSupervisor(StallConfig(240,2000,450));clock(s,0);s.assess(assessment())
        clock(s,240);s.decide();clock(s,320);s.decide();clock(s,770)
        state=SimpleNamespace(grasp_tool_executor=SimpleNamespace(phase=SimpleNamespace(value=phase)))
        assert s.check_tool(state).reason==expected


def test_clock_validation_and_same_step_flush():
    s=StallSupervisor();assert clock(s,0) is None;assert clock(s,0) is None
    clock(s,8);assert clock(s,7).action=='stop'
    s.reset();assert s.clock(None,0).action=='stop'
    s.reset();assert s.clock({'step':0,'dt_s':float('nan'),'stop_supported':True},0).action=='stop'


def test_recent_context_budget_survives_rewording_and_wall_delays(monkeypatch):
    monkeypatch.setenv('VOLO_STALL_SUPERVISOR','1')
    calls=[]
    def model(system,message):
        calls.append((system,message))
        return json.dumps({'status':'in_progress','action':'continue','progress':assessment()})
    handler=VLMFailureHandler(model,lambda *a,**k:[],lambda o:np.zeros((8,8,3),np.uint8),lambda o:[])
    state=SimpleNamespace(episode_step=0,subgoals=['Put orange in bowl'],current_subgoal_idx=0,
                          original_instruction='Put orange in bowl',initial_image=None,initial_extra_images=None,
                          infer_count=0,log=Mock())
    def observation(step):return {'__supervisor':{'step':step,'dt_s':1/15,'stop_supported':True}}
    handler.on_episode_start(observation(0),state)
    results=[]
    for step in [80,160,240,320]:
        state.episode_step=step
        # Arbitrarily large wall-clock values must not affect the decisions.
        monkeypatch.setattr('time.time',lambda:1e12+step*1000)
        result=handler.step(observation(step),state)
        results.append(result.action if result else None)
        if step==240:
            state.subgoals=['Place the orange into the bowl']
            handler.on_subgoal_advanced(observation(step),state,0)
    assert results==[None,None,'replan','grasp_tool']
    recent=[p for p in calls[-1][1] if p.get('type')=='text' and p['text'].startswith('RECENT')]
    assert len(recent)==2 and 'step 160' in recent[0]['text'] and 'step 240' in recent[1]['text']
    assert handler.supervisor.replans==1


def test_disabled_mode_keeps_original_prompt(monkeypatch):
    monkeypatch.delenv('VOLO_STALL_SUPERVISOR',raising=False)
    h=VLMFailureHandler(Mock(),Mock(),Mock(),Mock())
    assert h.supervisor is None and 'BOUNDED PROGRESS' not in h._system_prompt


def test_failed_activation_becomes_structured_stop_not_policy_retry():
    from vlm_orchestrator.strategies.subgoal import SubgoalStrategy
    strategy=object.__new__(SubgoalStrategy)
    s=StallSupervisor();clock(s,0);clock(s,240);s.assess(assessment());s.decide()
    clock(s,320);decision=s.decide()
    strategy._failure_handler=SimpleNamespace(supervisor=s)
    strategy._activate_grasp_tool=Mock(return_value=False)
    state=SimpleNamespace(subgoals=['Put orange in bowl'],current_subgoal_idx=0,
                          grasp_tool_active=False,place_tool_active=False,log=Mock())
    _,state=strategy._execute_handler_result({},state,decision)
    assert state.supervisor_stop['reason']=='tool_activation_failed'
    assert state.flush_actions is True


def test_missing_subgoals_cannot_bypass_deadline(monkeypatch):
    monkeypatch.setenv('VOLO_STALL_SUPERVISOR','1')
    handler=VLMFailureHandler(Mock(),Mock(),Mock(),Mock())
    state=SimpleNamespace(episode_step=0,subgoals=[])
    handler.on_episode_start({'__supervisor':{'step':0,'dt_s':1/15,'stop_supported':True}},state)
    state.episode_step=480
    assert handler.step({'__supervisor':{'step':480,'dt_s':1/15,'stop_supported':True}},state).action=='stop'


def test_replan_uses_recent_failure_evidence_and_preserves_attempt_budget():
    from vlm_orchestrator.strategies.subgoal import SubgoalStrategy
    strategy=object.__new__(SubgoalStrategy)
    s=StallSupervisor();clock(s,0);clock(s,240);s.assess(assessment());s.decide()
    strategy._failure_handler=SimpleNamespace(supervisor=s,_recent_snapshots=[(240,[np.zeros((4,4,3),np.uint8)])])
    strategy._recycle_count=0;strategy._original_subgoals=['Place orange in bowl']
    strategy.ctx=SimpleNamespace(vlm_camera_labels=('Front',[]),prompt_style='robolab',front_image_key=None)
    strategy._vlm_call=Mock(side_effect=RuntimeError('failed replan'))
    state=SimpleNamespace(original_instruction='Place orange in bowl',initial_image=None,log=Mock())
    assert strategy._recycle({},state,np.zeros((4,4,3),np.uint8)) is False
    message=strategy._vlm_call.call_args.args[1]
    assert 'Recent visual stall evidence' in str(message) and 'orange remains on the table' in str(message)
    assert s.replans==1


def test_replanned_instruction_applied_before_policy_query():
    from vlm_orchestrator.strategies.subgoal import SubgoalStrategy
    strategy=object.__new__(SubgoalStrategy)
    decision=Mock(action='replan')
    strategy._failure_handler=SimpleNamespace(supervisor=object(),step=lambda *args:decision)
    state=SimpleNamespace(rewritten_instruction='Replanned goal')
    strategy._execute_handler_result=Mock(return_value=({},state))
    strategy.ctx=SimpleNamespace(set_prompt=lambda obs,text:dict(obs,prompt=text))
    obs,_=strategy._on_step({},state)
    assert obs['prompt']=='Replanned goal'


@pytest.mark.parametrize('raises',[False,True])
def test_failure_during_tool_step_discards_its_chunk(raises):
    from vlm_orchestrator.proxy import _step_with_supervision
    from vlm_orchestrator.strategies.subgoal import SubgoalStrategy
    strategy=object.__new__(SubgoalStrategy)
    s=StallSupervisor();clock(s,0);s.tool='grasp';s.tool_start=0
    strategy._failure_handler=SimpleNamespace(supervisor=s)
    executor=SimpleNamespace(phase=SimpleNamespace(value='approaching'))
    def step(*args):
        executor.phase.value='failed'
        if raises:raise RuntimeError('tool failed')
        return {'actions':np.zeros((8,8))}
    executor.step=step
    state=SimpleNamespace(grasp_tool_executor=executor,subgoals=['Pick orange'],current_subgoal_idx=0,
                          grasp_tool_active=True,place_tool_active=False,log=Mock())
    response=_step_with_supervision(strategy,{},state,'grasp')
    assert 'actions' not in response and 'orchestrator_stop' in response
