"""Opt-in pick/place watchdog driven only by actual executed control time."""
from dataclasses import dataclass
import math
import os
from .base import HandlerResult, STATUS_FAILURE, ACTION_REPLAN, ACTION_GRASP, ACTION_PLACE, ACTION_STOP

PROMPT = '''
BOUNDED PROGRESS ASSESSMENT (overrides earlier recovery restrictions):
Compare recent snapshots, not just whether the task is still achievable.
Report an additional JSON object "progress" with these fields:
"phase": "unknown"|"approach"|"grasp"|"lift"|"transport"|"placement",
"evidence": a concrete visible explanation, "target_confirmed": true|false,
"target": the target's visual noun phrase, "held": true|false|null,
"destination": a full placement phrase, e.g. "in the white bowl".
held refers to the correct target; use null if occluded/ambiguous. A closing
or moving empty gripper is not a grasp. Grasp/lift/transport require visible
evidence the correct object is held. Placement requires visible release at
the intended destination. Mere arm movement or a task remaining possible is
not progress. The controller enforces retry limits. Correct-object missed
grasps may now trigger recovery after replanning; wrong-object-only rules do
not apply. Never use simulator ground truth as evidence. Use unknown if unsure.
'''


@dataclass(frozen=True)
class StallConfig:
    stall_steps: int = 240
    hard_steps: int = 480
    tool_steps: int = 450

    @classmethod
    def environment(cls):
        values = [int(os.environ.get(key,default)) for key,default in
                  [('STALL_STEPS',240),('STALL_HARD_STEPS',480),('STALL_TOOL_STEPS',450)]]
        if min(values) <= 0 or values[1] <= values[0]:
            raise ValueError('Invalid watchdog control-step limits')
        return cls(*values)


class StallSupervisor:
    def __init__(self, config=None):
        self.config = config or StallConfig()
        self.reset()

    def reset(self):
        self.step = self.dt = None
        self.started = self.progress_step = self.recovery_step = self.rank = 0
        self.replans = self.grasps = self.places = 0
        self.tool = self.tool_start = self.stop_reason = None
        self.assessment = {}
        self.history = []

    def clock(self, metadata, step):
        if not isinstance(metadata, dict) or metadata.get('stop_supported') is not True:
            return self.abort('missing_supervisor_client_capability')
        raw_step, dt = metadata.get('step'), metadata.get('dt_s')
        if (type(raw_step) is not int or raw_step < 0 or raw_step != step or
                not isinstance(dt,(int,float)) or not math.isfinite(dt) or dt <= 0):
            return self.abort('invalid_simulator_clock')
        if self.step is None:
            self.started = self.progress_step = raw_step
            self.dt = float(dt)
        elif raw_step < self.step or not math.isclose(dt,self.dt,abs_tol=1e-12):
            return self.abort('nonmonotonic_or_changed_simulator_clock')
        self.step = raw_step  # equal is valid for same-step flush/re-query
        return None

    def snapshot(self):
        return {'step':self.step,'dt_s':self.dt,'replans':self.replans,'grasps':self.grasps,
                'places':self.places,'tool':self.tool,'last_progress_step':self.progress_step,
                'elapsed_control_s':(self.step-self.started)*self.dt if self.dt is not None else None,
                'stop_reason':self.stop_reason,'recent_assessments':self.history[-3:]}

    def abort(self, reason):
        self.stop_reason = self.stop_reason or reason
        return HandlerResult(status=STATUS_FAILURE,action=ACTION_STOP,reason=self.stop_reason,
                             extra={'supervisor':self.snapshot()})

    def assess(self, data):
        a = data if isinstance(data,dict) else {}
        valid = (a.get('target_confirmed') is True and isinstance(a.get('target'),str)
                 and bool(a['target'].strip()) and isinstance(a.get('evidence'),str)
                 and len(a['evidence'].strip()) >= 10)
        phase = a.get('phase')
        if not isinstance(phase,str) or phase not in ('unknown','approach','grasp','lift','transport','placement'):
            valid = False
            phase = 'unknown'
        rank = {'grasp':2,'lift':3,'transport':4,'placement':5}.get(phase,0)
        held = a.get('held')
        valid = valid and (held is True if rank in (2,3,4) else held is False if rank == 5 else True)
        if valid and rank > self.rank:
            self.rank = rank
            self.progress_step = self.step
        self.assessment = a if valid else {}
        self.history.append({'step':self.step,'assessment':self.assessment or {'phase':'unknown'}})
        self.history = self.history[-3:]

    def check_tool(self, state):
        if self.tool:
            executor = getattr(state,f'{self.tool}_tool_executor',None)
            phase = getattr(getattr(executor,'phase',None),'value',None)
            if phase == 'failed':
                return self.abort(f'{self.tool}_tool_failed')
            if phase == 'done':
                self.tool = self.tool_start = None
            elif self.step-self.tool_start >= self.config.tool_steps:
                return self.abort('tool_control_timeout')
        return None

    def decide(self, *, allow_recovery=True):
        if self.stop_reason:
            return self.abort(self.stop_reason)
        elapsed = self.step-self.progress_step
        if elapsed >= self.config.hard_steps:
            return self.abort('no_verified_milestone_control_timeout')
        if self.tool or not allow_recovery or elapsed < self.config.stall_steps:
            return None
        if self.replans == 0:
            self.replans = 1; self.recovery_step = self.step
            return HandlerResult(status=STATUS_FAILURE,action=ACTION_REPLAN,
                                 reason='sustained_stall: '+str(self.assessment))
        # The next monitor can confirm persistent stall after replanning;
        # restarting a full window would collide with the absolute 480 cap.
        if self.step <= self.recovery_step:
            return None
        a = self.assessment
        if not a or type(a.get('held')) is not bool:
            return self.abort('unresolved_recovery_target_or_held_state')
        if a['held'] is False and self.grasps == 0 and self.places == 0:
            self.grasps = 1; self.tool = 'grasp'; self.tool_start = self.step
            return HandlerResult(status=STATUS_FAILURE,action=ACTION_GRASP,
                                 reason='bounded_correct_target_missed_grasp',grasp_target=a['target'])
        if a['held'] is True and self.places == 0 and isinstance(a.get('destination'),str) and a['destination'].strip():
            self.places = 1; self.tool = 'place'; self.tool_start = self.step
            return HandlerResult(status=STATUS_FAILURE,action=ACTION_PLACE,
                                 reason='bounded_target_held_placement_stall',place_destination=a['destination'],
                                 place_held_object=a['target'])
        return self.abort('recovery_budget_exhausted_or_unresolved_destination')
