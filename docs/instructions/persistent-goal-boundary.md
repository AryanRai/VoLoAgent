# Fresh goals within a persistent physical session

The simulator-owned session lives in experimental's `persistent_session.py`.
It retains Isaac and submits sequential instructions over the existing client
connection. This is not persistent LLM conversation or a new network API.

The client opts in using `__fresh_goal: true`, a unique `__episode_id` string
`session_id:goal_index`, and real `__step` control metadata. At an existing
new-episode boundary, the proxy first finalizes old logging, resets old grasp/
place executors, and creates fresh `SessionState`. The unchanged strategy's
episode initializer resets failure monitors and decomposes the new instruction
from current observations. Prior instructions/images/subgoals are not carried
into the new planning context. Repeated queries with the same marker do not
reset ongoing work. The response acknowledges `orchestrator_goal_id`; the
session client rejects unacknowledged actions before execution.

No model weights are reloaded and no model-specific normalization or action
conversion changes. π0.5 and GR00T's current inference servers require no
additional recurrent-state reset. The marker is removed before policy inference.
Unmarked benchmark clients retain existing proxy behavior.

Task success and timeout remain simulator/caller evaluator responsibilities,
never an inference from Astra's language response. Classical recovery and
supervisor-stop attribution remain on their existing paths.

Validation: `python -m pytest tests/test_fresh_goal.py tests/test_proxy.py -q`.
See experimental's `PERSISTENT_SESSION_RESULTS.md` for cross-repository and real
simulator evidence. Branch `exp/persistent-session`; no main merge.
