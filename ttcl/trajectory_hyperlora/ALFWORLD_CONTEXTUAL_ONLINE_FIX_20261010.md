# ALFWorld contextual online pilot and runner fix

The prequential ALFWorld runner was still passing old relational fields to a
task-conditioned contextual checkpoint. It now encodes the complete public
trajectory as a bounded contextual text field and applies declared head/tail
truncation at 2048 tokens.

On the reviewed 18-game train sequence, using identical environment settings
for the paired arms, the fixed runner completed all 18 games without an
infrastructure failure. The no-adapter arm won 6/18 and the accumulated online
trajectory arm won 4/18 (online-only 1, base-only 3). This is a baseline
diagnostic, not the v101 result; it prevents context-shape failures from being
mistaken for learning effects.

The next run will use the same binding and budget with the v101 persistent
basis-state arm.
