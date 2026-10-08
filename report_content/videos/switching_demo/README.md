# Policy-switching demo (simulation)

Recorded 2026-10-08 with `unitree_rl_mjlab/scripts/record_switching_demo.py`. One robot, one
trial, 70 s: it follows a scripted walking leader (0.55 m/s, 2 m ahead) along a 42.5 m course
of flat ground, rough ground, a 10-step flight up and down, more rough ground and a 5-step
flight up and down (15 cm risers, 0.30 m tread). Training sensor noise is on.

- `switching_demo.mp4`: the active locomotion policy is switched between three specialists
  by the terrain class under the robot's footprint (0.30 m ahead of the base to 0.35 m
  behind it, non-flat winning), read from the course layout. 12 switches. Policies: the flat
  and rough specialists (`go2_spec_flat`, `go2_spec_rough`, `model_9999`) and stairs v8a
  (`go2_spec_stairs_v8a/model_3999`). The badge shows the active policy; the bottom strip
  shows the terrain along the course and which policy was active where.
- `same_course_one_policy.mp4`: the same course, leader and seed with stairs v8a alone and
  no switching. It also completes the course.

What the first clip shows is the switching mechanism working. It is not evidence that
switching helps: the study's result over randomised layouts is that switching between
specialists adds nothing over the best single policy (`objective.md`, "What the result
was"), and the second clip is the same thing in one sample. The switch here uses the true
course layout, not a perception system. A clip is one trial, not a success rate. Nothing
here has run on the real robot.
