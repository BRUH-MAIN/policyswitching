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

- `switching_by_own_scan.mp4`: the same course, leader and seed, but **the robot chooses the
  policy itself**. A small network (187 -> 128 -> 64 -> 3) reads the same 187-point height
  scan the locomotion policies read, with its training noise (+-10 cm per ray), and predicts
  flat / rough / stairs; the probabilities are smoothed (weight 0.3 on the newest scan) and a
  new class must persist for 3 steps before the policy changes. Nothing about the course
  layout reaches the robot; the layout strip is drawn for the viewer only, and the three
  bars show the classifier's smoothed output. In this trial it completes the course with 54
  switches and matches the ground-truth rule on 86% of steps. It enters and leaves all four
  flights within 0.1 s of the ground-truth switch; the extra switches are flat/rough
  flicker on the rough ground, whose 2-8 cm relief is under the scan noise.

  How the classifier was made (weights are not in git; `logs/` is ignored): scans recorded
  with `switch_follow.py --record-scans` on the `rough`, `stairs_up` and `stairs_down`
  courses (15 cm risers, 5- and 10-step flights, seeds 700 and 701, 128 robots each, noise
  on), trained with `scripts/scan_classifier_train.py`. Held-out per-scan accuracy 91.9%
  (recall: flat 89%, rough 79%, stairs 99.5%). Tested on the `multi` course at 15 cm
  risers, which it was not trained on (256 trials per arm, seeds 400 and 401): 99.6% of
  courses completed with the classifier choosing, against 100% with ground-truth
  switching; it switched 0.2-0.3 m before every terrain boundary and was never late. The
  filter setting was picked on that course, not on the demo trial. Smoothing harder fails:
  at weight 0.03 / hold 20 the switch comes late and 43% of trials are lost.

- `wandering_leader_own_scan.mp4` (110 s, recorded 2026-10-09 with
  `unitree_rl_mjlab/scripts/record_wandering_demo.py`): **the leader does not walk a straight
  line, and the robot chooses its policy itself.** On a wider course (45.5 x 10 m: flat
  plaza, rough ground, a 10-step 15 cm flight up to a raised landing, the flight down, a
  second plaza, more rough ground) the leader loops one and a quarter times round the first
  plaza (radius 2.5 m), crosses the first rough patch on a diagonal, walks up the stairs,
  circles on the landing (radius 2 m), walks down, and weaves through the second rough
  patch, at 0.7 m/s with the robot holding 1.5 m behind. The overlay is a top-down map: the
  course coloured by terrain, the leader's path, and the robot's trail coloured by the
  policy it was running. In this trial the robot finishes with 142 switches, matching the
  ground-truth rule on 86% of steps; the four switches to and from the stairs policy happen
  at the two flights and nowhere else.

  The first classifier (trained on straight-line walking only) flickered between flat and
  rough far more when the robot turned (257 switches, 68% agreement on an earlier version of
  this course), so it was retrained with scans from robots following the **mirror image** of
  this path on two other terrain seeds added: held-out per-scan accuracy 87% (flat 86%,
  rough 70%, stairs 99.6%). Checks on that classifier: on the straight `multi` course at
  15 cm risers, 256 of 256 trials completed (ground-truth switching: 256 of 256), never
  late at a boundary; on this wandering path, none of 64 robots (two terrain seeds, noise
  on) fell with the classifier choosing. Flat-versus-rough is still its weak point; stairs
  are not.

What the first clip shows is the switching mechanism working. It is not evidence that
switching helps: the study's result over randomised layouts is that switching between
specialists adds nothing over the best single policy (`objective.md`, "What the result
was"), and the second clip is the same thing in one sample. The first clip's switch uses the
true course layout; the third uses the robot's own (simulated) height scan, which on the
real robot would have to come from the LiDAR scan node. A clip is one trial, not a success rate. Nothing
here has run on the real robot.
