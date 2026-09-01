# Model Interfaces

Roz evaluates the predicted next latent: for a world model `W`, we score
`W(o_1:t) = z_{t+1}`, the representation of the scene *after* the observed
clip. Models differ in how much of that interface they actually expose, so
below we state exactly what is extracted from each one, and — where a model
has no forecast head — what we substitute and what that substitution costs.
Every readout is a single forward pass, and every model is given the same
256×256 clips with inter-frame prediction disabled in the encoder.

**V-JEPA 2.** We encode the full clip in a single forward pass and obtain the
next latent through V-JEPA 2's native predictor. The real clip occupies the
first 62 of the encoder's 64 frame slots and the remaining slots hold the
frozen final frame; the predictor is asked for the token positions of that
padded tail, conditioned on all of the real clip. The resulting embedding is
the model's forecast of the scene one temporal token (two frames, ~0.1 s)
past the end of the observation. Target positions are represented inside the
predictor by a learned mask token and a positional embedding, so the
placeholder pixels never enter the prediction itself; they only sit in the
encoder's field of view, where they carry no information the clip did not
already contain. We note that V-JEPA 2 pretrains with multi-block masks at a
temporal mask ratio of approximately one — spatial blocks spanning every
frame — so it was never trained to predict later timesteps from earlier ones,
and this readout is therefore out of distribution. Because the invariant is
relative and the identical condition applies to A, B and C, the shift cancels
in scoring, but the readout should not be read as the model used as trained.

**V-JEPA 2-AC.** The action-conditioned variant is the only model in the suite
trained directly on the objective Roz scores, so we use it without
substitution. Its encoder is applied per frame — each frame is duplicated to
fill the two-frame tubelet — which means every temporal dependency passes
through the block-causal predictor, and the forecast is future-blind by
construction. The predictor's output at the final position is the
representation of the next frame, taken in one teacher-forced pass. It
requires a 7-dimensional action and a 7-dimensional proprioceptive state per
frame; we supply the null "observe" action as a zero vector, which is in
distribution because actions are unnormalized relative deltas and zero denotes
an end effector that did not move, and we hold the state at a fixed plausible
end-effector pose with the gripper open, because the state is an absolute pose
in the robot base frame for which zero is kinematically impossible. Our scenes
contain no manipulator, so the action's coordinate frame is undefined; this is
precisely why a zero action is the only defensible choice.

**Qwen3.6.** The model is autoregressive, so the hidden state at the final
video-token position is the state from which the next token is predicted, and
we take that as the next latent. We read the middle-plus-one decoder layer,
restrict to video tokens, and keep only the last temporal step's grid. This is
an approximation rather than a forecast head: the model predicts discrete text
tokens, not scene latents, and the representation we extract is the one it
would use to do so.

**Cosmos 3.** Cosmos exposes no forecast of the scene beyond the clip on the
path we can drive. Running the pipeline for one denoising step with
`output_type="latent"` returns a VAE-space tensor covering the same frames
that were supplied as conditioning — a one-step reconstruction of the observed
window rather than a prediction of the next one. We therefore read the
generation stream's middle-plus-one decoder layer over the final frames of the
clip and treat it as an encoding of the observed end state. Scores for Cosmos
should be read as measuring what its representation retains, not what it
forecasts.

**FastWAM.** FastWAM is a world-action model: its forward pass encodes the
current observation and denoises an action, and it emits no future-state
latent at all. We capture the hidden state of the middle-plus-one block of the
action expert during that pass, conditioned on a fixed cached prompt embedding
in place of a task instruction. The model sees only the final frame, supplied
as the two stereo views concatenated. We retain FastWAM deliberately despite
the mismatch: it is the clearest illustration of the paper's argument, in that
the model scoring highest on tasks reducible to image similarity is also the
one whose latent is least a prediction of the world's next state.

**AnonWM.** Our own checkpoints consume pre-encoded visual tokens rather than
pixels, and we read the 32-dimensional `latent_head` output over the clip's
final second. Like Cosmos, this is an encoding of the observed end state; the
model's predictive head (`visual_head`, trained for masked-encoding
prediction) is not used here, so the ladder measures what successive
checkpoints retain rather than what they forecast.

**Distance and pooling.** For every model we report the unreduced (flat)
vector under both cosine and L1 distance. Mean-pooled variants are computed
but not reported, except where a pooled readout is the only one a model
exposes at a comparable dimension; pooling discards spatial arrangement and
systematically inverts results on position-sensitive tasks, which we treat as
a property of the readout rather than of the model.
