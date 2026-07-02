// Planner Stage 4 — Refine.
// Applies local optimisation to the surviving candidate(s).
//
// Buffers:
//   0: candidates    u32[]   (surviving candidates after prune)
//   1: keep_flags    u32[]   (mask from stage 3)
//   2: refined_out   u32[]   (output: refined plan state)
//
// TODO: implement local perturbation / gradient step / beam refinement.

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // if keep_flags[gid.x]: refined_out[out_pos] = refine(candidates[gid.x])
    let _ = gid;
}
