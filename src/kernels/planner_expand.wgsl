// Planner Stage 1 — Expand.
// Generates candidate next states from the current plan block.
//
// Buffers:
//   0: plan_state     u32[]   (encoded current plan state)
//   1: candidates_out u32[]   (output: expanded candidate states)
//   2: candidate_len  u32[1]  (atomic counter for candidates_out)
//
// TODO: implement action enumeration and state transition logic.

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // For each candidate action at position gid.x:
    //   apply action to plan_state → write candidate to candidates_out
    let _ = gid;
}
