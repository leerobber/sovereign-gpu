// Planner Stage 2 — Score.
// Assigns a fitness score to each candidate state.
//
// Buffers:
//   0: candidates  u32[]   (candidate states from expand)
//   1: scores      f32[]   (output: one score per candidate)
//   2: goal        u32[]   (encoded goal state for comparison)
//
// TODO: implement heuristic or learned scoring function.

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // score[gid.x] = heuristic(candidates[gid.x], goal)
    let _ = gid;
}
