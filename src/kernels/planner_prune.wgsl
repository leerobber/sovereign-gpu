// Planner Stage 3 — Prune.
// Keeps the top-k scoring candidates; marks the rest as dead.
//
// Buffers:
//   0: scores         f32[]   (scores from stage 2)
//   1: keep_flags     u32[]   (1 = keep, 0 = prune; one per candidate)
//   2: top_k          u32[1]  (k — number of candidates to retain)
//
// TODO: implement parallel top-k selection (e.g. bitonic sort or threshold).

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // keep_flags[gid.x] = (rank_of(scores[gid.x]) < top_k) ? 1 : 0
    let _ = gid;
}
