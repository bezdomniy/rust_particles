struct Ubo {
    transform: mat4x4<f32>,
    dt: f32
};

struct Particle {
    pos: vec2<f32>,
    vel: vec2<f32>,
};

@group(0)
@binding(0)
var<uniform> ubo: Ubo;

@group(0) 
@binding(1) 
var<storage, read_write> particles: array<Particle>;

@compute
@workgroup_size(64)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_invocation_id: vec3<u32>,
) {
    let idx = workgroup_id.x * 64u + local_invocation_id.x;
    let p = particles[idx];
    let dt = ubo.dt;

    particles[idx].pos = p.pos + p.vel * dt;
}