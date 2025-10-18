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
@workgroup_size(16,16,1)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_invocation_id: vec3<u32>,
) {
    let group_idx1 = workgroup_id.x * 16u;
    let group_idx2 = workgroup_id.y * 16u;

    let idx1 = group_idx1 + local_invocation_id.x;
    let idx2 = group_idx2 + local_invocation_id.y;

    if (idx1 == idx2) {
        return;
    }

    //TODO - add to ubo
    let g = 0.0000001;
    let viscosity = 0.7;

    let p1 = particles[idx1];
    let p2 = particles[idx2];

    let d = distance(p1.pos, p2.pos);
    let dir = normalize(p2.pos - p1.pos);
    let f = g * (1.0/(d)); // TODO: should distance be squared?

    let force = dir * f;

    particles[idx1].vel += (force * (1.0 - viscosity)) * ubo.dt;
    particles[idx1].pos += particles[idx1].vel;
}