struct VertexOutput {
    @location(0) cls: u32,
    @builtin(position) position: vec4<f32>,
};

@vertex
fn main_vs(
    @location(0) particle_pos: vec2<f32>,
    @location(1) particle_cls: u32,
    @location(2) position: vec2<f32>,
) -> VertexOutput {
    var result: VertexOutput;
    result.cls = particle_cls;
    result.position = vec4<f32>(position + particle_pos, 0.0, 1.0);
    return result;
}

@fragment
fn main_fs(vertex: VertexOutput) -> @location(0) vec4<f32> {
    if (vertex.cls == 3u) {
        return vec4<f32>(1f,1f,1f,1f);
    }
    var color = vec4<f32>(0f,0f,0f,1f);
    color[vertex.cls] = 1f;
    return color;
}