struct VertexOutput {
    @location(0) cls: u32,
    @builtin(position) position: vec4<f32>,
};

@vertex
fn main_vs(
    @location(0) particle_data: vec4<f32>,
    @location(1) particle_cls: u32,
    @location(2) position: vec2<f32>,
) -> VertexOutput {
    var result: VertexOutput;
    result.cls = particle_cls;
    result.position = vec4<f32>(position + particle_data.xy, 0.0, 1.0);
    return result;
}

@fragment
fn main_fs(vertex: VertexOutput) -> @location(0) vec4<f32> {
    if (vertex.cls == 3u) {
        return vec4<f32>(1f,1f,1f,1f);
    }
    var color = vec4<f32>(0f,0f,0f,1f);

    if (vertex.cls == 0u) {
        color.x = 1f;
    }
    else if (vertex.cls == 1u) {
        color.y = 1f;
    }
    else if (vertex.cls == 2u) {
        color.z = 1f;
    }
    else {
        color.w = 1f;
    }
    return color;
}