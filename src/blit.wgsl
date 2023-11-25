struct VertexInput {
   @builtin(vertex_index) index: u32,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_main(vertex: VertexInput) -> VertexOutput {
    var result: VertexOutput;
    if (vertex.index == 0u) {
        result.position = vec4<f32>(-1.0, -1.0, 0.0, 1.0);
    } else if (vertex.index == 1u) {
        result.position = vec4<f32>(1.0, -1.0, 0.0, 1.0);
    } else if (vertex.index == 2u) {
        result.position = vec4<f32>(-1.0, 1.0, 0.0, 1.0);
    } else if (vertex.index == 3u) {
        result.position = vec4<f32>(1.0, 1.0, 0.0, 1.0);
    }
    return result;
}

@group(0)
@binding(0)
var texture_rgba: texture_2d<f32>;

@fragment
fn fs_main(vertex: VertexOutput) -> @location(0) vec4<f32> {
    var color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    for (var i: i32 = 0; i < 2; i++) {
        for (var j: i32 = 0; j < 2; j++) {
            let c = vec2<i32>(i32(vertex.position.x * 2.0) + i, i32(vertex.position.y * 2.0) + j);
            color += textureLoad(texture_rgba, c, 0);
        }
    }
    return color / 4.0;
}