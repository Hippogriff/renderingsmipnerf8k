struct Globals {
    view_proj: mat4x4<f32>,
    cam_pos: vec4<f32>,
};

@group(0)
@binding(0)
var<uniform> u_globals: Globals;

struct VertexOutput {
    @builtin(position) proj_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) world_pos: vec3<f32>,
};

@vertex
fn vs_main(
    @location(0) position: vec4<f32>,
    @location(1) uv: vec2<f32>,
) -> VertexOutput {
    var result: VertexOutput;
    result.proj_position = u_globals.view_proj * position;
    result.world_pos = position.xyz;
    result.uv = uv;
    return result;
}

@group(0)
@binding(1)
var frame_depth: texture_2d<f32>;
@group(0)
@binding(2)
var frame_rgba: texture_2d<f32>;
@group(0)
@binding(3)
var color_texture: texture_2d<f32>;
@group(0)
@binding(4)
var alpha_texture: texture_2d<f32>;
@group(0)
@binding(5)
var samp: sampler;

@group(0)
@binding(6)
var lobe_color0: texture_2d<f32>;
@group(0)
@binding(7)
var lobe_color1: texture_2d<f32>;
@group(0)
@binding(8)
var lobe_color2: texture_2d<f32>;
@group(0)
@binding(9)
var lobe_color3: texture_2d<f32>;
@group(0)
@binding(10)
var lobe_color4: texture_2d<f32>;
@group(0)
@binding(11)
var lobe_color5: texture_2d<f32>;
// @group(0)
// @binding(12)
// var lobe_color6: texture_2d<f32>;
// @group(0)
// @binding(13)
// var lobe_color7: texture_2d<f32>;
// @group(0)
// @binding(14)
// var lobe_color8: texture_2d<f32>;
// @group(0)
// @binding(15)
// var lobe_color9: texture_2d<f32>;

@group(0)
@binding(16)
var lobe_lambda_axis0: texture_2d<f32>;
@group(0)
@binding(17)
var lobe_lambda_axis1: texture_2d<f32>;
@group(0)
@binding(18)
var lobe_lambda_axis2: texture_2d<f32>;
@group(0)
@binding(19)
var lobe_lambda_axis3: texture_2d<f32>;
@group(0)
@binding(20)
var lobe_lambda_axis4: texture_2d<f32>;
@group(0)
@binding(21)
var lobe_lambda_axis5: texture_2d<f32>;
// @group(0)
// @binding(22)
// var lobe_lambda_axis6: texture_2d<f32>;
// @group(0)
// @binding(23)
// var lobe_lambda_axis7: texture_2d<f32>;
// @group(0)
// @binding(24)
// var lobe_lambda_axis8: texture_2d<f32>;
// @group(0)
// @binding(25)
// var lobe_lambda_axis9: texture_2d<f32>;


struct FragmentOutput {
    @location(0) color: vec4<f32>,
    @location(1) frag_depth: f32,
};

fn inv_sigmoid(x: vec3<f32>) -> vec3<f32> {
    return -log(1.0 / x - 1.0);
}

@fragment
fn fs_main(vertex: VertexOutput) -> FragmentOutput {
    let c = vec2<i32>(i32(vertex.proj_position.x), i32(vertex.proj_position.y));
    let prev_depth = textureLoad(frame_depth, c, 0).x;
    var uv = vertex.uv;
    let temp = uv.x;
    uv.x = uv.y;
    uv.y = temp;
    // uv.y = 1.0 - uv.y;

    let ddx = dpdx(uv);
    let ddy = dpdy(uv);

    let eps = 1e-7;
    if (vertex.proj_position.z <= prev_depth + eps) {
        discard;
    }

    let prev_color = textureLoad(frame_rgba, c, 0);
    let prev_rgb = prev_color.xyz;
    let prev_alpha = prev_color.w;

    if (prev_alpha > 0.99) {
        discard;
    }

    let alpha = textureSampleGrad(alpha_texture, samp, uv, ddx, ddy).x;

    if (alpha * (1.0 - prev_alpha) < 0.05) {
        discard;
    }

    let diffuse = textureSampleGrad(color_texture, samp, uv, ddx, ddy).xyz;

    let view_dir = normalize(vertex.world_pos - u_globals.cam_pos.xyz);

    var lobe_colors: array<vec3<f32>, 10>;
    lobe_colors[0] = textureSampleGrad(lobe_color0, samp, uv, ddx, ddy).xyz;
    lobe_colors[1] = textureSampleGrad(lobe_color1, samp, uv, ddx, ddy).xyz;
    lobe_colors[2] = textureSampleGrad(lobe_color2, samp, uv, ddx, ddy).xyz;
    lobe_colors[3] = textureSampleGrad(lobe_color3, samp, uv, ddx, ddy).xyz;
    lobe_colors[4] = textureSampleGrad(lobe_color4, samp, uv, ddx, ddy).xyz;
    lobe_colors[5] = textureSampleGrad(lobe_color5, samp, uv, ddx, ddy).xyz;
    // lobe_colors[6] = textureSampleGrad(lobe_color6, samp, uv, ddx, ddy).xyz;
    // lobe_colors[7] = textureSampleGrad(lobe_color7, samp, uv, ddx, ddy).xyz;
    // lobe_colors[8] = textureSampleGrad(lobe_color8, samp, uv, ddx, ddy).xyz;
    // lobe_colors[9] = textureSampleGrad(lobe_color9, samp, uv, ddx, ddy).xyz;
    

    var lobe_lambda_axes: array<vec3<f32>, 10>;
    lobe_lambda_axes[0] = textureSampleGrad(lobe_lambda_axis0, samp, uv, ddx, ddy).xyz;
    lobe_lambda_axes[1] = textureSampleGrad(lobe_lambda_axis1, samp, uv, ddx, ddy).xyz;
    lobe_lambda_axes[2] = textureSampleGrad(lobe_lambda_axis2, samp, uv, ddx, ddy).xyz;
    lobe_lambda_axes[3] = textureSampleGrad(lobe_lambda_axis3, samp, uv, ddx, ddy).xyz;
    lobe_lambda_axes[4] = textureSampleGrad(lobe_lambda_axis4, samp, uv, ddx, ddy).xyz;
    lobe_lambda_axes[5] = textureSampleGrad(lobe_lambda_axis5, samp, uv, ddx, ddy).xyz;
    // lobe_lambda_axes[6] = textureSampleGrad(lobe_lambda_axis6, samp, uv, ddx, ddy).xyz;
    // lobe_lambda_axes[7] = textureSampleGrad(lobe_lambda_axis7, samp, uv, ddx, ddy).xyz;
    // lobe_lambda_axes[8] = textureSampleGrad(lobe_lambda_axis8, samp, uv, ddx, ddy).xyz;
    // lobe_lambda_axes[9] = textureSampleGrad(lobe_lambda_axis9, samp, uv, ddx, ddy).xyz;

    var rgb = inv_sigmoid(diffuse);
    for (var i: i32 = 0; i < 6; i++) {
        let lobe_color = inv_sigmoid(lobe_colors[i]);
        let lobe_lambda_axis = lobe_lambda_axes[i];

        let lambda = exp((lobe_lambda_axis.x * 7.5) - 2.5);
        let azimuth = 6.283 * (lobe_lambda_axis.y - 0.5);
        let elevation = 3.142 * lobe_lambda_axis.z;

        let axis = vec3<f32>(
            sin(elevation) * cos(azimuth),
            sin(elevation) * sin(azimuth),
            cos(elevation)
        );

        let gaussian_lobe = exp(-lambda * (1.0 - dot(axis, view_dir)));
        rgb += gaussian_lobe * lobe_color;
    }

    rgb = 1.0 / (1.0 + exp(-rgb));

    let blend_rgb = prev_rgb + (1.0 - prev_alpha) * alpha * rgb;
    let blend_alpha = prev_alpha + (1.0 - prev_alpha) * alpha;

    var result: FragmentOutput;
    result.color = vec4<f32>(blend_rgb, blend_alpha);
    result.frag_depth = vertex.proj_position.z;
    return result;
}
