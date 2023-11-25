use std::{borrow::Cow, f32::consts, io::BufReader, iter, mem, ops::Deref, num::NonZeroU32};

use bytemuck::{Pod, Zeroable};
use image::DynamicImage;
use wasm_bindgen_futures::JsFuture;
use wgpu::util::DeviceExt;

use wasm_bindgen::prelude::*;

mod scene;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GlobalUniforms {
    proj: [[f32; 4]; 4],
    cam_pos: [f32; 4],
}

struct Pass {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    uniform_buf: Option<wgpu::Buffer>,
}

fn rotate_vec(vec: glam::Vec3, delta_lat: f32, delta_long: f32) -> glam::Vec3 {
    let length = vec.length();
    let lat = (vec.z / length).asin() + delta_lat;
    let long = (vec.y / length).atan2(vec.x / length) + delta_long;
    glam::Vec3::new(
        length * lat.cos() * long.cos(),
        length * lat.cos() * long.sin(),
        length * lat.sin(),
    )
}

struct Resources {
    models: Vec<tobj::Model>,
    textures: Vec<image::DynamicImage>,
}

async fn load_mesh() -> Vec<tobj::Model> {
    let mut opts = web_sys::RequestInit::new();
    opts.method("GET");
    opts.mode(web_sys::RequestMode::Cors);

    let url = "/renderingsmipnerf8k/data/mesh_segmentation.obj";
    let request = web_sys::Request::new_with_str_and_init(url, &opts).unwrap();

    let window = web_sys::window().unwrap();
    let resp_value = JsFuture::from(window.fetch_with_request(&request))
        .await
        .unwrap();
    let resp: web_sys::Response = resp_value.dyn_into().unwrap();

    let text = JsFuture::from(resp.text().unwrap()).await.unwrap();
    let text = text.as_string().unwrap();

    tobj::load_obj_buf_async(
        &mut BufReader::new(text.as_bytes()),
        &tobj::LoadOptions {
            single_index: true,
            triangulate: true,
            ..Default::default()
        },
        |_| async { Err(tobj::LoadError::GenericFailure) },
    )
    .await
    .unwrap()
    .0
}

async fn load_textures() -> Vec<image::DynamicImage> {
    let mut opts = web_sys::RequestInit::new();
    opts.method("GET");
    opts.mode(web_sys::RequestMode::Cors);

    // let urls = vec!["/color.png", "/alpha.png"];
    let urls = vec![
        "/renderingsmipnerf8k/data/diffuse.png",
        "/renderingsmipnerf8k/data/alpha.png",
        "/renderingsmipnerf8k/data/color_0.png",
        "/renderingsmipnerf8k/data/color_1.png",
        "/renderingsmipnerf8k/data/color_2.png",
        "/renderingsmipnerf8k/data/color_3.png",
        "/renderingsmipnerf8k/data/color_4.png",
        "/renderingsmipnerf8k/data/color_5.png",
        // "/color_6.png",
        // "/color_7.png",
        // "/color_8.png",
        // "/color_9.png",
        "/renderingsmipnerf8k/data/lambda_axis_0.png",
        "/renderingsmipnerf8k/data/lambda_axis_1.png",
        "/renderingsmipnerf8k/data/lambda_axis_2.png",
        "/renderingsmipnerf8k/data/lambda_axis_3.png",
        "/renderingsmipnerf8k/data/lambda_axis_4.png",
        "/renderingsmipnerf8k/data/lambda_axis_5.png",
        // "/lambda_axis_6.png",
        // "/lambda_axis_7.png",
        // "/lambda_axis_8.png",
        // "/lambda_axis_9.png",
    ];

    let mut requests = Vec::new();
    for url in urls {
        let request = web_sys::Request::new_with_str_and_init(url, &opts).unwrap();
        requests.push(request);
    }

    let window = web_sys::window().unwrap();
    let mut responses = Vec::new();
    for request in requests {
        let resp_value = JsFuture::from(window.fetch_with_request(&request))
            .await
            .unwrap();
        let resp: web_sys::Response = resp_value.dyn_into().unwrap();
        responses.push(resp);
    }

    let mut textures = Vec::new();
    for resp in responses {
        let array_buffer = JsFuture::from(resp.array_buffer().unwrap()).await.unwrap();
        let array_buffer: js_sys::ArrayBuffer = array_buffer.dyn_into().unwrap();
        let array_buffer: Vec<u8> = js_sys::Uint8Array::new(&array_buffer).to_vec();
        textures.push(image::load_from_memory(&array_buffer).unwrap());
    }

    textures
}

async fn load_resources_async() -> Resources {
    let models_future = load_mesh();
    let textures_future = load_textures();
    Resources {
        models: models_future.await,
        textures: textures_future.await,
    }
}

struct TexAlloc {
    tex: wgpu::Texture,
    view: wgpu::TextureView,
}

struct MobileNeRFScene {
    camera_target: glam::Vec3,
    camera_pos: glam::Vec3,
    front_pass: Pass,
    blit_pass: Pass,
    front_depth: TexAlloc,
    front_pseudo_depth: TexAlloc,
    back_pseudo_depth: TexAlloc,
    front_rgba: TexAlloc,
    back_rgba: TexAlloc,
    vertex_buf: wgpu::Buffer,
    index_buf: wgpu::Buffer,
    index_count: usize,
    width: u32,
    height: u32,
    last_render: f64,
    fps_interval: f64,
    fps_count: u32,
    hovered: bool,
    dragging: bool,
    mouse_x: f32,
    mouse_y: f32,
}

impl MobileNeRFScene {
    const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

    fn generate_matrix(eye: glam::Vec3, aspect_ratio: f32) -> glam::Mat4 {
        let projection = glam::Mat4::perspective_rh(consts::FRAC_PI_4, aspect_ratio, 0.01, 50.0);
        let view = glam::Mat4::look_at_rh(eye, glam::Vec3::new(0f32, 0.0, 0.0), glam::Vec3::Z);
        projection * view
    }

    fn create_texture(
        size: wgpu::Extent3d,
        format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
        device: &wgpu::Device,
        // is_array: bool,
    ) -> TexAlloc {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            size,
            mip_level_count: 1,
            sample_count: 1,
            // dimension: if is_array {wgpu::TextureDimension::D3} else {wgpu::TextureDimension::D2},
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            label: None,
            view_formats: &[],
        });

        TexAlloc {
            view: texture.create_view(&wgpu::TextureViewDescriptor::default()),
            tex: texture,
        }
    }

    fn upload_texture<P: image::Pixel, Scalar>(
        im: image::ImageBuffer<P, Vec<Scalar>>,
        format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> TexAlloc
    where
        Vec<Scalar>: Deref<Target = [P::Subpixel]>,
        P::Subpixel: Pod,
    {
        let size = wgpu::Extent3d {
            width: im.width(),
            height: im.height(),
            depth_or_array_layers: 1,
        };

        let texture = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                label: None,
                view_formats: &[],
            },
            bytemuck::cast_slice(im.as_raw()),
        );

        TexAlloc {
            view: texture.create_view(&wgpu::TextureViewDescriptor::default()),
            tex: texture,
        }
    }
}

impl scene::Scene for MobileNeRFScene {
    fn optional_features() -> wgpu::Features {
        wgpu::Features::default()
    }

    fn init(
        config: &wgpu::SurfaceConfiguration,
        _adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        resources: Resources,
    ) -> Self {
        // Create the vertex and index buffers

        if resources.models.len() > 1 {
            log::warn!("OBJ contained multiple models; using only the first.");
        }
        let model = resources.models.into_iter().take(1).next().unwrap();
        assert!(model.mesh.positions.len() / 3 == model.mesh.texcoords.len() / 2);
        let mut buf_data = Vec::<f32>::new();
        for i in 0..model.mesh.positions.len() / 3 {
            buf_data.push(model.mesh.positions[i * 3]);
            buf_data.push(model.mesh.positions[i * 3 + 1]);
            buf_data.push(model.mesh.positions[i * 3 + 2]);
            buf_data.push(model.mesh.texcoords[i * 2]);
            buf_data.push(model.mesh.texcoords[i * 2 + 1]);
        }

        let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&buf_data),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&model.mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        let index_count = model.mesh.indices.len();

        let vertex_attr = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x2];
        let vb_desc = wgpu::VertexBufferLayout {
            array_stride: 20 as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &vertex_attr,
        };

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
        });

        let viewport_size = wgpu::Extent3d {
            width: config.width * 2,
            height: config.height * 2,
            depth_or_array_layers: 1,
        };

        let front_depth = Self::create_texture(
            viewport_size,
            Self::DEPTH_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
            device,
        );
        let front_pseudo_depth = Self::create_texture(
            viewport_size,
            wgpu::TextureFormat::R32Float,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            device,
        );
        let back_pseudo_depth = Self::create_texture(
            viewport_size,
            wgpu::TextureFormat::R32Float,
            wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST,
            device,
        );
        let front_rgba = Self::create_texture(
            viewport_size,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            device,
        );
        let back_rgba = Self::create_texture(
            viewport_size,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST,
            device,
        );

        let mut textures = resources.textures.into_iter();
        let color_texture = textures.next().unwrap();
        let alpha_texture = textures.next().unwrap();
        let mut lobe_color_textures: Vec<image::RgbaImage> = textures.map(|t| t.into_rgba8()).collect();
        let lambda_axis_textures: Vec<image::RgbaImage> = lobe_color_textures.drain(
            lobe_color_textures.len() / 2..lobe_color_textures.len()).collect();
        let num_lobes = lambda_axis_textures.len();

        let color_texture = Self::upload_texture(
            color_texture.into_rgba8(),
            wgpu::TextureFormat::Rgba8UnormSrgb,
            wgpu::TextureUsages::TEXTURE_BINDING,
            device,
            queue,
        );

        let alpha_texture = Self::upload_texture(
            alpha_texture.into_luma8(),
            wgpu::TextureFormat::R8Unorm,
            wgpu::TextureUsages::TEXTURE_BINDING,
            device,
            queue,
        );

        let lobe_color_textures: Vec<TexAlloc> = {
            lobe_color_textures.into_iter().map(|t| {
                Self::upload_texture(
                    t,
                    wgpu::TextureFormat::Rgba8UnormSrgb,
                    wgpu::TextureUsages::TEXTURE_BINDING,
                    device,
                    queue,
                )
            }).collect()
        };

        let lambda_axis_textures: Vec<TexAlloc> = {
            lambda_axis_textures.into_iter().map(|t| {
                Self::upload_texture(
                    t,
                    wgpu::TextureFormat::Rgba8Unorm,
                    wgpu::TextureUsages::TEXTURE_BINDING,
                    device,
                    queue,
                )
            }).collect()
        };

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: None,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Create pipeline layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0, // global
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            mem::size_of::<GlobalUniforms>() as _
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },

                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },

                wgpu::BindGroupLayoutEntry {
                    binding: 11,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // wgpu::BindGroupLayoutEntry {
                //     binding: 12,
                //     visibility: wgpu::ShaderStages::FRAGMENT,
                //     ty: wgpu::BindingType::Texture {
                //         multisampled: false,
                //         sample_type: wgpu::TextureSampleType::Float { filterable: true },
                //         view_dimension: wgpu::TextureViewDimension::D2,
                //     },
                //     count: None,
                // },
                // wgpu::BindGroupLayoutEntry {
                //     binding: 13,
                //     visibility: wgpu::ShaderStages::FRAGMENT,
                //     ty: wgpu::BindingType::Texture {
                //         multisampled: false,
                //         sample_type: wgpu::TextureSampleType::Float { filterable: true },
                //         view_dimension: wgpu::TextureViewDimension::D2,
                //     },
                //     count: None,
                // },
                // wgpu::BindGroupLayoutEntry {
                //     binding: 14,
                //     visibility: wgpu::ShaderStages::FRAGMENT,
                //     ty: wgpu::BindingType::Texture {
                //         multisampled: false,
                //         sample_type: wgpu::TextureSampleType::Float { filterable: true },
                //         view_dimension: wgpu::TextureViewDimension::D2,
                //     },
                //     count: None,
                // },
                // wgpu::BindGroupLayoutEntry {
                //     binding: 15,
                //     visibility: wgpu::ShaderStages::FRAGMENT,
                //     ty: wgpu::BindingType::Texture {
                //         multisampled: false,
                //         sample_type: wgpu::TextureSampleType::Float { filterable: true },
                //         view_dimension: wgpu::TextureViewDimension::D2,
                //     },
                //     count: None,
                // },
                wgpu::BindGroupLayoutEntry {
                    binding: 16,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 17,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 18,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 19,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 20,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 21,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // wgpu::BindGroupLayoutEntry {
                //     binding: 22,
                //     visibility: wgpu::ShaderStages::FRAGMENT,
                //     ty: wgpu::BindingType::Texture {
                //         multisampled: false,
                //         sample_type: wgpu::TextureSampleType::Float { filterable: true },
                //         view_dimension: wgpu::TextureViewDimension::D2,
                //     },
                //     count: None,
                // },
                // wgpu::BindGroupLayoutEntry {
                //     binding: 23,
                //     visibility: wgpu::ShaderStages::FRAGMENT,
                //     ty: wgpu::BindingType::Texture {
                //         multisampled: false,
                //         sample_type: wgpu::TextureSampleType::Float { filterable: true },
                //         view_dimension: wgpu::TextureViewDimension::D2,
                //     },
                //     count: None,
                // },
                // wgpu::BindGroupLayoutEntry {
                //     binding: 24,
                //     visibility: wgpu::ShaderStages::FRAGMENT,
                //     ty: wgpu::BindingType::Texture {
                //         multisampled: false,
                //         sample_type: wgpu::TextureSampleType::Float { filterable: true },
                //         view_dimension: wgpu::TextureViewDimension::D2,
                //     },
                //     count: None,
                // },
                // wgpu::BindGroupLayoutEntry {
                //     binding: 25,
                //     visibility: wgpu::ShaderStages::FRAGMENT,
                //     ty: wgpu::BindingType::Texture {
                //         multisampled: false,
                //         sample_type: wgpu::TextureSampleType::Float { filterable: true },
                //         view_dimension: wgpu::TextureViewDimension::D2,
                //     },
                //     count: None,
                // },

            ],
            label: None,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("main"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let mx_total = Self::generate_matrix(
            glam::Vec3::new(3.0f32, -6.0, 3.0),
            config.width as f32 / config.height as f32,
        );
        let forward_uniforms = GlobalUniforms {
            proj: mx_total.to_cols_array_2d(),
            cam_pos: [3.0f32, -6.0, 3.0, 0.0],
        };

        let color_target_state: wgpu::ColorTargetState = config.view_formats[0].into();

        let pseudo_depth_target_state = wgpu::ColorTargetState {
            format: wgpu::TextureFormat::R32Float,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        };

        let front_pass = {
            let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Uniform Buffer"),
                contents: bytemuck::bytes_of(&forward_uniforms),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

            // Create bind group
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&back_pseudo_depth.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&back_rgba.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&color_texture.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(&alpha_texture.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::TextureView(&lobe_color_textures[0].view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: wgpu::BindingResource::TextureView(&lobe_color_textures[1].view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: wgpu::BindingResource::TextureView(&lobe_color_textures[2].view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: wgpu::BindingResource::TextureView(&lobe_color_textures[3].view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 10,
                        resource: wgpu::BindingResource::TextureView(&lobe_color_textures[4].view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 11,
                        resource: wgpu::BindingResource::TextureView(&lobe_color_textures[5].view),
                    },
                    // wgpu::BindGroupEntry {
                    //     binding: 12,
                    //     resource: wgpu::BindingResource::TextureView(&lobe_color_textures[6].view),
                    // },
                    // wgpu::BindGroupEntry {
                    //     binding: 13,
                    //     resource: wgpu::BindingResource::TextureView(&lobe_color_textures[7].view),
                    // },
                    // wgpu::BindGroupEntry {
                    //     binding: 14,
                    //     resource: wgpu::BindingResource::TextureView(&lobe_color_textures[8].view),
                    // },
                    // wgpu::BindGroupEntry {
                    //     binding: 15,
                    //     resource: wgpu::BindingResource::TextureView(&lobe_color_textures[9].view),
                    // },
                    wgpu::BindGroupEntry {
                        binding: 16,
                        resource: wgpu::BindingResource::TextureView(&lambda_axis_textures[0].view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 17,
                        resource: wgpu::BindingResource::TextureView(&lambda_axis_textures[1].view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 18,
                        resource: wgpu::BindingResource::TextureView(&lambda_axis_textures[2].view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 19,
                        resource: wgpu::BindingResource::TextureView(&lambda_axis_textures[3].view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 20,
                        resource: wgpu::BindingResource::TextureView(&lambda_axis_textures[4].view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 21,
                        resource: wgpu::BindingResource::TextureView(&lambda_axis_textures[5].view),
                    },
                    // wgpu::BindGroupEntry {
                    //     binding: 22,
                    //     resource: wgpu::BindingResource::TextureView(&lambda_axis_textures[6].view),
                    // },
                    // wgpu::BindGroupEntry {
                    //     binding: 23,
                    //     resource: wgpu::BindingResource::TextureView(&lambda_axis_textures[7].view),
                    // },
                    // wgpu::BindGroupEntry {
                    //     binding: 24,
                    //     resource: wgpu::BindingResource::TextureView(&lambda_axis_textures[8].view),
                    // },
                    // wgpu::BindGroupEntry {
                    //     binding: 25,
                    //     resource: wgpu::BindingResource::TextureView(&lambda_axis_textures[9].view),
                    // },
                ],
                label: None,
            });

            // Create the render pipeline
            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("main"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vs_main",
                    buffers: &[vb_desc.clone()],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: "fs_main",
                    targets: &[
                        Some(color_target_state.clone()),
                        Some(pseudo_depth_target_state.clone()),
                    ],
                }),
                primitive: wgpu::PrimitiveState {
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: Self::DEPTH_FORMAT,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
            });

            Pass {
                pipeline,
                bind_group,
                uniform_buf: Some(uniform_buf),
            }
        };

        let blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("blit.wgsl"))),
        });

        let blit_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                }],
                label: None,
            });

        let blit_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &blit_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&front_rgba.view),
            }],
            label: None,
        });
        let blit_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("blit"),
            bind_group_layouts: &[&blit_bind_group_layout],
            push_constant_ranges: &[],
        });
        let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("blit"),
            layout: Some(&blit_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &blit_shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &blit_shader,
                entry_point: "fs_main",
                targets: &[Some(config.view_formats[0].into())],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let blit_pass = Pass {
            pipeline: blit_pipeline,
            bind_group: blit_bind_group,
            uniform_buf: None,
        };

        MobileNeRFScene {
            front_pass,
            blit_pass,
            front_depth,
            front_pseudo_depth,
            back_pseudo_depth,
            front_rgba,
            back_rgba,
            vertex_buf,
            index_buf,
            index_count,
            camera_target: glam::Vec3::new(1.0f32, -6.0, 3.0),
            camera_pos: glam::Vec3::new(1.0f32, -6.0, 3.0),
            width: config.width,
            height: config.height,
            last_render: web_sys::window().unwrap().performance().unwrap().now(),
            fps_interval: 0.0,
            fps_count: 0,
            hovered: false,
            dragging: false,
            mouse_x: 0.0,
            mouse_y: 0.0,
        }
    }

    fn update(&mut self, event: winit::event::WindowEvent) {
        match event {
            winit::event::WindowEvent::CursorMoved {
                device_id: _,
                position,
                ..
            } => {
                let delta_x = position.x as f32 - self.mouse_x;
                let delta_y = position.y as f32 - self.mouse_y;
                self.mouse_x = position.x as f32;
                self.mouse_y = position.y as f32;

                if self.dragging {
                    self.camera_target =
                        rotate_vec(self.camera_target, delta_y * 0.003, -delta_x * 0.003);
                }
            }
            winit::event::WindowEvent::CursorEntered { device_id: _ } => {
                self.hovered = true;
            }
            winit::event::WindowEvent::CursorLeft { device_id: _ } => {
                self.hovered = false;
            }
            winit::event::WindowEvent::MouseWheel {
                device_id: _,
                delta,
                phase: _,
                ..
            } => match delta {
                winit::event::MouseScrollDelta::LineDelta(_h, _v) => {
                    web_sys::console::log_1(&"SCROLL NOT IMPLEMENTED FOR THIS DEVICE".into());
                }
                winit::event::MouseScrollDelta::PixelDelta(delta) => {
                    let mult = 2.0f32.powf(-delta.y as f32 * 0.001);
                    self.camera_target *= mult;
                }
            },
            winit::event::WindowEvent::MouseInput {
                device_id: _,
                state,
                button,
                ..
            } => match button {
                winit::event::MouseButton::Left => {
                    self.dragging = state == winit::event::ElementState::Pressed;
                }
                _ => {}
            },
            _ => {}
        }
    }

    fn resize(
        &mut self,
        config: &wgpu::SurfaceConfiguration,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
    ) {
        self.width = config.width;
        self.height = config.height;

        let viewport_size = wgpu::Extent3d {
            width: config.width * 2,
            height: config.height * 2,
            depth_or_array_layers: 1,
        };

        self.front_depth = Self::create_texture(
            viewport_size,
            Self::DEPTH_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
            device,
        );
        self.front_pseudo_depth = Self::create_texture(
            viewport_size,
            wgpu::TextureFormat::R32Float,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            device,
        );
        self.back_pseudo_depth = Self::create_texture(
            viewport_size,
            wgpu::TextureFormat::R32Float,
            wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST,
            device,
        );
        self.front_rgba = Self::create_texture(
            viewport_size,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            device,
        );
        self.back_rgba = Self::create_texture(
            viewport_size,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST,
            device,
        );
    }

    fn render(
        &mut self,
        view: &wgpu::TextureView,
        _view_tex: &wgpu::Texture,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _spawner: &scene::Spawner,
    ) {
        let now = web_sys::window().unwrap().performance().unwrap().now();
        let dt = (now - self.last_render) / 1000.0;
        self.last_render = now;

        self.fps_count += 1;
        self.fps_interval += dt;
        if self.fps_interval > 2.0 {
            let fps = self.fps_count as f64 / self.fps_interval;
            let counter_elem = web_sys::window()
                .unwrap()
                .document()
                .unwrap()
                .get_element_by_id("fps-value")
                .unwrap();
            counter_elem.set_inner_html(&format!("{:.0}", fps));
            self.fps_interval = 0.0;
            self.fps_count = 0;
        }

        if !self.hovered {
            self.camera_target = rotate_vec(self.camera_target, 0.0, (dt * 0.1) as f32);
        }

        let cam_delta = self.camera_target - self.camera_pos;
        let start_len = self.camera_pos.length();
        let end_len = self.camera_target.length();
        let update_frac = 1.0f32 - 2.0f32.powf((-dt * 10.0) as f32);
        self.camera_pos += cam_delta * update_frac;
        let update_len = (end_len - start_len) * update_frac + start_len;
        self.camera_pos = self.camera_pos.normalize() * update_len;

        // update view-projection matrix
        let mx_total =
            Self::generate_matrix(self.camera_pos, self.width as f32 / self.height as f32);
        let mx_ref: &[f32; 16] = mx_total.as_ref();
        queue.write_buffer(
            self.front_pass.uniform_buf.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(mx_ref),
        );
        let pos = [self.camera_pos.x, self.camera_pos.y, self.camera_pos.z, 0.0];
        queue.write_buffer(
            self.front_pass.uniform_buf.as_ref().unwrap(),
            16 * 4,
            bytemuck::cast_slice(&pos),
        );

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        // forward pass
        encoder.push_debug_group("forward rendering pass");

        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    view: &self.front_pseudo_depth.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: true,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &self.back_pseudo_depth.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: true,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &self.front_rgba.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: true,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &self.back_rgba.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: true,
                    },
                }),
            ],
            depth_stencil_attachment: None,
        });

        let num_passes = 8;
        for i in 0..num_passes {
            let color_target_view = &self.front_rgba.view;
            {
                let mut front_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
                    color_attachments: &[
                        Some(wgpu::RenderPassColorAttachment {
                            view: color_target_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: true,
                            },
                        }),
                        Some(wgpu::RenderPassColorAttachment {
                            view: &self.front_pseudo_depth.view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: true,
                            },
                        }),
                    ],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.front_depth.view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: false,
                        }),
                        stencil_ops: None,
                    }),
                });
                front_pass.set_pipeline(&self.front_pass.pipeline);
                front_pass.set_bind_group(0, &self.front_pass.bind_group, &[]);

                front_pass.set_index_buffer(self.index_buf.slice(..), wgpu::IndexFormat::Uint32);
                front_pass.set_vertex_buffer(0, self.vertex_buf.slice(..));
                front_pass.draw_indexed(0..self.index_count as u32, 0, 0..1);
            }

            let blit_target_view = if i == num_passes - 1 {
                view
            } else {
                &self.back_rgba.view
            };

            if i == num_passes - 1 {
                let mut blit_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: blit_target_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: true,
                        },
                    })],
                    depth_stencil_attachment: None,
                });
                blit_pass.set_pipeline(&self.blit_pass.pipeline);
                blit_pass.set_bind_group(0, &self.blit_pass.bind_group, &[]);

                blit_pass.draw(0..4, 0..1);
                // encoder.copy_texture_to_texture(
                //     self.front_rgba.tex.as_image_copy(),
                //     view_tex.as_image_copy(),
                //     view_tex.size());
            } else {
                encoder.copy_texture_to_texture(
                    self.front_rgba.tex.as_image_copy(),
                    self.back_rgba.tex.as_image_copy(),
                    self.back_rgba.tex.size(),
                );
                encoder.copy_texture_to_texture(
                    self.front_pseudo_depth.tex.as_image_copy(),
                    self.back_pseudo_depth.tex.as_image_copy(),
                    self.back_pseudo_depth.tex.size(),
                );
            }
        }
        encoder.pop_debug_group();

        queue.submit(iter::once(encoder.finish()));
    }

    type ResourceType = Resources;

    fn load_resources() -> Box<dyn std::future::Future<Output = Self::ResourceType> + Unpin> {
        Box::new(Box::pin(load_resources_async()))
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub fn run() {
    scene::run::<MobileNeRFScene>("shadow");
}
