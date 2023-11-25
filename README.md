# MobileNeRF++ Viewer

Interactive WebGL2 implementation of MobileNeRF++ inference.

Based on the [wgpu examples](https://github.com/gfx-rs/wgpu/tree/trunk/examples).

## Setup Instructions

### Step One

[Install the Rust compiler](https://rustup.rs/)

### Step Two

Install `wasm-pack`:

    cargo install wasm-pack

### Step Three

Build the viewer and launch a local server using:

    ./build_and_serve.sh

## WebGL Performance

To run without VSync limits, start chrome with:

    chromium --disable-gpu-vsync --disable-frame-rate-limit