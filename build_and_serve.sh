#! /bin/sh

# wasm-pack build --dev --target web
# wasm-pack build --profiling --target web
wasm-pack build --target web

python3 -m http.server $1
