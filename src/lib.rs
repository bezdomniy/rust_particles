#![warn(clippy::all, rust_2018_idioms)]
mod app;

#[cfg(target_arch = "wasm32")]
pub use wasm_bindgen_rayon::init_thread_pool;

pub use app::App;
