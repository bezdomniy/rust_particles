#[cfg(not(target_arch = "wasm32"))]
fn main() {
    tracing_subscriber::fmt::init();
    let options = eframe::NativeOptions::default();

    let _ = eframe::run_native(
        "Particles",
        options,
        Box::new(|cc| Box::new(rust_particles::App::new(cc))),
    );
}

#[cfg(target_arch = "wasm32")]
fn main() {
    // Make sure panics are logged using `console.error`.
    console_error_panic_hook::set_once();

    // Redirect tracing to console.log and friends:
    tracing_wasm::set_as_global_default();

    let options = eframe::WebOptions::default();

    wasm_bindgen_futures::spawn_local(async {
        eframe::start_web(
            "canvas",
            options,
            Box::new(|cc| Box::new(rust_particles::App::new(cc))),
        )
        .await
        .expect("failed to start eframe");
    });
}
