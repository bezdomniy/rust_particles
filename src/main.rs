#[cfg(not(target_arch = "wasm32"))]
fn main() {
    env_logger::init();
    let options = eframe::NativeOptions::default();

    eframe::run_native(
        "Particles",
        options,
        Box::new(|cc| Box::new(rust_particles::App::new(cc))),
    );
}

#[cfg(target_arch = "wasm32")]
fn main() {
    console_error_panic_hook::set_once();

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
