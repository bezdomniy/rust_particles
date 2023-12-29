#[cfg(not(target_arch = "wasm32"))]
fn main() {
    env_logger::init();
    let options = eframe::NativeOptions::default();

    let _ = eframe::run_native(
        "Particles",
        options,
        Box::new(|cc| Box::new(rust_particles::App::new(cc))),
    );
}

#[cfg(target_arch = "wasm32")]
fn main() {
    // Redirect `log` message to `console.log` and friends:
    eframe::WebLogger::init(log::LevelFilter::Debug).ok();

    let options = eframe::WebOptions::default();

    wasm_bindgen_futures::spawn_local(async {
        eframe::WebRunner::new()
            .start(
                "canvas",
                options,
                Box::new(|cc| Box::new(rust_particles::App::new(cc))),
            )
            .await
            .expect("failed to start eframe");
    });
}
