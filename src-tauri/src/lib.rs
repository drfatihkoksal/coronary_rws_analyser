use tauri::Manager;

/// Coronary RWS Analyser - Tauri Application
///
/// This is the main entry point for the Tauri application.
/// The application provides:
/// - Native file dialogs for DICOM file selection
/// - File system access for reading/writing analysis results
/// - Shell access for launching the Python backend

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            #[cfg(debug_assertions)]
            {
                let window = app.get_webview_window("main").unwrap();
                window.open_devtools();
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
