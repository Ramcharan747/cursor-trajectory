use auto_launch::AutoLaunchBuilder;
use log::info;
use std::env;
use std::path::PathBuf;

/// Get the path to the current executable.
pub fn current_exe_path() -> PathBuf {
    env::current_exe().expect("Failed to get current executable path")
}

/// Build an AutoLaunch instance with consistent settings.
fn build_auto_launch() -> Result<auto_launch::AutoLaunch, String> {
    let exe_path = current_exe_path();
    let exe_str = exe_path.to_str().ok_or("Invalid executable path")?;

    AutoLaunchBuilder::new()
        .set_app_name("CursorCapture")
        .set_app_path(exe_str)
        .set_use_launch_agent(true)  // Use LaunchAgent on macOS (not AppleScript)
        .set_args(&["run"])
        .build()
        .map_err(|e| format!("Failed to build auto-launch config: {}", e))
}

/// Install auto-start so the app launches on user login.
pub fn install_autostart() -> Result<(), String> {
    let auto_launch = build_auto_launch()?;

    if auto_launch
        .is_enabled()
        .map_err(|e| format!("Failed to check auto-launch status: {}", e))?
    {
        info!("Auto-start is already enabled");
        return Ok(());
    }

    auto_launch
        .enable()
        .map_err(|e| format!("Failed to enable auto-start: {}", e))?;

    info!("Auto-start enabled successfully");
    Ok(())
}

/// Remove auto-start registration.
pub fn uninstall_autostart() -> Result<(), String> {
    let auto_launch = build_auto_launch()?;

    if !auto_launch
        .is_enabled()
        .map_err(|e| format!("Failed to check auto-launch status: {}", e))?
    {
        info!("Auto-start is already disabled");
        return Ok(());
    }

    auto_launch
        .disable()
        .map_err(|e| format!("Failed to disable auto-start: {}", e))?;

    info!("Auto-start disabled successfully");

    // Also remove our custom plist if it exists
    #[cfg(target_os = "macos")]
    {
        let plist_path = dirs::home_dir()
            .map(|h| h.join("Library/LaunchAgents/CursorCapture.plist"));
        if let Some(p) = plist_path {
            if p.exists() {
                // Unload first
                let _ = std::process::Command::new("launchctl")
                    .args(["unload", p.to_str().unwrap_or("")])
                    .output();
                let _ = std::fs::remove_file(&p);
                info!("Removed LaunchAgent plist");
            }
        }
    }

    Ok(())
}

/// Check if auto-start is currently enabled.
pub fn is_autostart_enabled() -> bool {
    match build_auto_launch() {
        Ok(al) => al.is_enabled().unwrap_or(false),
        Err(_) => false,
    }
}

// ─── macOS-specific functions ──────────────────────────────────────────────

/// Install a proper LaunchAgent plist with KeepAlive, logging, and environment.
#[cfg(target_os = "macos")]
pub fn install_launchd_plist() -> Result<(), String> {
    let exe_path = current_exe_path();
    let exe_str = exe_path.to_str().ok_or("Invalid executable path")?;

    let data_dir = dirs::home_dir()
        .ok_or("Cannot determine home directory")?
        .join("cursor_capture_data");

    // Ensure data directory exists
    std::fs::create_dir_all(&data_dir)
        .map_err(|e| format!("Failed to create data dir: {}", e))?;

    let log_path = data_dir.join("cursor_capture.log");

    let plist_content = format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
  <dict>
    <key>Label</key>
    <string>CursorCapture</string>
    <key>ProgramArguments</key>
    <array>
      <string>{}</string>
      <string>run</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{}</string>
    <key>StandardErrorPath</key>
    <string>{}</string>
    <key>EnvironmentVariables</key>
    <dict>
      <key>RUST_LOG</key>
      <string>info</string>
    </dict>
  </dict>
</plist>"#,
        exe_str,
        log_path.display(),
        log_path.display()
    );

    let plist_path = dirs::home_dir()
        .ok_or("Cannot determine home directory")?
        .join("Library/LaunchAgents/CursorCapture.plist");

    // Ensure LaunchAgents directory exists
    if let Some(parent) = plist_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create LaunchAgents dir: {}", e))?;
    }

    // Unload existing agent if running
    let _ = std::process::Command::new("launchctl")
        .args(["unload", plist_path.to_str().unwrap_or("")])
        .output();

    std::fs::write(&plist_path, plist_content)
        .map_err(|e| format!("Failed to write plist: {}", e))?;

    info!("LaunchAgent plist installed at: {}", plist_path.display());
    Ok(())
}

/// Open macOS Accessibility settings so the user can grant permission.
#[cfg(target_os = "macos")]
pub fn open_accessibility_settings() {
    let _ = std::process::Command::new("open")
        .arg("x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility")
        .spawn();
}

/// Load the LaunchAgent so it runs as a daemon managed by launchd.
#[cfg(target_os = "macos")]
pub fn load_launchd_agent() -> Result<(), String> {
    let plist_path = dirs::home_dir()
        .ok_or("Cannot determine home directory")?
        .join("Library/LaunchAgents/CursorCapture.plist");

    if !plist_path.exists() {
        return Err("LaunchAgent plist not found. Run install first.".into());
    }

    let output = std::process::Command::new("launchctl")
        .args(["load", plist_path.to_str().unwrap_or("")])
        .output()
        .map_err(|e| format!("Failed to run launchctl: {}", e))?;

    if output.status.success() {
        info!("LaunchAgent loaded successfully");
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        // "already loaded" is not an error for us
        if stderr.contains("already loaded") || stderr.contains("service already loaded") {
            info!("LaunchAgent was already loaded");
            Ok(())
        } else {
            Err(format!("launchctl load failed: {}", stderr))
        }
    }
}

/// Print platform-specific status information.
pub fn print_status() {
    let exe_path = current_exe_path();
    println!("╔══════════════════════════════════════════════════╗");
    println!("║            CursorCapture — Status               ║");
    println!("╚══════════════════════════════════════════════════╝");
    println!();
    println!("  Executable:  {}", exe_path.display());
    println!(
        "  Auto-start:  {}",
        if is_autostart_enabled() {
            "enabled ✓"
        } else {
            "disabled ✗"
        }
    );

    #[cfg(target_os = "macos")]
    {
        println!("  Platform:    macOS (LaunchAgent)");
        let plist_path = dirs::home_dir()
            .map(|h| h.join("Library/LaunchAgents/CursorCapture.plist"));
        if let Some(p) = plist_path {
            println!(
                "  LaunchAgent: {} ({})",
                p.display(),
                if p.exists() { "exists ✓" } else { "not found ✗" }
            );
        }

        // Check if daemon is running
        let running = std::process::Command::new("launchctl")
            .args(["list"])
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).contains("CursorCapture"))
            .unwrap_or(false);
        println!(
            "  Daemon:      {}",
            if running { "running ✓" } else { "not running ✗" }
        );
    }

    #[cfg(target_os = "windows")]
    {
        println!("  Platform:    Windows (Registry Run)");
    }

    let data_dir = dirs::home_dir()
        .map(|h| h.join("cursor_capture_data"))
        .unwrap_or_default();
    println!("  Data dir:    {}", data_dir.display());

    if data_dir.exists() {
        let mut total_size: u64 = 0;
        let mut file_count: u64 = 0;
        let mut latest_file: Option<String> = None;
        if let Ok(entries) = std::fs::read_dir(&data_dir) {
            for entry in entries.flatten() {
                if let Ok(meta) = entry.metadata() {
                    if meta.is_file() {
                        let name = entry.file_name().to_string_lossy().to_string();
                        if name.starts_with("cursor_") {
                            total_size += meta.len();
                            file_count += 1;
                            latest_file = Some(name);
                        }
                    }
                }
            }
        }
        println!(
            "  Data files:  {} ({:.2} MB)",
            file_count,
            total_size as f64 / (1024.0 * 1024.0)
        );
        if let Some(f) = latest_file {
            println!("  Latest:      {}", f);
        }
    } else {
        println!("  Data dir:    (not created yet)");
    }
    println!();
}
