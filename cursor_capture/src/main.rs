mod recorder;
mod storage;
mod platform;

use clap::{Parser, Subcommand};
use log::info;
use std::sync::atomic::Ordering;

use crate::recorder::{Recorder, RecorderConfig};
use crate::storage::Storage;

/// CursorCapture — Silent cursor movement recorder for trajectory research.
///
/// Records mouse movement at 60Hz to JSONL files for later analysis.
/// Designed to run as a background daemon that auto-starts on login.
#[derive(Parser)]
#[command(name = "cursor_capture")]
#[command(about = "Silent cursor movement recorder for trajectory research")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Start recording cursor movements
    Run,
    /// Install auto-start and begin recording (also the default behavior)
    Install,
    /// Remove auto-start registration and stop
    Uninstall,
    /// Show current status (auto-start, data directory, disk usage)
    Status,
}

fn main() {
    // Initialize logger — suppress for status command
    let args: Vec<String> = std::env::args().collect();
    let is_status = args.get(1).map_or(false, |a| a == "status");

    if !is_status {
        env_logger::Builder::from_env(
            env_logger::Env::default().default_filter_or("info"),
        )
        .format_timestamp_millis()
        .init();
    }

    let cli = Cli::parse();

    // Default behavior (no subcommand): smart auto-install + run
    // This makes double-clicking the binary "just work"
    match cli.command {
        None => smart_start(),
        Some(Commands::Run) => run_recorder(),
        Some(Commands::Install) => do_install(),
        Some(Commands::Uninstall) => {
            match platform::uninstall_autostart() {
                Ok(()) => {
                    println!("✓ CursorCapture auto-start removed");
                    println!("  Data files in ~/cursor_capture_data/ are preserved.");
                    println!();
                    println!("  To also stop the running instance:");

                    #[cfg(target_os = "macos")]
                    println!("    launchctl unload ~/Library/LaunchAgents/CursorCapture.plist");

                    #[cfg(target_os = "windows")]
                    println!("    Close the CursorCapture process from Task Manager");
                }
                Err(e) => {
                    eprintln!("✗ Failed to uninstall: {}", e);
                    std::process::exit(1);
                }
            }
        }
        Some(Commands::Status) => {
            platform::print_status();
        }
    }
}

/// Smart start: if not already installed, install first, then run.
/// This is the default when the binary is double-clicked with no arguments.
fn smart_start() {
    if platform::is_autostart_enabled() {
        // Already installed — just run
        run_recorder();
    } else {
        // First time — install, then run
        println!();
        println!("╔══════════════════════════════════════════════════╗");
        println!("║        CursorCapture — First Time Setup         ║");
        println!("╚══════════════════════════════════════════════════╝");
        println!();
        println!("  This app records cursor movement data for");
        println!("  trajectory research. It runs silently in the");
        println!("  background and starts automatically on login.");
        println!();
        println!("  Data is saved locally to:");
        println!("    ~/cursor_capture_data/");
        println!();
        println!("  Setting up...");
        println!();

        do_install();
    }
}

/// Install auto-start and begin recording.
fn do_install() {
    // Step 1: Register auto-start
    match platform::install_autostart() {
        Ok(()) => {
            println!("  ✓ Auto-start registered");
        }
        Err(e) => {
            eprintln!("  ✗ Failed to register auto-start: {}", e);
            eprintln!("    Continuing anyway — recording will work,");
            eprintln!("    but won't auto-start on next login.");
            eprintln!();
        }
    }

    // Step 2: Platform-specific setup
    #[cfg(target_os = "macos")]
    {
        // Install proper LaunchAgent plist with KeepAlive and logging
        match platform::install_launchd_plist() {
            Ok(()) => println!("  ✓ LaunchAgent daemon configured"),
            Err(e) => eprintln!("  ⚠ LaunchAgent setup: {}", e),
        }

        println!();
        println!("  ┌──────────────────────────────────────────────┐");
        println!("  │  macOS requires Accessibility permission     │");
        println!("  │  for cursor monitoring.                      │");
        println!("  │                                              │");
        println!("  │  A settings window will open — please add    │");
        println!("  │  and enable CursorCapture in the list.       │");
        println!("  └──────────────────────────────────────────────┘");
        println!();

        // Automatically open Accessibility settings
        platform::open_accessibility_settings();

        println!("  Waiting for permission... (press Enter once granted)");
        let mut input = String::new();
        let _ = std::io::stdin().read_line(&mut input);

        // Load the LaunchAgent so it runs as a daemon
        match platform::load_launchd_agent() {
            Ok(()) => {
                println!("  ✓ Daemon started successfully");
                println!();
                println!("╔══════════════════════════════════════════════════╗");
                println!("║  ✓ Setup complete!                              ║");
                println!("║                                                 ║");
                println!("║  CursorCapture is now running in the background ║");
                println!("║  and will auto-start on every login.            ║");
                println!("║                                                 ║");
                println!("║  You can close this window.                     ║");
                println!("╚══════════════════════════════════════════════════╝");
                println!();

                // Don't start inline — the daemon is already running
                return;
            }
            Err(e) => {
                eprintln!("  ⚠ Could not start daemon: {}", e);
                eprintln!("    Starting inline instead...");
                eprintln!();
            }
        }
    }

    #[cfg(target_os = "windows")]
    {
        // Windows doesn't need special permissions for mouse monitoring
        println!("  ✓ Startup entry created");
        println!();
        println!("╔══════════════════════════════════════════════════╗");
        println!("║  ✓ Setup complete!                              ║");
        println!("║                                                 ║");
        println!("║  CursorCapture is now running and will          ║");
        println!("║  auto-start on every login.                     ║");
        println!("║                                                 ║");
        println!("║  You can close this window.                     ║");
        println!("╚══════════════════════════════════════════════════╝");
        println!();
    }

    // Fall through to inline recording
    run_recorder();
}

fn run_recorder() {
    info!("CursorCapture v{} starting", env!("CARGO_PKG_VERSION"));

    let config = RecorderConfig::default();
    let storage = Storage::new(500); // 500 MB cap

    let recorder = Recorder::new(config, storage);
    let running = recorder.running_flag();

    // Set up graceful shutdown on Ctrl+C / SIGTERM
    let shutdown_flag = running.clone();
    ctrlc::set_handler(move || {
        println!();
        println!("Shutting down... flushing data to disk.");
        shutdown_flag.store(false, Ordering::Relaxed);
    })
    .expect("Failed to set Ctrl+C handler");

    info!("Recording cursor movements. Press Ctrl+C to stop.");

    // This blocks until shutdown
    recorder.start();

    info!("CursorCapture stopped cleanly");
}
