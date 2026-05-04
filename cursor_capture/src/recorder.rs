use crate::storage::{CursorEvent, Storage};
use rdev::{listen, Event, EventType};
use std::sync::mpsc::{self, Sender, Receiver};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use log::{info, warn, error};

/// Configuration for the recorder.
pub struct RecorderConfig {
    /// Number of events to buffer before flushing to disk.
    pub buffer_size: usize,
    /// How often to run storage maintenance (compress, cap) in seconds.
    pub maintenance_interval_secs: u64,
}

impl Default for RecorderConfig {
    fn default() -> Self {
        RecorderConfig {
            buffer_size: 5000,               // larger buffer for high-throughput
            maintenance_interval_secs: 3600,  // every hour
        }
    }
}

/// The Recorder captures mouse movement events and writes them to storage.
///
/// **Capture philosophy: per-pixel, not per-timestamp.**
///
/// We record EVERY distinct pixel position the OS reports. No time-based
/// throttling. The only filter is position deduplication — if the cursor
/// hasn't moved to a new integer pixel, we don't record.
///
/// This gives us the true continuous trajectory the cursor follows,
/// with zero spatial data loss. The OS + mouse hardware determines the
/// effective sample rate (typically 125Hz–1000Hz depending on the mouse).
///
/// Data volume estimates (8 hours active use):
///   - Standard mouse (125Hz): ~3.6M events/day ≈ 160MB raw → ~30MB compressed
///   - Gaming mouse (1000Hz): ~28M events/day ≈ 1.2GB raw → ~200MB compressed  
///   - 500MB cap → at least 2-3 weeks of continuous collection
pub struct Recorder {
    config: RecorderConfig,
    storage: Arc<Storage>,
    running: Arc<AtomicBool>,
}

impl Recorder {
    pub fn new(config: RecorderConfig, storage: Storage) -> Self {
        Recorder {
            config,
            storage: Arc::new(storage),
            running: Arc::new(AtomicBool::new(true)),
        }
    }

    /// Get a handle to the running flag (for graceful shutdown).
    pub fn running_flag(&self) -> Arc<AtomicBool> {
        self.running.clone()
    }

    /// Start recording. This blocks the calling thread.
    pub fn start(self) {
        let (tx, rx): (Sender<CursorEvent>, Receiver<CursorEvent>) = mpsc::channel();
        let running = self.running.clone();
        let storage = self.storage.clone();
        let buffer_size = self.config.buffer_size;
        let maintenance_interval = Duration::from_secs(self.config.maintenance_interval_secs);

        // Shared event counter for watchdog diagnostics
        let total_events = Arc::new(AtomicU64::new(0));
        let events_for_writer = total_events.clone();
        let events_for_watchdog = total_events.clone();

        // Writer thread: receives events from the channel and writes in batches
        let writer_running = running.clone();
        let writer_handle = thread::spawn(move || {
            let mut buffer: Vec<CursorEvent> = Vec::with_capacity(buffer_size);
            let mut last_maintenance = Instant::now();
            let mut session_events: u64 = 0;

            info!("Writer thread started");

            loop {
                // Try to receive with a timeout so we can check the running flag
                match rx.recv_timeout(Duration::from_millis(500)) {
                    Ok(event) => {
                        buffer.push(event);
                        session_events += 1;
                        events_for_writer.store(session_events, Ordering::Relaxed);

                        if buffer.len() >= buffer_size {
                            storage.write_batch(&buffer);
                            info!("Flushed {} events to disk (total session: {})",
                                  buffer.len(), session_events);
                            buffer.clear();
                        }
                    }
                    Err(mpsc::RecvTimeoutError::Timeout) => {
                        // Flush partial buffer periodically
                        if !buffer.is_empty() {
                            storage.write_batch(&buffer);
                            buffer.clear();
                        }
                    }
                    Err(mpsc::RecvTimeoutError::Disconnected) => {
                        // Channel closed, flush and exit
                        if !buffer.is_empty() {
                            storage.write_batch(&buffer);
                        }
                        break;
                    }
                }

                // Run maintenance periodically
                if last_maintenance.elapsed() >= maintenance_interval {
                    info!("Running storage maintenance...");
                    storage.maintenance();
                    last_maintenance = Instant::now();
                }

                if !writer_running.load(Ordering::Relaxed) {
                    // Final flush
                    if !buffer.is_empty() {
                        storage.write_batch(&buffer);
                    }
                    storage.flush();
                    break;
                }
            }

            info!("Writer thread exiting (total events this session: {})", session_events);
        });

        // Watchdog thread: checks if events are arriving after startup
        let watchdog_running = running.clone();
        let watchdog_handle = thread::spawn(move || {
            // Wait 10 seconds after startup, then check if any events arrived
            thread::sleep(Duration::from_secs(10));

            if !watchdog_running.load(Ordering::Relaxed) {
                return;
            }

            let count = events_for_watchdog.load(Ordering::Relaxed);
            if count == 0 {
                warn!("═══════════════════════════════════════════════════════════");
                warn!("  ⚠  NO CURSOR EVENTS RECEIVED after 10 seconds!");
                warn!("");
                #[cfg(target_os = "macos")]
                {
                    warn!("  This usually means Accessibility permission is not granted.");
                    warn!("");
                    warn!("  To fix this:");
                    warn!("  1. Open System Settings → Privacy & Security → Accessibility");
                    warn!("  2. Click the + button");
                    warn!("  3. Navigate to this binary and add it");
                    warn!("  4. Make sure the toggle is ON");
                    warn!("  5. Restart this app");
                    warn!("");
                    warn!("  If running from a terminal, you may also need to add");
                    warn!("  your terminal app (Terminal.app or iTerm2) to the list.");
                }
                #[cfg(target_os = "windows")]
                {
                    warn!("  This is unexpected on Windows. Possible causes:");
                    warn!("  - Antivirus blocking input monitoring");
                    warn!("  - Running in a sandboxed environment");
                    warn!("  Try running as Administrator.");
                }
                warn!("═══════════════════════════════════════════════════════════");
            } else {
                info!("Watchdog: {} events captured in first 10 seconds ✓", count);
            }

            // Periodic status logging (every 5 minutes)
            loop {
                thread::sleep(Duration::from_secs(300));
                if !watchdog_running.load(Ordering::Relaxed) {
                    break;
                }
                let count = events_for_watchdog.load(Ordering::Relaxed);
                info!("Status: {} total events captured this session", count);
            }
        });

        // Listener: capture mouse events on the main/listener thread
        let listener_running = running.clone();

        // Per-pixel deduplication state (no time-based throttle!)
        // We use atomic-like integers for the last pixel position to avoid mutex overhead
        // in the hot path. Using Mutex<(i32, i32)> for simplicity since the lock is
        // uncontended (only the callback thread writes).
        let last_pixel = Arc::new(std::sync::Mutex::new((i32::MIN, i32::MIN)));

        let lp_clone = last_pixel.clone();
        let lr_clone = listener_running.clone();

        info!("Starting mouse listener (per-pixel mode, buffer={})", buffer_size);
        info!("Recording EVERY distinct pixel position — zero spatial data loss");

        // Clone tx for the closure; the original tx will be dropped after listen() returns
        // to signal the writer thread that no more events are coming.
        let tx_clone = tx.clone();
        drop(tx);

        let callback = move |event: Event| {
            if !lr_clone.load(Ordering::Relaxed) {
                return;
            }

            if let EventType::MouseMove { x, y } = event.event_type {
                // Convert to integer pixels — this is our dedup key.
                // Sub-pixel differences don't matter for trajectory research.
                let px = x.round() as i32;
                let py = y.round() as i32;

                // Per-pixel deduplication: only record if the cursor moved
                // to a new integer pixel position. This is the ONLY filter.
                {
                    let mut last = lp_clone.lock().unwrap();
                    if last.0 == px && last.1 == py {
                        return; // Same pixel, skip
                    }
                    *last = (px, py);
                }

                // Create the event with high-resolution timestamp
                // Using microseconds for sub-millisecond precision at high sample rates
                let timestamp_us = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_micros() as i64;

                let cursor_event = CursorEvent {
                    x: px as f64,
                    y: py as f64,
                    t: timestamp_us,
                };

                // Send to writer thread — if channel is full/closed, just drop
                let _ = tx_clone.send(cursor_event);
            }
        };

        // rdev::listen blocks, so we run it here
        if let Err(e) = listen(callback) {
            error!("Mouse listener error: {:?}", e);
        }

        // Signal writer to stop and wait
        running.store(false, Ordering::Relaxed);
        let _ = writer_handle.join();
        let _ = watchdog_handle.join();
    }
}
