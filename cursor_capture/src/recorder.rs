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
    /// Minimum interval between recorded events (throttle).
    /// At 60Hz this is ~16ms.
    pub min_interval_ms: u64,
    /// Number of events to buffer before flushing to disk.
    pub buffer_size: usize,
    /// How often to run storage maintenance (compress, cap) in seconds.
    pub maintenance_interval_secs: u64,
    /// Idle threshold: stop recording after this many ms of no movement.
    pub idle_threshold_ms: u64,
}

impl Default for RecorderConfig {
    fn default() -> Self {
        RecorderConfig {
            min_interval_ms: 16,   // ~60 Hz
            buffer_size: 1000,
            maintenance_interval_secs: 3600, // every hour
            idle_threshold_ms: 2000,         // 2 seconds
        }
    }
}

/// The Recorder captures mouse movement events and writes them to storage.
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
                return; // Shutting down, don't warn
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
        let min_interval = Duration::from_millis(self.config.min_interval_ms);
        let _idle_threshold = Duration::from_millis(self.config.idle_threshold_ms);
        let listener_running = running.clone();

        // State for throttling and idle detection
        let last_event_time = Arc::new(std::sync::Mutex::new(Instant::now() - min_interval));
        let last_position = Arc::new(std::sync::Mutex::new((f64::NAN, f64::NAN)));
        let idle_since = Arc::new(std::sync::Mutex::new(Option::<Instant>::None));

        let let_clone = last_event_time.clone();
        let lp_clone = last_position.clone();
        let is_clone = idle_since.clone();
        let lr_clone = listener_running.clone();

        info!("Starting mouse listener ({}Hz, buffer={})", 
              1000 / self.config.min_interval_ms, buffer_size);

        // Clone tx for the closure; the original tx will be dropped after listen() returns
        // to signal the writer thread that no more events are coming.
        let tx_clone = tx.clone();
        drop(tx);

        let callback = move |event: Event| {
            if !lr_clone.load(Ordering::Relaxed) {
                return;
            }

            if let EventType::MouseMove { x, y } = event.event_type {
                let now = Instant::now();

                // Throttle: skip if too soon since last event
                {
                    let mut last = let_clone.lock().unwrap();
                    if now.duration_since(*last) < min_interval {
                        return;
                    }
                    *last = now;
                }

                // Check if position actually changed (avoid duplicate stationary events)
                {
                    let mut last_pos = lp_clone.lock().unwrap();
                    let (lx, ly) = *last_pos;
                    if (x - lx).abs() < 0.5 && (y - ly).abs() < 0.5 {
                        // Position hasn't meaningfully changed — update idle tracker
                        let mut idle = is_clone.lock().unwrap();
                        if idle.is_none() {
                            *idle = Some(now);
                        }
                        return;
                    }
                    *last_pos = (x, y);
                }

                // Reset idle tracker since we moved
                {
                    let mut idle = is_clone.lock().unwrap();
                    *idle = None;
                }

                // Create the event with system timestamp
                let timestamp = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as i64;

                let cursor_event = CursorEvent { x, y, t: timestamp };

                // Send to writer thread — if channel is full/closed, just drop
                let _ = tx_clone.send(cursor_event);
            }
        };

        // rdev::listen blocks, so we run it here
        // If it returns an error, log it
        if let Err(e) = listen(callback) {
            error!("Mouse listener error: {:?}", e);
        }

        // Signal writer to stop and wait
        // tx_clone is moved into the closure, so when listen() returns and the
        // closure is dropped, the sender is dropped, causing the writer's recv to
        // return Disconnected.
        running.store(false, Ordering::Relaxed);
        let _ = writer_handle.join();
        let _ = watchdog_handle.join();
    }
}
