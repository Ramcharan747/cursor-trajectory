use serde::Serialize;
use std::io::{BufWriter, Write};
use std::fs::{self, File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use chrono::{Local, Timelike, Datelike};
use flate2::write::GzEncoder;
use flate2::Compression;
use log::{info, warn, error};

/// A single cursor event ready for serialization.
#[derive(Serialize, Clone, Debug)]
pub struct CursorEvent {
    pub x: f64,
    pub y: f64,
    /// Milliseconds since Unix epoch
    pub t: i64,
}

/// Manages writing cursor events to JSONL files with hourly rotation,
/// background compression of old files, and storage cap enforcement.
pub struct Storage {
    data_dir: PathBuf,
    current_file: Mutex<Option<CurrentFile>>,
    max_storage_bytes: u64,
}

struct CurrentFile {
    writer: BufWriter<File>,
    hour_key: String,
    events_written: u64,
}

impl Storage {
    /// Create a new Storage instance. Creates the data directory if it doesn't exist.
    pub fn new(max_storage_mb: u64) -> Self {
        let data_dir = Self::default_data_dir();
        fs::create_dir_all(&data_dir).expect("Failed to create data directory");
        info!("Data directory: {}", data_dir.display());

        Storage {
            data_dir,
            current_file: Mutex::new(None),
            max_storage_bytes: max_storage_mb * 1024 * 1024,
        }
    }

    /// Returns the platform-appropriate data directory.
    fn default_data_dir() -> PathBuf {
        let home = dirs::home_dir().expect("Cannot determine home directory");
        home.join("cursor_capture_data")
    }

    /// Returns the data directory path.
    #[allow(dead_code)]
    pub fn data_dir(&self) -> &Path {
        &self.data_dir
    }

    /// Generate the hour key for file naming, e.g. "2026-05-04_14"
    fn hour_key() -> String {
        let now = Local::now();
        format!(
            "{:04}-{:02}-{:02}_{:02}",
            now.year(),
            now.month(),
            now.day(),
            now.hour()
        )
    }

    /// Get the filename for a given hour key.
    fn filename_for_hour(&self, hour_key: &str) -> PathBuf {
        self.data_dir.join(format!("cursor_{}.jsonl", hour_key))
    }

    /// Write a batch of events to the current file, rotating if the hour has changed.
    pub fn write_batch(&self, events: &[CursorEvent]) {
        if events.is_empty() {
            return;
        }

        let current_hour = Self::hour_key();
        let mut file_guard = self.current_file.lock().unwrap();

        // Check if we need to rotate
        let needs_new_file = match &*file_guard {
            None => true,
            Some(cf) => cf.hour_key != current_hour,
        };

        if needs_new_file {
            // Close old file
            if let Some(mut old) = file_guard.take() {
                let _ = old.writer.flush();
                info!(
                    "Rotated file for hour {}, wrote {} events",
                    old.hour_key, old.events_written
                );
            }

            // Open new file
            let path = self.filename_for_hour(&current_hour);
            let file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .expect("Failed to open data file");

            *file_guard = Some(CurrentFile {
                writer: BufWriter::with_capacity(64 * 1024, file),
                hour_key: current_hour,
                events_written: 0,
            });

            info!("Opened new data file: {}", path.display());
        }

        // Write events
        if let Some(ref mut cf) = *file_guard {
            for event in events {
                if let Ok(line) = serde_json::to_string(event) {
                    let _ = writeln!(cf.writer, "{}", line);
                    cf.events_written += 1;
                }
            }
            let _ = cf.writer.flush();
        }
    }

    /// Flush any buffered data to disk.
    pub fn flush(&self) {
        let mut file_guard = self.current_file.lock().unwrap();
        if let Some(ref mut cf) = *file_guard {
            let _ = cf.writer.flush();
        }
    }

    /// Compress JSONL files older than `age_hours` hours.
    pub fn compress_old_files(&self, age_hours: i64) {
        let cutoff = Local::now() - chrono::Duration::hours(age_hours);
        let entries = match fs::read_dir(&self.data_dir) {
            Ok(e) => e,
            Err(_) => return,
        };

        for entry in entries.flatten() {
            let path = entry.path();
            let name = match path.file_name().and_then(|n| n.to_str()) {
                Some(n) => n.to_string(),
                None => continue,
            };

            // Only process uncompressed JSONL files
            if !name.starts_with("cursor_") || !name.ends_with(".jsonl") {
                continue;
            }

            // Parse the hour key from filename: cursor_2026-05-04_14.jsonl
            let hour_key = name
                .strip_prefix("cursor_")
                .and_then(|s| s.strip_suffix(".jsonl"))
                .unwrap_or("");

            // Parse date from hour key
            let file_time = match chrono::NaiveDateTime::parse_from_str(
                &format!("{}:00:00", hour_key.replace('_', " ")),
                "%Y-%m-%d %H:%M:%S",
            ) {
                Ok(t) => t.and_local_timezone(Local).single(),
                Err(_) => continue,
            };

            if let Some(ft) = file_time {
                if ft < cutoff {
                    if let Err(e) = self.gzip_file(&path) {
                        warn!("Failed to compress {}: {}", path.display(), e);
                    } else {
                        info!("Compressed: {}", path.display());
                    }
                }
            }
        }
    }

    /// Gzip a single file and delete the original.
    fn gzip_file(&self, path: &Path) -> std::io::Result<()> {
        let gz_path = path.with_extension("jsonl.gz");
        let input = fs::read(path)?;
        let gz_file = File::create(&gz_path)?;
        let mut encoder = GzEncoder::new(gz_file, Compression::default());
        encoder.write_all(&input)?;
        encoder.finish()?;
        fs::remove_file(path)?;
        Ok(())
    }

    /// Enforce storage cap by removing oldest files.
    pub fn enforce_storage_cap(&self) {
        let mut files: Vec<(PathBuf, u64)> = Vec::new();
        let mut total_size: u64 = 0;

        let entries = match fs::read_dir(&self.data_dir) {
            Ok(e) => e,
            Err(_) => return,
        };

        for entry in entries.flatten() {
            let path = entry.path();
            if let Ok(meta) = path.metadata() {
                if meta.is_file() {
                    let size = meta.len();
                    total_size += size;
                    files.push((path, size));
                }
            }
        }

        if total_size <= self.max_storage_bytes {
            return;
        }

        // Sort by name (oldest first, since filenames contain timestamps)
        files.sort_by(|a, b| a.0.cmp(&b.0));

        // Only delete compressed files first
        let mut gz_files: Vec<_> = files
            .iter()
            .filter(|(p, _)| p.extension().map_or(false, |e| e == "gz"))
            .cloned()
            .collect();
        gz_files.sort_by(|a, b| a.0.cmp(&b.0));

        for (path, size) in &gz_files {
            if total_size <= self.max_storage_bytes {
                break;
            }
            if let Err(e) = fs::remove_file(path) {
                error!("Failed to delete {}: {}", path.display(), e);
            } else {
                info!("Deleted old file to free space: {}", path.display());
                total_size -= size;
            }
        }
    }

    /// Run maintenance tasks: compress old files and enforce storage cap.
    pub fn maintenance(&self) {
        self.compress_old_files(24);
        self.enforce_storage_cap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn test_cursor_event_serialization() {
        let event = CursorEvent {
            x: 100.0,
            y: 200.0,
            t: 1714857600123,
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"x\":100.0"));
        assert!(json.contains("\"y\":200.0"));
        assert!(json.contains("\"t\":1714857600123"));
    }
}
