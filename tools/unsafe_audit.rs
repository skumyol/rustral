//! Unsafe Code Audit Tool for MNR
//!
//! Scans the codebase for unsafe blocks and validates SAFETY documentation.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Result of an unsafe block audit.
#[derive(Debug, Clone)]
pub struct UnsafeBlock {
    pub file: PathBuf,
    pub line: usize,
    pub has_safety_comment: bool,
    pub context: String,
    pub reason: Option<String>,
}

/// Audit report for unsafe code.
#[derive(Debug)]
pub struct UnsafeAuditReport {
    pub total_blocks: usize,
    pub documented_blocks: usize,
    pub undocumented_blocks: usize,
    pub by_file: HashMap<PathBuf, Vec<UnsafeBlock>>,
    pub recommendations: Vec<String>,
}

impl UnsafeAuditReport {
    /// Create a new empty report.
    pub fn new() -> Self {
        Self {
            total_blocks: 0,
            documented_blocks: 0,
            undocumented_blocks: 0,
            by_file: HashMap::new(),
            recommendations: Vec::new(),
        }
    }

    /// Add an unsafe block to the report.
    pub fn add_block(&mut self, block: UnsafeBlock) {
        if block.has_safety_comment {
            self.documented_blocks += 1;
        } else {
            self.undocumented_blocks += 1;
        }
        self.total_blocks += 1;

        self.by_file
            .entry(block.file.clone())
            .or_default()
            .push(block);
    }

    /// Generate a markdown report.
    pub fn generate_report(&self) -> String {
        let mut report = String::new();

        report.push_str("# Unsafe Code Audit Report\n\n");
        report.push_str(&format!("**Total unsafe blocks:** {}\n\n", self.total_blocks));
        report.push_str(&format!("- ✅ Documented: {}\n", self.documented_blocks));
        report.push_str(&format!("- ❌ Undocumented: {}\n\n", self.undocumented_blocks));

        if self.undocumented_blocks > 0 {
            report.push_str("## Undocumented Unsafe Blocks\n\n");

            for (file, blocks) in &self.by_file {
                let undocumented: Vec<_> = blocks
                    .iter()
                    .filter(|b| !b.has_safety_comment)
                    .collect();

                if !undocumented.is_empty() {
                    report.push_str(&format!("### `{}`\n\n", file.display()));

                    for block in undocumented {
                        report.push_str(&format!(
                            "- Line {}: `{}`\n",
                            block.line,
                            block.context.trim()
                        ));
                    }
                    report.push('\n');
                }
            }
        }

        if !self.recommendations.is_empty() {
            report.push_str("## Recommendations\n\n");
            for (i, rec) in self.recommendations.iter().enumerate() {
                report.push_str(&format!("{}. {}\n", i + 1, rec));
            }
            report.push('\n');
        }

        // Summary statistics
        report.push_str("## By File Statistics\n\n");
        report.push_str("| File | Total | Documented | Coverage |\n");
        report.push_str("|------|-------|------------|----------|\n");

        let mut files: Vec<_> = self.by_file.iter().collect();
        files.sort_by_key(|(f, _)| f.as_path());

        for (file, blocks) in files {
            let total = blocks.len();
            let documented = blocks.iter().filter(|b| b.has_safety_comment).count();
            let coverage = if total > 0 {
                (documented as f64 / total as f64 * 100.0) as usize
            } else {
                0
            };

            report.push_str(&format!(
                "| `{}` | {} | {} | {}% |\n",
                file.display(),
                total,
                documented,
                coverage
            ));
        }

        report
    }

    /// Check if audit passes (all blocks documented).
    pub fn passes(&self) -> bool {
        self.undocumented_blocks == 0
    }
}

/// Audit a single Rust file for unsafe blocks.
pub fn audit_file(path: &Path) -> Vec<UnsafeBlock> {
    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };

    let mut blocks = Vec::new();
    let lines: Vec<_> = content.lines().collect();

    for (i, line) in lines.iter().enumerate() {
        let line_num = i + 1;

        // Check for unsafe block
        if line.contains("unsafe {") || line.contains("unsafe fn") {
            // Check previous 5 lines for SAFETY comment
            let start = i.saturating_sub(5);
            let context_lines = &lines[start..=i];

            let context = context_lines.join("\n");
            let has_safety = context_lines
                .iter()
                .any(|l| {
                    l.contains("SAFETY:")
                        || l.contains("// Safety")
                        || l.contains("# Safety")
                        || l.contains("/// SAFETY")
                });

            // Try to extract reason
            let reason = context_lines.iter().rev().find_map(|l| {
                if l.contains("SAFETY:") {
                    Some(l.split("SAFETY:").nth(1).unwrap_or("").trim().to_string())
                } else {
                    None
                }
            });

            blocks.push(UnsafeBlock {
                file: path.to_path_buf(),
                line: line_num,
                has_safety_comment: has_safety,
                context: line.to_string(),
                reason,
            });
        }
    }

    blocks
}

/// Run audit on a directory.
pub fn audit_directory(dir: &Path) -> UnsafeAuditReport {
    let mut report = UnsafeAuditReport::new();

    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();

            if path.is_file() && path.extension().map(|e| e == "rs").unwrap_or(false) {
                let blocks = audit_file(&path);
                for block in blocks {
                    report.add_block(block);
                }
            } else if path.is_dir() {
                // Skip target directory
                if path.file_name() != Some(std::ffi::OsStr::new("target")) {
                    let sub_report = audit_directory(&path);
                    for (file, blocks) in sub_report.by_file {
                        for block in blocks {
                            report.add_block(block);
                        }
                    }
                }
            }
        }
    }

    // Add recommendations
    if report.undocumented_blocks > 0 {
        report.recommendations.push(
            "Add SAFETY: comments to all undocumented unsafe blocks".to_string()
        );
        report.recommendations.push(
            "Explain what invariants are upheld and why the operation is safe".to_string()
        );
    }

    if report.total_blocks > 10 {
        report.recommendations.push(
            "Consider if all unsafe blocks are necessary - could safe abstractions be used?".to_string()
        );
    }

    report
}

/// Main function for the audit tool.
fn main() {
    let args: Vec<String> = std::env::args().collect();

    let target_dir = args.get(1).map(|s| s.as_str()).unwrap_or("crates");
    let path = Path::new(target_dir);

    println!("Scanning for unsafe code in: {}", path.display());
    println!();

    let report = audit_directory(path);

    // Print summary
    println!("Unsafe Code Audit Summary");
    println!("=========================");
    println!("Total unsafe blocks: {}", report.total_blocks);
    println!("  Documented: {}", report.documented_blocks);
    println!("  Undocumented: {}", report.undocumented_blocks);
    println!();

    if !report.passes() {
        println!("❌ Audit FAILED: Found undocumented unsafe blocks");
        println!();

        // Show undocumented blocks
        println!("Undocumented Blocks:");
        for (file, blocks) in &report.by_file {
            let undocumented: Vec<_> = blocks
                .iter()
                .filter(|b| !b.has_safety_comment)
                .collect();

            if !undocumented.is_empty() {
                println!("\n  {}:", file.display());
                for block in undocumented {
                    println!("    Line {}: {}", block.line, block.context.trim());
                }
            }
        }
    } else {
        println!("✅ Audit PASSED: All unsafe blocks are documented");
    }

    println!();

    // Write report to file
    let report_md = report.generate_report();
    let _ = fs::write("unsafe_audit_report.md", report_md);
    println!("Full report written to: unsafe_audit_report.md");

    // Exit with error if failed
    if !report.passes() {
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_audit_file_with_safety_comment() {
        let temp_dir = std::env::temp_dir();
        let test_file = temp_dir.join("test_unsafe.rs");

        let code = r#"
// SAFETY: We own the pointer and it's valid
unsafe { some_ffi_call() }

// No safety comment
unsafe { another_call() }
"#;

        let mut file = std::fs::File::create(&test_file).unwrap();
        file.write_all(code.as_bytes()).unwrap();

        let blocks = audit_file(&test_file);

        assert_eq!(blocks.len(), 2);
        assert!(blocks[0].has_safety_comment);
        assert!(!blocks[1].has_safety_comment);

        let _ = std::fs::remove_file(&test_file);
    }

    #[test]
    fn test_audit_report() {
        let mut report = UnsafeAuditReport::new();

        report.add_block(UnsafeBlock {
            file: PathBuf::from("test.rs"),
            line: 10,
            has_safety_comment: true,
            context: "unsafe { }".to_string(),
            reason: Some("Valid pointer".to_string()),
        });

        report.add_block(UnsafeBlock {
            file: PathBuf::from("test.rs"),
            line: 20,
            has_safety_comment: false,
            context: "unsafe { }".to_string(),
            reason: None,
        });

        assert_eq!(report.total_blocks, 2);
        assert_eq!(report.documented_blocks, 1);
        assert_eq!(report.undocumented_blocks, 1);
        assert!(!report.passes());

        let md = report.generate_report();
        assert!(md.contains("Total unsafe blocks: 2"));
        assert!(md.contains("test.rs"));
    }
}
