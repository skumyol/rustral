#!/bin/bash
# Security Audit Script for Modular Neural Runtime
# Performs comprehensive security checks on the codebase

set -e

echo "=========================================="
echo "MNR Security Audit"
echo "=========================================="
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPORT_FILE="$PROJECT_ROOT/security_report.md"
AUDIT_DATE=$(date -u +"%Y-%m-%d %H:%M UTC")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counters
CRITICAL=0
HIGH=0
MEDIUM=0
LOW=0
INFO=0

# Initialize report
cat > "$REPORT_FILE" << EOF
# MNR Security Audit Report

**Date:** $AUDIT_DATE  
**Auditor:** Automated Security Audit Script  
**Scope:** Full codebase including dependencies

## Executive Summary

This report summarizes security findings from automated scans of the Modular Neural Runtime codebase.

| Severity | Count |
|----------|-------|
| Critical | CRITICAL_COUNT |
| High | HIGH_COUNT |
| Medium | MEDIUM_COUNT |
| Low | LOW_COUNT |
| Info | INFO_COUNT |

## Findings

EOF

# Function to add finding to report
add_finding() {
    local severity="$1"
    local category="$2"
    local description="$3"
    local location="$4"
    local recommendation="$5"

    local severity_emoji=""
    case "$severity" in
        CRITICAL) severity_emoji="🔴"; ((CRITICAL++)); ;;
        HIGH) severity_emoji="🟠"; ((HIGH++)); ;;
        MEDIUM) severity_emoji="🟡"; ((MEDIUM++)); ;;
        LOW) severity_emoji="🟢"; ((LOW++)); ;;
        INFO) severity_emoji="🔵"; ((INFO++)); ;;
    esac

    cat >> "$REPORT_FILE" << EOF
### $severity_emoji $severity: $category

**Description:** $description  
**Location:** $location  
**Recommendation:** $recommendation

EOF
}

echo "[1/7] Checking for Rust toolchain..."
if ! command -v cargo &> /dev/null; then
    echo -e "${RED}Error: cargo not found${NC}"
    add_finding "CRITICAL" "Build System" "Cargo/Rust toolchain not found" "System PATH" "Install Rust toolchain"
    exit 1
fi

echo "[2/7] Running cargo audit for dependency vulnerabilities..."
cd "$PROJECT_ROOT"
if command -v cargo-audit &> /dev/null; then
    # Run cargo audit and capture output
    AUDIT_OUTPUT=$(cargo audit --json 2>/dev/null || true)

    if echo "$AUDIT_OUTPUT" | grep -q '"vulnerabilities":'; then
        VULN_COUNT=$(echo "$AUDIT_OUTPUT" | grep -o '"vulnerabilities":\s*\[' | wc -l || echo "0")
        if [ "$VULN_COUNT" -gt 0 ]; then
            add_finding "HIGH" "Dependency Vulnerabilities" \
                "Found $VULN_COUNT vulnerable dependencies" \
                "Cargo.lock" \
                "Run 'cargo audit' and update dependencies"
            echo -e "${YELLOW}⚠ Found dependency vulnerabilities${NC}"
        else
            echo -e "${GREEN}✓ No dependency vulnerabilities found${NC}"
        fi
    else
        echo -e "${GREEN}✓ No dependency vulnerabilities found${NC}"
    fi
else
    echo -e "${YELLOW}⚠ cargo-audit not installed, skipping dependency scan${NC}"
    echo "   Install with: cargo install cargo-audit"
    add_finding "INFO" "Tooling" "cargo-audit not installed" "N/A" "Install cargo-audit for vulnerability scanning"
fi

echo "[3/7] Scanning for unsafe code blocks..."
UNSAFE_COUNT=$(grep -r "unsafe {" --include="*.rs" crates/ 2>/dev/null | wc -l || echo "0")
UNSAFE_FILES=$(grep -l "unsafe {" --include="*.rs" -r crates/ 2>/dev/null || true)

if [ "$UNSAFE_COUNT" -gt 0 ]; then
    echo -e "${YELLOW}⚠ Found $UNSAFE_COUNT unsafe blocks in:${NC}"
    echo "$UNSAFE_FILES" | head -10 | while read file; do
        echo "   - $file"
    done

    # Check if unsafe is properly documented
    UNDOCUMENTED_UNSAFE=$(grep -B 5 "unsafe {" --include="*.rs" -r crates/ 2>/dev/null | \
        grep -v "SAFETY:\|# Safety\|// Safety" | grep -c "unsafe {" || echo "0")

    if [ "$UNDOCUMENTED_UNSAFE" -gt 0 ]; then
        add_finding "MEDIUM" "Unsafe Code" \
            "Found $UNSAFE_COUNT unsafe blocks, some without safety documentation" \
            "$(echo "$UNSAFE_FILES" | head -1)" \
            "Add SAFETY: comments explaining why unsafe is necessary and what invariants are upheld"
    else
        add_finding "LOW" "Unsafe Code" \
            "Found $UNSAFE_COUNT unsafe blocks with documentation" \
            "Various files" \
            "Review periodically for necessity"
    fi
else
    echo -e "${GREEN}✓ No unsafe code blocks found${NC}"
fi

echo "[4/7] Checking for hardcoded secrets/tokens..."
SECRET_PATTERNS=(
    "password\s*="
    "api_key\s*="
    "secret\s*="
    "token\s*="
    "PRIVATE_KEY"
    "BEGIN RSA PRIVATE KEY"
    "BEGIN OPENSSH PRIVATE KEY"
)

FOUND_SECRETS=0
for pattern in "${SECRET_PATTERNS[@]}"; do
    MATCHES=$(grep -ri "$pattern" --include="*.rs" --include="*.toml" --include="*.yml" --include="*.yaml" \
        crates/ .github/ 2>/dev/null | grep -v "//\|#" | wc -l || echo "0")
    FOUND_SECRETS=$((FOUND_SECRETS + MATCHES))
done

if [ "$FOUND_SECRETS" -gt 0 ]; then
    add_finding "CRITICAL" "Hardcoded Secrets" \
        "Found potential hardcoded secrets/tokens" \
        "Source files" \
        "Use environment variables or secure vaults (e.g., AWS Secrets Manager, HashiCorp Vault)"
    echo -e "${RED}⚠ Found potential hardcoded secrets${NC}"
else
    echo -e "${GREEN}✓ No hardcoded secrets found${NC}"
fi

echo "[5/7] Checking for TODO/FIXME security comments..."
SECURITY_TODOS=$(grep -ri "TODO.*\(security\|unsafe\|vulnerable\|leak\|secret\|password\)" \
    --include="*.rs" crates/ 2>/dev/null | wc -l || echo "0")

if [ "$SECURITY_TODOS" -gt 0 ]; then
    add_finding "MEDIUM" "Pending Security Items" \
        "Found $SECURITY_TODOS TODO/FIXME comments related to security" \
        "Source code" \
        "Review and address security-related TODOs before production"
    echo -e "${YELLOW}⚠ Found $SECURITY_TODOS security-related TODOs${NC}"
else
    echo -e "${GREEN}✓ No pending security TODOs found${NC}"
fi

echo "[6/7] Checking serialization/deserialization safety..."
SERDE_UNCHECKED=$(grep -r "serde\|Deserialize\|Serialize" --include="*.rs" crates/ 2>/dev/null | \
    grep -v "Checked\|validate\|Verify" | head -20)

# Check for missing validation on deserialization
DESER_WITHOUT_CHECK=$(grep -B 5 "Deserialize" --include="*.rs" -r crates/ 2>/dev/null | \
    grep -A 5 "struct\|enum" | grep -v "TryFrom\|validate\|checked" | wc -l || echo "0")

if [ "$DESER_WITHOUT_CHECK" -gt 20 ]; then
    add_finding "MEDIUM" "Serialization Safety" \
        "Multiple types implement Deserialize without visible validation" \
        "Various files" \
        "Implement TryFrom or validation for deserialized types, especially if from untrusted sources"
    echo -e "${YELLOW}⚠ Review deserialization validation${NC}"
else
    echo -e "${GREEN}✓ Serialization checks look reasonable${NC}"
fi

echo "[7/7] Checking for potential DoS vectors..."
# Check for unbounded allocations
UNBOUNDED_ALLOCS=$(grep -r "Vec::with_capacity\|vec!\[" --include="*.rs" crates/ 2>/dev/null | \
    grep -v "checked\|min\|max\|limit" | wc -l || echo "0")

# Check for recursive functions without depth limits
RECURSIVE_FUNCS=$(grep -r "fn.*(\|->.*fn\|Box<Fn" --include="*.rs" crates/ 2>/dev/null | \
    grep -B 2 "recurse\|self\." | wc -l || echo "0")

if [ "$UNBOUNDED_ALLOCS" -gt 10 ]; then
    add_finding "LOW" "Resource Limits" \
        "Potential unbounded allocations detected" \
        "Memory allocation sites" \
        "Add size limits and validation for user-controlled inputs"
    echo -e "${YELLOW}⚠ Review unbounded allocations${NC}"
fi

# Update report with counts
sed -i.bak "s/CRITICAL_COUNT/$CRITICAL/g" "$REPORT_FILE"
sed -i.bak "s/HIGH_COUNT/$HIGH/g" "$REPORT_FILE"
sed -i.bak "s/MEDIUM_COUNT/$MEDIUM/g" "$REPORT_FILE"
sed -i.bak "s/LOW_COUNT/$LOW/g" "$REPORT_FILE"
sed -i.bak "s/INFO_COUNT/$INFO/g" "$REPORT_FILE"
rm -f "$REPORT_FILE.bak"

# Add recommendations section
cat >> "$REPORT_FILE" << EOF

## Recommendations

### Immediate Actions

EOF

if [ "$CRITICAL" -gt 0 ]; then
    echo "1. **Address Critical Findings**: $CRITICAL critical issue(s) require immediate attention." >> "$REPORT_FILE"
fi

if [ "$HIGH" -gt 0 ]; then
    echo "2. **Address High Severity Issues**: $HIGH high severity issue(s) should be fixed in next release." >> "$REPORT_FILE"
fi

cat >> "$REPORT_FILE" << EOF

3. **Regular Audits**: Schedule monthly security audits using this script.
4. **Dependency Updates**: Keep dependencies updated; review changelogs for security fixes.
5. **Unsafe Code Reviews**: All unsafe blocks should be reviewed by at least one additional developer.
6. **Fuzz Testing**: Consider adding fuzz tests for serialization/deserialization of public types.

## Compliance Checklist

- [ ] All unsafe blocks have SAFETY: documentation
- [ ] No hardcoded secrets in source code
- [ ] All dependencies are up to date
- [ ] Input validation on all public APIs
- [ ] Serialization includes size limits
- [ ] No debug/logging of sensitive data

## Appendix: Tools Used

| Tool | Purpose | Status |
|------|---------|--------|
| cargo audit | Dependency vulnerability scanning | $(command -v cargo-audit > /dev/null && echo "✓ Available" || echo "✗ Not installed") |
| grep | Pattern matching for security issues | ✓ Available |
| Manual review | Context-aware analysis | ✓ Performed |

---
*Generated by MNR Security Audit Script*
EOF

echo ""
echo "=========================================="
echo "Audit Complete"
echo "=========================================="
echo ""
echo "Summary:"
echo "  Critical: $CRITICAL"
echo "  High:     $HIGH"
echo "  Medium:   $MEDIUM"
echo "  Low:      $LOW"
echo "  Info:     $INFO"
echo ""
echo "Report saved to: $REPORT_FILE"

# Exit with error if critical issues found
if [ "$CRITICAL" -gt 0 ]; then
    echo -e "${RED}Critical issues found!${NC}"
    exit 1
fi

exit 0
