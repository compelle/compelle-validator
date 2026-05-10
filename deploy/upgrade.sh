#!/usr/bin/env bash
# upgrade.sh — pull latest compelle-validator code + restart service.
# Run on validator VPS as root or via sudo.
#
# Validators: when announcement says "please upgrade", run this.
# Auto-update is intentionally NOT enabled — you stay in control of what runs.
#
# Usage:
#   sudo /opt/compelle-validator/deploy/upgrade.sh
#
# Optionally pin a specific commit/tag instead of HEAD:
#   sudo /opt/compelle-validator/deploy/upgrade.sh v1.2.3

set -euo pipefail

REPO_DIR="${COMPELLE_REPO_DIR:-/opt/compelle-validator}"
SERVICE="${COMPELLE_SERVICE:-compelle-validator}"
# Default branch can be overridden via $COMPELLE_TARGET_REF (e.g., set
# COMPELLE_TARGET_REF=origin/staging in /etc/compelle/watchtower.env to canary
# a branch on a single operator before merging to main). CLI arg still wins
# when non-blank. Empty/blank values from any path fall back to origin/main —
# same posture as the COMPELLE_PUSH_URL fallback fix.
TARGET_REF="${COMPELLE_TARGET_REF:-}"
AUTO=0

# Parse args: optional ref, optional --auto flag (used by watchtower timer).
# Blank positional args are ignored so `upgrade.sh --auto ""` doesn't end up
# with TARGET_REF="" and abort under set -u.
for a in "$@"; do
    case "$a" in
        --auto) AUTO=1 ;;
        "") ;;
        *) TARGET_REF="$a" ;;
    esac
done

# Final safety net: empty from env, empty from CLI, never set — all → origin/main.
TARGET_REF="${TARGET_REF:-origin/main}"

echo "=== compelle-validator upgrade ==="
echo "  repo: $REPO_DIR"
echo "  target: $TARGET_REF"
echo "  service: $SERVICE"
echo

cd "$REPO_DIR"

echo "=== git fetch ==="
git fetch --tags --quiet

echo
echo "=== current → target ==="
echo "  current: $(git rev-parse --short HEAD) ($(git log -1 --format=%s))"
echo "  target:  $(git rev-parse --short "$TARGET_REF") ($(git log -1 --format=%s "$TARGET_REF"))"
echo

if [ "$(git rev-parse HEAD)" = "$(git rev-parse "$TARGET_REF")" ]; then
    echo "already at target. Nothing to do."
    exit 0
fi

# Optional supply-chain hardening. Set COMPELLE_REQUIRE_SIGNED=1 in
# /etc/compelle/watchtower.env to refuse any target that is not a gpg-signed
# annotated tag. Operators must first import the maintainer's public key
# (the maintainer publishes their key fingerprint on multiple channels;
# verify out-of-band before trust). With this set, watchtower will refuse
# to deploy origin/main or any unsigned ref. Use COMPELLE_TARGET_REF=vX.Y.Z
# (a specific signed release tag) alongside this flag.
if [ "${COMPELLE_REQUIRE_SIGNED:-0}" = "1" ]; then
    target_type=$(git cat-file -t "$TARGET_REF" 2>/dev/null || echo "unknown")
    if [ "$target_type" != "tag" ]; then
        echo "REFUSED (COMPELLE_REQUIRE_SIGNED=1): target $TARGET_REF is a $target_type, not an annotated tag"
        echo "Set COMPELLE_TARGET_REF to a signed tag like v0.2.0, or unset COMPELLE_REQUIRE_SIGNED."
        exit 1
    fi
    if ! git verify-tag "$TARGET_REF" 2>&1; then
        echo "REFUSED (COMPELLE_REQUIRE_SIGNED=1): gpg signature on $TARGET_REF did not verify"
        echo "Either the maintainer's key is not in your gpg keyring, or the tag is forged."
        exit 1
    fi
    echo "  signed tag verified: $TARGET_REF"
fi

echo "=== reset to target (DESTRUCTIVE: any local changes will be lost) ==="
if [ "$AUTO" = "1" ]; then
    echo "  --auto flag set; skipping confirmation"
else
    read -p "Continue? [y/N] " ans
    if [ "$ans" != "y" ] && [ "$ans" != "Y" ]; then
        echo "aborted"
        exit 1
    fi
fi

git reset --hard "$TARGET_REF"

echo
echo "=== reinstall deps if pyproject changed ==="
if git diff HEAD@{1} HEAD --name-only | grep -q pyproject.toml; then
    echo "pyproject.toml changed; running pip install"
    "$REPO_DIR/.venv/bin/pip" install -e . --quiet
fi

echo
echo "=== restart service ==="
systemctl restart "$SERVICE"
sleep 3
systemctl is-active "$SERVICE"

echo
echo "=== first 10 log lines after restart ==="
journalctl -u "$SERVICE" --since "30 sec ago" --no-pager | tail -10

echo
echo "=== upgrade complete. Watch with: journalctl -u $SERVICE -f"
