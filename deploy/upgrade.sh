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
TARGET_REF="origin/main"
AUTO=0

# Parse args: optional ref, optional --auto flag (used by watchtower timer)
for a in "$@"; do
    case "$a" in
        --auto) AUTO=1 ;;
        *) TARGET_REF="$a" ;;
    esac
done

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
