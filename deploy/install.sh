#!/usr/bin/env bash
# install.sh — first-time setup on a fresh validator VPS.
#
# Run as root after cloning the repo to /opt/compelle-validator and creating
# /etc/compelle/validator.env with your CHUTES_API_KEY + BT_WALLET_NAME etc.
#
# Idempotent — safe to re-run. Will refresh the systemd unit symlinks and
# restart services to pick up unit-file changes.
#
# Auto-update is **ON BY DEFAULT** (compelle-watchtower.timer enabled). To opt
# out: `systemctl disable --now compelle-watchtower.timer` after running this.
set -euo pipefail

REPO_DIR="${COMPELLE_REPO_DIR:-/opt/compelle-validator}"

if [ "$EUID" -ne 0 ]; then
    echo "must run as root (systemctl + /etc writes)"
    exit 1
fi

if [ ! -d "$REPO_DIR/deploy" ]; then
    echo "REPO_DIR=$REPO_DIR does not look like a clone of compelle-validator"
    exit 1
fi

echo "=== installing systemd units from $REPO_DIR/deploy ==="
for unit in compelle-validator.service compelle-watchtower.service compelle-watchtower.timer; do
    src="$REPO_DIR/deploy/$unit"
    dst="/etc/systemd/system/$unit"
    if [ ! -f "$src" ]; then
        echo "  skip: $src does not exist"
        continue
    fi
    cp "$src" "$dst"
    echo "  installed: $dst"
done

echo ""
echo "=== ensuring /etc/compelle exists ==="
mkdir -p /etc/compelle

if [ ! -f /etc/compelle/validator.env ]; then
    if [ -f "$REPO_DIR/deploy/compelle.env.example" ]; then
        cp "$REPO_DIR/deploy/compelle.env.example" /etc/compelle/validator.env
        chmod 600 /etc/compelle/validator.env
        echo "  seeded /etc/compelle/validator.env from compelle.env.example"
        echo "  EDIT THIS FILE with your CHUTES_API_KEY + BT_WALLET_NAME etc, then re-run install.sh"
        echo ""
        echo "  exiting so you can edit before services start"
        exit 0
    else
        echo "  no validator.env and no example to seed from — create /etc/compelle/validator.env manually"
        exit 1
    fi
fi

echo ""
echo "=== installing python deps ==="
if [ ! -d "$REPO_DIR/.venv" ]; then
    python3 -m venv "$REPO_DIR/.venv"
fi
"$REPO_DIR/.venv/bin/pip" install -e "$REPO_DIR" --quiet

echo ""
echo "=== reloading systemd ==="
systemctl daemon-reload

echo ""
echo "=== enabling services ==="
systemctl enable --now compelle-validator.service
echo "  compelle-validator: $(systemctl is-active compelle-validator)"
systemctl enable --now compelle-watchtower.timer
echo "  compelle-watchtower.timer: $(systemctl is-active compelle-watchtower.timer)"

echo ""
echo "=== done ==="
echo ""
echo "Watchtower is ON by default and will auto-pull origin/main every 10 min."
echo "To disable auto-update:"
echo "  systemctl disable --now compelle-watchtower.timer"
echo ""
echo "To canary a non-main branch on this validator:"
echo "  echo 'COMPELLE_TARGET_REF=origin/staging' > /etc/compelle/watchtower.env"
echo "  systemctl daemon-reload && systemctl restart compelle-watchtower.timer"
echo ""
echo "Logs:"
echo "  validator:  journalctl -u compelle-validator -f  (or tail validator.log)"
echo "  watchtower: journalctl -u compelle-watchtower -f"
