# QM2 Data Analysis App — Production Deployment Guide

This guide walks you through deploying `app.py` on a Linux server (RHEL/Rocky/Ubuntu)
at CHESS/CLASSE using **Gunicorn** as the WSGI server and **Nginx** as the reverse proxy.

---

## Files in this package

| File | Purpose |
|------|---------|
| `app.py` | The Flask application |
| `requirements.txt` | Python dependencies |
| `gunicorn.conf.py` | Gunicorn production configuration |
| `qm2app.service` | systemd service unit |
| `nginx_qm2app.conf` | Nginx reverse-proxy config |

---

## Step 1 — Pick a server & check NFS access

The app reads data from `/nfs/chess/id4baux/2026-1`, so it **must run on a machine
that has the CHESS NFS mounts**. A CLASSE analysis node (e.g. `lnx201`) is ideal.

```bash
ls /nfs/chess/id4baux/2026-1    # should show your experiment folders
```

---

## Step 2 — Create a dedicated service account

Running as root is a security risk. Create a non-login user:

```bash
sudo useradd --system --no-create-home --shell /sbin/nologin qm2app
```

---

## Step 3 — Install the app

```bash
# Choose a deployment directory
sudo mkdir -p /opt/qm2app
sudo chown qm2app:qm2app /opt/qm2app

# Copy files
sudo cp app.py gunicorn.conf.py requirements.txt /opt/qm2app/
```

---

## Step 4 — Create a Python virtual environment

```bash
cd /opt/qm2app
sudo -u qm2app python3 -m venv venv
sudo -u qm2app venv/bin/pip install --upgrade pip
sudo -u qm2app venv/bin/pip install -r requirements.txt
```

> **Tip for CHESS/CLASSE:** If `nxs-analysis-tools` or `nexusformat` are already
> installed system-wide (e.g. via conda/module), activate that environment first and
> then install only `flask gunicorn` on top of it, or use:
> ```bash
> pip install --extra-index-url ... flask gunicorn
> ```

---

## Step 5 — Create runtime directories & log directory

```bash
sudo mkdir -p /var/log/qm2app /run/qm2app
sudo chown qm2app:qm2app /var/log/qm2app /run/qm2app

# Also make sure the cache dir from app.py exists
sudo mkdir -p /tmp/lslice_cache
sudo chown qm2app:qm2app /tmp/lslice_cache
```

---

## Step 6 — Test Gunicorn manually (before touching systemd)

```bash
cd /opt/qm2app
sudo -u qm2app venv/bin/gunicorn -c gunicorn.conf.py app:app
```

Open `http://<server-ip>:5000` in a browser. If the app loads, Ctrl-C and move on.

---

## Step 7 — Install the systemd service

```bash
sudo cp qm2app.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable qm2app    # start automatically on boot
sudo systemctl start  qm2app

# Check it's running
sudo systemctl status qm2app
sudo journalctl -u qm2app -f    # live log tail
```

---

## Step 8 — Install & configure Nginx

```bash
# Install nginx if not already present
sudo dnf install nginx    # RHEL/Rocky
# or
sudo apt install nginx    # Ubuntu/Debian

# Edit the config to set your real hostname
nano nginx_qm2app.conf
# → Change  server_name qm2.classe.cornell.edu;  to your actual hostname or IP

sudo cp nginx_qm2app.conf /etc/nginx/sites-available/qm2app
sudo ln -s /etc/nginx/sites-available/qm2app /etc/nginx/sites-enabled/qm2app

# Test config then reload
sudo nginx -t
sudo systemctl reload nginx
```

---

## Step 9 — SSL certificate (recommended)

For an internal CLASSE server, request a certificate from the CLASSE sysadmin team
or use Let's Encrypt if the server is publicly reachable:

```bash
sudo certbot --nginx -d qm2.classe.cornell.edu
```

If you don't have a cert yet, uncomment the **HTTP-only** server block at the bottom
of `nginx_qm2app.conf` and comment out the HTTPS block.

---

## Updating the app

When you make changes to `app.py`:

```bash
sudo cp app.py /opt/qm2app/app.py
sudo systemctl reload qm2app    # graceful reload — no downtime
```

---

## Useful commands

```bash
# Check app status
sudo systemctl status qm2app

# View live logs
sudo journalctl -u qm2app -f

# View Gunicorn access log
sudo tail -f /var/log/qm2app/access.log

# Restart fully (e.g. after dependency update)
sudo systemctl restart qm2app

# Stop the app
sudo systemctl stop qm2app
```

---

## Port summary

| Component | Address | Notes |
|-----------|---------|-------|
| Gunicorn | `127.0.0.1:5000` | Internal only, not exposed |
| Nginx HTTP | `0.0.0.0:80` | Redirects to HTTPS |
| Nginx HTTPS | `0.0.0.0:443` | Public-facing endpoint |

The firewall only needs ports **80** and **443** open.
Port 5000 should **not** be open externally.

```bash
# Open firewall ports (firewalld — RHEL/Rocky)
sudo firewall-cmd --permanent --add-service=http
sudo firewall-cmd --permanent --add-service=https
sudo firewall-cmd --reload
```

---

## Troubleshooting

**App returns 502 Bad Gateway**
→ Gunicorn is not running. Check: `sudo systemctl status qm2app`

**NFS files not found**
→ The `qm2app` user may not have read access to the NFS mount.
  Add it to the appropriate group: `sudo usermod -aG chess_users qm2app`

**Slow first request after restart**
→ Normal — the app loads HDF5 metadata on first access. Subsequent requests use the cache.

**Gunicorn timeout on large slices**
→ Increase `timeout` in `gunicorn.conf.py` and `proxy_read_timeout` in the Nginx config.
  Current defaults: 120 s.
