"""Quick k3s cluster status check via SSH."""
import os

import paramiko

host = os.getenv("WINGS_K3S_HOST", "7.6.52.110")
username = os.getenv("WINGS_K3S_USER", "root")
password = os.getenv("WINGS_K3S_PASSWORD")
if not password:
    raise SystemExit("WINGS_K3S_PASSWORD is required; do not store SSH passwords in source files.")

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(host, username=username, password=password)

cmds = [
    ("K3s Nodes", "kubectl get nodes -o wide"),
    ("Pods (all ns)", "kubectl get pods -A --no-headers | head -30"),
    ("Namespaces", "kubectl get ns --no-headers"),
]
for label, cmd in cmds:
    _, stdout, stderr = c.exec_command(cmd)
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    print(f"=== {label} ===")
    print(out or err or "(empty)")
    print()

c.close()
