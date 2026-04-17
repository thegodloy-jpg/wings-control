"""Quick k3s cluster status check via SSH."""
import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("7.6.52.110", username="root", password="xfusion@1234!")

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
