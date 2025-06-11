"""
scripts/distribute.py

SSH-based orchestration utilities for NeuroPRIN distributed training:
- scan_network: detect reachable hosts on subnet
- generate_ssh_key: create SSH keypair if missing
- distribute_ssh_key_and_files: copy key and scripts to remote nodes
- start_clients: launch remote clients via Paramiko
- run_distributed_processing: send data chunks over socket and gather results
"""
import os
import socket
import threading
import paramiko
import ipaddress
import json
import pickle
from typing import List, Dict, Any, Tuple

# Default SSH parameters
default_ssh_user = os.getenv('SSH_USER', 'ubuntu')
default_ssh_key = os.path.expanduser(os.getenv('SSH_KEY_PATH', '~/.ssh/id_rsa'))


def scan_network(cidr: str) -> List[str]:
    """
    Scan the given CIDR for hosts that respond on port 22 (SSH).
    Returns a list of IP addresses.
    """
    network = ipaddress.ip_network(cidr)
    reachable = []
    def _probe(ip: str):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.5)
            s.connect((ip, 22))
            reachable.append(ip)
        except Exception:
            pass
        finally:
            s.close()
    threads = []
    for host in network.hosts():
        t = threading.Thread(target=_probe, args=(str(host),))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    return reachable


def generate_ssh_key(key_path: str = default_ssh_key) -> Tuple[str, str]:
    """
    Generate an RSA SSH keypair at key_path (no passphrase).
    Returns (public_key_path, private_key_path).
    Does nothing if key already exists.
    """
    pub = key_path + '.pub'
    if os.path.exists(key_path) and os.path.exists(pub):
        return pub, key_path
    cmd = f"ssh-keygen -t rsa -b 4096 -f {key_path} -N ''"
    os.makedirs(os.path.dirname(key_path), exist_ok=True)
    os.system(cmd)
    return pub, key_path


def distribute_ssh_key_and_files(
    nodes: List[str],
    local_files: List[str],
    remote_dir: str = '~/neuroprin'
) -> None:
    """
    Copy SSH public key and local_files to each node under remote_dir via Paramiko/SFTP.
    """
    pub, priv = generate_ssh_key()
    for host in nodes:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(host, username=default_ssh_user, key_filename=priv)
        sftp = ssh.open_sftp()
        # Ensure directory
        try:
            sftp.chdir(remote_dir)
        except IOError:
            sftp.mkdir(remote_dir)
            sftp.chdir(remote_dir)
        # Copy files
        for f in local_files + [pub]:
            fname = os.path.basename(f)
            sftp.put(os.path.expanduser(f), os.path.join(remote_dir, fname))
        sftp.close()
        ssh.close()


def start_clients(
    nodes: List[str],
    script: str = 'client.py',
    remote_dir: str = '~/neuroprin',
    python_cmd: str = 'python3'
) -> None:
    """
    SSH into each node and launch the given script in background.
    """
    for host in nodes:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(host, username=default_ssh_user, key_filename=default_ssh_key)
        cmd = f"cd {remote_dir} && nohup {python_cmd} {script} > client.log 2>&1 &"
        ssh.exec_command(cmd)
        ssh.close()


def run_distributed_processing(
    nodes: List[str],
    data_chunks: List[Any],
    port: int = 5000,
    timeout: int = 30
) -> Dict[str, Any]:
    """
    Send serialized data_chunks to each node over TCP socket on given port.
    Collect and return a mapping node->result.
    Assumes client is listening on <port> and echoes back pickled result.
    """
    results: Dict[str, Any] = {}
    def _worker(host: str, chunk: Any):
        addr = (host, port)
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(timeout)
            s.connect(addr)
            payload = pickle.dumps(chunk)
            s.sendall(len(payload).to_bytes(8, 'big') + payload)
            # receive length + payload
            length_bytes = s.recv(8)
            length = int.from_bytes(length_bytes, 'big')
            data = b''
            while len(data) < length:
                packet = s.recv(min(4096, length - len(data)))
                if not packet:
                    break
                data += packet
            result = pickle.loads(data)
            results[host] = result
        except Exception as e:
            results[host] = {'error': str(e)}
        finally:
            s.close()
    threads = []
    for host, chunk in zip(nodes, data_chunks):
        t = threading.Thread(target=_worker, args=(host, chunk))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    return results


if __name__ == '__main__':  # simple CLI
    import argparse
    parser = argparse.ArgumentParser(description='Distribute NeuroPRIN tasks')
    parser.add_argument('--cidr', type=str, help='Subnet to scan (e.g. 192.168.1.0/24)')
    parser.add_argument('--files', nargs='+', help='Local files to distribute')
    parser.add_argument('--script', type=str, default='client.py', help='Client script to run on nodes')
    parser.add_argument('--port', type=int, default=5000, help='Port for distributed processing')
    args = parser.parse_args()

    if args.cidr and args.files:
        nodes = scan_network(args.cidr)
        distribute_ssh_key_and_files(nodes, args.files)
        start_clients(nodes, script=args.script)
        print(f"Started clients on: {nodes}")
    else:
        parser.print_help()
