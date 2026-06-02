1. Install Prerequisites
	sudo apt-get update
	sudo apt-get install -y ca-certificates curl gnupg lsb-release

2. Add Docker's GPG Key
	sudo mkdir -p /etc/apt/keyrings
	curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

3. Set Up the Repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

4. Install Docker Engine
	sudo apt-get update
	sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

5. Verify the Installation
	sudo docker run hello-world


You want Docker to **only start when you manually start it** — and stay stopped otherwise. This means **disabling Docker's auto-start on boot**.

---

### Disable Docker Auto-start

```bash
sudo systemctl disable docker
sudo systemctl disable docker.socket
```

---

### This means:
- ✅ Docker will **stay stopped** when your PC boots
- ✅ You can **manually start** it whenever you need it
- ✅ When done, it goes back to stopped after reboot

---

### When you WANT to use Docker:

**Start it manually:**
```bash
sudo systemctl start docker
```

**Stop it when done:**
```bash
sudo systemctl stop docker
```

**Check its current status:**
```bash
sudo systemctl status docker
```

---

### Verify auto-start is disabled:
```bash
sudo systemctl is-enabled docker
# Should output: disabled
```

---

### Quick Reference

| Command | Purpose |
|---------|---------|
| `sudo systemctl disable docker` | Stop auto-start on boot |
| `sudo systemctl start docker` | Start Docker manually |
| `sudo systemctl stop docker` | Stop Docker manually |
| `sudo systemctl status docker` | Check if running or stopped |

This way Docker is fully under **your control** and won't run in the background unnecessarily.

Why Docker is Still Running

- Docker is disabled (won't auto-start on next boot) but it's still running now because disabling only affects future boots — it doesn't stop the currently running instance.

---

- Boot PC
  ↓
Docker = Stopped ✅
  ↓
Need Docker? → sudo systemctl start docker
  ↓
Use Docker...
  ↓
Done? → sudo systemctl stop docker && sudo systemctl stop docker.socket
  ↓
Reboot → Docker = Stopped again ✅
