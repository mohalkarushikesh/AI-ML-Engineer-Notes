# Docker Cheatsheet

---

## Service Control

| Command | Description |
|---------|-------------|
| `sudo systemctl start docker` | Start Docker (socket follows automatically) |
| `sudo systemctl stop docker docker.socket` | Stop Docker + socket (always stop both!) |
| `sudo systemctl restart docker` | Restart Docker |
| `sudo systemctl status docker` | Check if Docker is running |
| `sudo systemctl enable docker` | Enable auto-start on boot |
| `sudo systemctl disable docker docker.socket` | Disable auto-start on boot |
| `sudo systemctl is-enabled docker` | Check if auto-start is enabled |

> **Tip:** To start — only `start docker` (socket follows). To stop — stop **both** `docker` and `docker.socket`, otherwise the socket will wake Docker back up.

---

## Images

| Command | Description |
|---------|-------------|
| `sudo docker images` | List all downloaded images |
| `sudo docker pull ubuntu` | Download an image |
| `sudo docker rmi ubuntu` | Delete an image |
| `sudo docker rmi -f ubuntu` | Force delete an image |
| `sudo docker image prune` | Remove all unused images |
| `sudo docker inspect ubuntu` | View image details |
| `sudo docker tag ubuntu myubuntu:v1` | Tag / rename an image |

---

## Containers

| Command | Description |
|---------|-------------|
| `sudo docker ps` | List running containers |
| `sudo docker ps -a` | List all containers (including stopped) |
| `sudo docker run ubuntu` | Create and start a container |
| `sudo docker run -it ubuntu bash` | Run with interactive terminal |
| `sudo docker run -d ubuntu sleep 999` | Run in background (detached) |
| `sudo docker run --name mybox ubuntu` | Give the container a custom name |
| `sudo docker run --rm ubuntu` | Auto-delete container after it exits |
| `sudo docker start goofy_galileo` | Start a stopped container |
| `sudo docker stop goofy_galileo` | Stop a running container |
| `sudo docker restart goofy_galileo` | Restart a container |
| `sudo docker rm goofy_galileo` | Delete a stopped container |
| `sudo docker rm -f goofy_galileo` | Force delete (even if running) |

---

## Inside Containers

| Command | Description |
|---------|-------------|
| `sudo docker exec -it mybox bash` | Open a shell inside a running container |
| `sudo docker exec mybox ls /` | Run a single command inside a container |
| `sudo docker logs mybox` | View container logs |
| `sudo docker logs -f mybox` | Follow live logs |
| `sudo docker cp mybox:/app/log.txt .` | Copy a file from container to host |
| `sudo docker cp file.txt mybox:/app/` | Copy a file from host into container |
| `sudo docker top mybox` | View running processes inside container |
| `sudo docker stats` | Live CPU and memory usage of all containers |

---

## Ports & Volumes

| Command | Description |
|---------|-------------|
| `sudo docker run -p 8080:80 nginx` | Map host port 8080 → container port 80 |
| `sudo docker run -v /host/path:/container/path ubuntu` | Mount a folder (bind mount) |
| `sudo docker volume create myvol` | Create a named volume |
| `sudo docker volume ls` | List all volumes |
| `sudo docker volume rm myvol` | Delete a volume |
| `sudo docker run -v myvol:/data ubuntu` | Use a named volume in a container |

---

## Build & Dockerfile

| Command | Description |
|---------|-------------|
| `sudo docker build -t myapp .` | Build image from Dockerfile in current folder |
| `sudo docker build -t myapp:v2 .` | Build with a version tag |
| `sudo docker build --no-cache -t myapp .` | Build without using cache |
| `sudo docker history myapp` | Show image layer history |

---

## Cleanup

| Command | Description |
|---------|-------------|
| `sudo docker system prune` | Remove stopped containers + dangling images |
| `sudo docker system prune -a` | Remove everything unused (images, containers, volumes, networks) |
| `sudo docker container prune` | Remove all stopped containers |
| `sudo docker image prune -a` | Remove all unused images |
| `sudo docker volume prune` | Remove unused volumes |

---

## Networking

| Command | Description |
|---------|-------------|
| `sudo docker network ls` | List all networks |
| `sudo docker network create mynet` | Create a custom network |
| `sudo docker run --network mynet ubuntu` | Connect a container to a network |
| `sudo docker network inspect mynet` | View network details |
| `sudo docker network rm mynet` | Delete a network |

---

## Common Flags

| Flag | Description |
|------|-------------|
| `-it` | Interactive terminal (combine `-i` and `-t`) |
| `-d` | Detached / run in background |
| `-p 8080:80` | Port mapping (host:container) |
| `-v /host:/app` | Volume / folder mount |
| `--name mybox` | Set a custom container name |
| `--rm` | Auto-delete container after it exits |
| `-e KEY=value` | Set an environment variable |
| `--network mynet` | Attach to a specific network |
| `-f` | Force (used with rm, rmi, etc.) |

---

## Exit Codes

| Code | Meaning |
|------|---------|
| `Exited (0)` | Container ran and stopped successfully |
| `Exited (1)` | Container stopped with an error |
| `Exited (137)` | Container was force-killed (OOM or `docker kill`) |

---

## Image Status (EXTRA column)

| Status | Meaning |
|--------|---------|
| `U` (In Use) | At least one container (running or stopped) was created from this image — cannot delete the image until the container is removed |


Solution — create a new container correctly
- bashsudo docker run -it --name my_ubuntu ubuntu bash
This time it will have OpenStdin: true and Tty: true, so bash will stay alive when you start/stop it.
Then to reuse it next time:
- bashsudo docker start -i my_ubuntu
