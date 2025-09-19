# Port Allocation Strategy

## Overview

Each use case runs in its own dedicated container with specific port assignments to enable:
- **Independent scaling** and resource allocation
- **Parallel experiment execution** across use cases
- **Isolated measurement** of performance and costs
- **No port conflicts** when running multiple use cases simultaneously

## Port Allocation Scheme

### Core Infrastructure
- **8000**: Main orchestrator API (coordinates all use cases)
- **5433**: Shared PostgreSQL database (optional, for cross-use-case data)

### Use Case APIs
- **8001**: System Design use case
- **8002**: Content Generation use case (future)
- **8003**: Code Review use case (future)
- **8004**: Data Analysis use case (future)
- **8005**: Planning use case (future)
- **8006-8099**: Reserved for additional use cases

### Research/Jupyter Environments
- **8891**: System Design Jupyter Lab
- **8892**: Content Generation Jupyter Lab (future)
- **8893**: Code Review Jupyter Lab (future)
- **8894**: Data Analysis Jupyter Lab (future)
- **8895**: Planning Jupyter Lab (future)
- **8896-8999**: Reserved for additional research environments

### Use Case Databases (if needed)
- **5434**: System Design dedicated database
- **5435**: Content Generation dedicated database (future)
- **5436**: Code Review dedicated database (future)
- **5437-5499**: Reserved for additional use case databases

## Current Active Ports

### System Design Use Case
```bash
# API Server
http://localhost:8001/docs     # API documentation
http://localhost:8001/health   # Health check
http://localhost:8001/chat     # Chat endpoint

# Research Environment (with --profile research)
http://localhost:8891          # Jupyter Lab
```

### Main Orchestrator
```bash
# Coordination API
http://localhost:8000/docs     # Main API documentation
http://localhost:8000/         # Health and use case discovery
```

## Usage Examples

### Start All Use Cases
```bash
# Start all use cases in development mode
docker-compose up -d

# Start with research environments
docker-compose --profile research up -d
```

### Start Specific Use Case
```bash
# Start only system design use case
docker-compose up -d system_design

# Start system design with research environment
docker-compose --profile research up -d system_design system_design_research
```

### Test Individual Use Cases
```bash
# Test system design use case directly
curl http://localhost:8001/health

# Chat with system design use case
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Design a web app", "mode": "reflection"}'

# Compare modes in system design
curl -X POST http://localhost:8001/chat/compare \
  -H "Content-Type: application/json" \
  -d '{"message": "Design a scalable system"}'
```

## Research Benefits

### Independent Measurement
- Each use case has **dedicated CPU, memory, and network resources**
- **Isolated experiment execution** prevents interference between use cases
- **Accurate performance metrics** without cross-contamination

### Parallel Experiments
- Run **multiple experiments simultaneously** across different use cases
- **Compare use case effectiveness** for reflection patterns
- **Scale individual use cases** based on research demands

### Resource Allocation
```bash
# Scale system design use case independently
docker-compose up -d --scale system_design=3

# Allocate more resources to specific use case
docker-compose -f docker-compose.yml -f docker-compose.override.yml up -d
```

## Development Workflow

### Adding New Use Cases

1. **Create use case directory structure:**
```bash
mkdir -p use_cases/new_use_case/docker
```

2. **Assign ports** (next available in sequence):
   - API: 8002 (next available)
   - Jupyter: 8892 (corresponding research port)
   - Database: 5435 (if needed)

3. **Create Docker files** following the system_design pattern

4. **Update main docker-compose.yml** with new service

### Port Conflict Resolution

If ports are already in use on your system:

```bash
# Check what's using a port
netstat -tulpn | grep :8001

# Kill process using port
sudo kill -9 <PID>

# Or change port in docker-compose.yml
ports:
  - "127.0.0.1:8101:8001"  # Map to different external port
```

## Monitoring and Debugging

### Health Checks
```bash
# Check all use case health
curl http://localhost:8001/health  # System Design
curl http://localhost:8002/health  # Content Generation (future)
curl http://localhost:8003/health  # Code Review (future)
```

### Container Logs
```bash
# View logs for specific use case
docker-compose logs system_design

# Follow logs in real-time
docker-compose logs -f system_design

# View all use case logs
docker-compose logs
```

### Resource Usage
```bash
# Monitor container resource usage
docker stats system_design_use_case

# Monitor all research containers
docker stats $(docker ps --format "table {{.Names}}" | grep -E "(use_case|research)")
```

This architecture ensures that your research experiments run in truly isolated environments, giving you accurate and reliable measurements for your reflection vs. capability study.
