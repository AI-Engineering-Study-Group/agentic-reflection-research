# 🗺️ Codebase Navigation Map

## 📁 **Project Structure Overview**

```
agentic-research-framework/
│
├── 📚 DOCUMENTATION
│   ├── README.md                           # 🏠 Main entry point with navigation
│   ├── COMPREHENSIVE_BUILD_SUMMARY.md     # 📖 Complete technical guide (1,240 lines)
│   ├── docs/USE_CASE_EXTENSION_GUIDE.md   # 🔧 Beginner development guide (2,000+ lines)
│   ├── docs/PORT_ALLOCATION.md            # 🌐 Service organization strategy
│   ├── Iterative Reflection vs.txt        # 📊 Research methodology
│   └── Chapter 4_ Reflection.txt          # 🧠 Producer-Critic pattern theory
│
├── 🧠 CORE FRAMEWORK
│   ├── framework/
│   │   ├── base_agent.py                  # 🤖 Agent abstraction + ADK integration
│   │   ├── base_orchestrator.py           # 🔄 Reflection workflow management
│   │   └── base_evaluator.py              # 📊 Multi-dimensional quality assessment
│   │
│   ├── config/
│   │   ├── settings.py                    # ⚙️ Global configuration
│   │   └── experiments/                   # 🧪 Research experiment definitions
│   │
│   └── api/
│       ├── main.py                        # 🎛️ Main orchestrator (Port 8000)
│       └── use_case_server.py             # 🎯 Use case server (Port 8001+)
│
├── 🎯 USE CASES (Pluggable Domains)
│   └── system_design/                     # ✅ WORKING - GCP Architecture
│       ├── agents.py                      # 🏗️ GCP Producer + Technical Critic
│       ├── orchestrator.py               # 🔄 System design workflow
│       ├── evaluator.py                  # 📊 6-dimensional quality evaluation
│       ├── config.py                     # ⚙️ Quality dimensions + test scenarios
│       ├── tools/
│       │   └── cloud_pricing.py          # 💰 GCP/AWS/Azure pricing tools
│       └── docker/                       # 🐳 Independent deployment
│           ├── Dockerfile                # Container definition
│           └── docker-compose.yml        # Standalone service
│
├── 🧪 RESEARCH INFRASTRUCTURE
│   ├── research/
│   │   ├── experiment_orchestrator.py    # 🔬 Automated experiment runner
│   │   ├── data/                         # 📈 Experiment results
│   │   └── notebooks/                    # 📓 Jupyter analysis environment
│   │
│   └── main.py                           # 🖥️ CLI interface for experiments
│
├── 🐳 DEPLOYMENT
│   ├── docker-compose.yml                # 🚀 Multi-service orchestration
│   ├── docker/                           # 🏗️ Main framework containers
│   ├── .dockerignore                     # 📦 Build optimization
│   └── scripts/                          # 🔧 Deployment scripts
│
└── ⚙️ CONFIGURATION
    ├── pyproject.toml                     # 📦 Dependencies & build config
    ├── env.example                        # 🔑 Environment template
    └── .env                              # 🔒 Your secrets (not in git)
```

---

## 🎯 **Quick Navigation by Purpose**

### **🚀 I want to get started quickly**
1. **[README.md](./README.md)** - Main entry point
2. **[env.example](./env.example)** - Setup your environment
3. **[Quick Start section](./README.md#quick-start)** - 5-minute setup

### **🔬 I want to understand the research**
1. **[Iterative Reflection vs.txt](./Iterative%20Reflection%20vs.txt)** - Research methodology
2. **[Chapter 4_ Reflection.txt](./Chapter%204_%20Reflection.txt)** - Reflection pattern theory
3. **[Research Results](./README.md#research-results-validated)** - Validated findings

### **🏗️ I want to understand the architecture**
1. **[COMPREHENSIVE_BUILD_SUMMARY.md](./COMPREHENSIVE_BUILD_SUMMARY.md)** - Complete technical guide
2. **[framework/](./framework/)** - Core implementation
3. **[api/](./api/)** - API layer and endpoints

### **🎯 I want to see a working example**
1. **[use_cases/system_design/](./use_cases/system_design/)** - Complete reference implementation
2. **[agents.py](./use_cases/system_design/agents.py)** - Producer-Critic agents
3. **[tools/cloud_pricing.py](./use_cases/system_design/tools/cloud_pricing.py)** - Domain tools

### **🔧 I want to build my own use case**
1. **[docs/USE_CASE_EXTENSION_GUIDE.md](./docs/USE_CASE_EXTENSION_GUIDE.md)** - Complete development guide
2. **[Content Generation Tutorial](./docs/USE_CASE_EXTENSION_GUIDE.md#complete-beginner-tutorial)** - Step-by-step example
3. **[Learning Resources](./docs/USE_CASE_EXTENSION_GUIDE.md#learning-path-for-beginners)** - Beginner-friendly links

### **🐳 I want to deploy and scale**
1. **[docker-compose.yml](./docker-compose.yml)** - Multi-service deployment
2. **[docs/PORT_ALLOCATION.md](./docs/PORT_ALLOCATION.md)** - Service organization
3. **[Docker configurations](./use_cases/system_design/docker/)** - Container setup

### **📊 I want to run experiments**
1. **[config/experiments/](./config/experiments/)** - Experiment templates
2. **[research/](./research/)** - Analysis tools and notebooks
3. **[main.py](./main.py)** - CLI experiment runner

---

## 🎓 **Learning Path by Experience Level**

### **🆕 Complete Beginner**
1. **[Learning Resources](./docs/USE_CASE_EXTENSION_GUIDE.md#learning-path-for-beginners)** - AI agents, ADK, Docker basics
2. **[Chapter 4_ Reflection.txt](./Chapter%204_%20Reflection.txt)** - Understand the pattern
3. **[Quick Start](./README.md#quick-start)** - Get it running
4. **[Beginner Tutorial](./docs/USE_CASE_EXTENSION_GUIDE.md#complete-beginner-tutorial)** - Build your first use case

### **🐍 Python Developer**
1. **[COMPREHENSIVE_BUILD_SUMMARY.md](./COMPREHENSIVE_BUILD_SUMMARY.md)** - Technical architecture
2. **[framework/base_agent.py](./framework/base_agent.py)** - ADK integration patterns
3. **[use_cases/system_design/](./use_cases/system_design/)** - Reference implementation
4. **[Extension Guide](./docs/USE_CASE_EXTENSION_GUIDE.md)** - Build new domains

### **🔬 AI Researcher**
1. **[Iterative Reflection vs.txt](./Iterative%20Reflection%20vs.txt)** - Research design
2. **[Research Results](./README.md#research-results-validated)** - Validated findings
3. **[Quality Evaluation](./framework/base_evaluator.py)** - Assessment methodology
4. **[Experiment Runner](./research/experiment_orchestrator.py)** - Systematic testing

### **☁️ Cloud Engineer**
1. **[System Design Use Case](./use_cases/system_design/)** - GCP specialization
2. **[Cloud Pricing Tools](./use_cases/system_design/tools/cloud_pricing.py)** - Multi-cloud comparison
3. **[API Examples](./README.md#test-the-api-endpoints)** - Test real scenarios
4. **[Quality Dimensions](./use_cases/system_design/config.py)** - Professional assessment

---

## 🔍 **Find What You Need**

### **📖 Documentation**
- **Getting Started**: [README.md](./README.md)
- **Complete Guide**: [COMPREHENSIVE_BUILD_SUMMARY.md](./COMPREHENSIVE_BUILD_SUMMARY.md)
- **Extension Guide**: [docs/USE_CASE_EXTENSION_GUIDE.md](./docs/USE_CASE_EXTENSION_GUIDE.md)
- **Research Method**: [Iterative Reflection vs.txt](./Iterative%20Reflection%20vs.txt)

### **🤖 Agents & AI**
- **Base Classes**: [framework/](./framework/)
- **Working Example**: [use_cases/system_design/agents.py](./use_cases/system_design/agents.py)
- **ADK Integration**: [framework/base_agent.py](./framework/base_agent.py)
- **Reflection Pattern**: [Chapter 4_ Reflection.txt](./Chapter%204_%20Reflection.txt)

### **🌐 APIs & Services**
- **Main API**: [api/main.py](./api/main.py) (Port 8000)
- **Use Case API**: [api/use_case_server.py](./api/use_case_server.py) (Port 8001+)
- **Endpoints**: [README.md#api-usage](./README.md#test-the-api-endpoints)
- **Health Checks**: All services have `/health` endpoints

### **🐳 Deployment**
- **Docker Compose**: [docker-compose.yml](./docker-compose.yml)
- **Containers**: [use_cases/system_design/docker/](./use_cases/system_design/docker/)
- **Port Strategy**: [docs/PORT_ALLOCATION.md](./docs/PORT_ALLOCATION.md)
- **Environment**: [env.example](./env.example)

### **🧪 Research & Experiments**
- **Experiment Runner**: [research/experiment_orchestrator.py](./research/experiment_orchestrator.py)
- **Quality Evaluation**: [framework/base_evaluator.py](./framework/base_evaluator.py)
- **Config Templates**: [config/experiments/](./config/experiments/)
- **Analysis Tools**: [research/notebooks/](./research/notebooks/)

### **🛠️ Tools & Utilities**
- **Cloud Pricing**: [use_cases/system_design/tools/cloud_pricing.py](./use_cases/system_design/tools/cloud_pricing.py)
- **CLI Interface**: [main.py](./main.py)
- **Configuration**: [config/settings.py](./config/settings.py)
- **Scripts**: [scripts/](./scripts/)

---

## 🎯 **Common Tasks**

### **Start the Framework**
```bash
docker-compose up -d system_design
curl http://localhost:8001/health
```

### **Test Research Functionality**
```bash
# Baseline test
curl -X POST http://localhost:8001/chat -d '{"message":"Design a web app","mode":"baseline"}'

# Reflection test  
curl -X POST http://localhost:8001/chat -d '{"message":"Design a web app","mode":"reflection"}'
```

### **Add New Use Case**
1. Follow **[Extension Guide](./docs/USE_CASE_EXTENSION_GUIDE.md)**
2. Copy **[system_design](./use_cases/system_design/)** as template
3. Customize agents and quality dimensions
4. Deploy on new port (8002, 8003, etc.)

### **Run Research Experiments**
```bash
python main.py run-experiment config/experiments/system_design_pilot.json
```

### **Analyze Results**
```bash
# Start Jupyter environment
docker-compose --profile research up -d
# Open http://localhost:8891
```

---

## 🏆 **Project Achievements**

- ✅ **40 files, 8,190+ lines of code**
- ✅ **Working Producer-Critic reflection pattern**
- ✅ **Complete ADK integration**
- ✅ **Multi-dimensional quality evaluation**
- ✅ **Docker containerization**
- ✅ **Research results validation**
- ✅ **Professional documentation**
- ✅ **Beginner-friendly guides**
- ✅ **Extensible architecture**
- ✅ **Production-ready deployment**

**Ready for research, deployment, and community validation!** 🚀✨
