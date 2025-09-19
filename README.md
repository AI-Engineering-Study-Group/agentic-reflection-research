# Agentic Research Framework

A modular framework for researching **iterative reflection vs model capability** in AI systems, with a focus on cloud system design use cases.

## Overview

This framework addresses the fundamental research question: **Can iterative reflection with lower-capability models match or exceed the performance of single-pass higher-capability models while maintaining cost efficiency?**

### Research Design Rationale

**Default Model: `gemini-2.5-flash-lite`**
- **No thinking capability by default** - provides clean baseline without internal reasoning
- **Lower cost** - perfect for testing cost-effectiveness of reflection
- **Clear signal** - when reflection improves Flash-Lite output, it demonstrates the power of iterative improvement

**Research Hypothesis**: `Flash-Lite + Reflection ≥ Pro (single-pass)` in quality while maintaining cost efficiency.

### Key Features

- **Modular Architecture**: Pluggable use cases with consistent research infrastructure
- **Producer-Critic Pattern**: Implements reflection patterns from AI research
- **Multi-dimensional Evaluation**: Comprehensive quality assessment across multiple criteria
- **Cost Tracking**: Detailed analysis of API costs and token usage
- **Expert Integration**: Framework for incorporating domain expert evaluations
- **Statistical Analysis**: Built-in significance testing and comparative analysis

## Quick Start

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- Google AI API key

### Setup

1. **Clone and setup environment:**
```bash
git clone <repository-url>
cd agentic-research-framework
cp env.example .env
# Edit .env with your Google API key (REQUIRED)
```

2. **Install dependencies with UV:**
```bash
# Install UV if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install project dependencies
uv sync --all-extras
```

3. **Start the research framework:**
```bash
# Option 1: All use cases with Docker
docker-compose up -d

# Option 2: Specific use case only
docker-compose up -d system_design

# Option 3: With research environments
docker-compose --profile research up -d

# Option 4: Direct Python (single use case)
USE_CASE=system_design API_PORT=8001 uvicorn api.use_case_server:app --reload
```

4. **Access the APIs:**
- **Main Orchestrator**: http://localhost:8000/docs
- **System Design**: http://localhost:8001/docs (dedicated container)
- **System Design Research**: http://localhost:8891 (Jupyter Lab)
- **Health Checks**: All endpoints have `/health` for monitoring

### Test the API Endpoints

```bash
# Test system design use case directly (dedicated container)
curl -X POST "http://localhost:8001/chat" \
     -H "Content-Type: application/json" \
     -d '{
       "message": "Design a web application for 10k users",
       "mode": "reflection",
       "model": "gemini-2.5-flash-lite"
     }'

# Test main orchestrator (coordinates all use cases)
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{
       "message": "Design a web application for 10k users",
       "mode": "reflection",
       "use_case": "system_design"
     }'

# Compare modes side-by-side (system design container)
curl -X POST "http://localhost:8001/chat/compare" \
     -H "Content-Type: application/json" \
     -d '{
       "message": "Design a scalable e-commerce platform",
       "model": "gemini-2.5-flash-lite"
     }'

# Get available models and use cases
curl "http://localhost:8000/models"        # Main orchestrator
curl "http://localhost:8001/info"          # System design specific info
```

### Frontend Integration Example

```javascript
// Chat with system design use case (dedicated container)
const response = await fetch('http://localhost:8001/chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    message: "Design a system for Black Friday traffic",
    mode: "reflection",
    model: "gemini-2.5-flash-lite",
    reflection_iterations: 3
  })
});

const result = await response.json();
console.log(result.response); // Agent's system design
console.log(result.quality_score); // Quality evaluation
console.log(result.use_case); // "system_design"
console.log(result.resource_usage); // Container and isolation info

// Compare approaches in isolated environment
const comparison = await fetch('http://localhost:8001/chat/compare', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    message: "Design a scalable microservices architecture"
  })
});
```

## Architecture

### Framework Core

- **`framework/base_agent.py`**: Abstract base class for all agents
- **`framework/base_orchestrator.py`**: Workflow orchestration with reflection patterns  
- **`framework/base_evaluator.py`**: Multi-dimensional quality evaluation

### Use Cases

- **`use_cases/system_design/`**: Cloud architecture design with cost comparison
  - Agents: Producer (generates designs), Critic (evaluates and suggests improvements)
  - Tools: GCP/AWS/Azure pricing, security analysis, scaling calculations
  - Evaluator: 6-dimensional quality assessment

### Research Infrastructure

- **`research/experiment_orchestrator.py`**: Systematic experiment execution
- **`research/results_analyzer.py`**: Statistical analysis and visualization
- **Cost tracking and performance monitoring**

## System Design Use Case

The system design use case specializes in:

- **GCP-focused architecture** with multi-cloud cost comparison
- **Scalability analysis** for traffic spikes (e.g., Black Friday scenarios)
- **Security and compliance** assessment (PCI DSS, GDPR, SOX)
- **Cost optimization** across cloud providers
- **Expert evaluation** integration for validation

### Quality Dimensions

1. **Technical Accuracy** (25%): Correctness of technical decisions
2. **Cost Optimization** (20%): Cost efficiency across providers
3. **Security Posture** (20%): Security best practices and compliance
4. **Scalability Design** (15%): Ability to handle growth and peak loads
5. **Completeness** (10%): Coverage of all requirements
6. **Clarity** (10%): Documentation and explanation quality

## Research Methodology

### Experimental Design

- **Models**: Test across Gemini Flash, Flash-Lite, and Pro
- **Reflection Configs**: Baseline (0), Light (2-3), Deep (5+) iterations
- **Scenarios**: Simple web apps to complex enterprise systems
- **Evaluation**: Multi-dimensional quality + cost analysis
- **Repetitions**: Multiple runs for statistical significance

### Key Research Questions

1. **Quality**: At what point does reflection-enhanced Flash match Pro-level quality?
2. **Efficiency**: What are the cost and speed trade-offs?
3. **Task Dependency**: Which types of tasks benefit most from reflection?
4. **Convergence**: How many reflection cycles are optimal?

## Development

### Adding New Use Cases

1. Create use case directory: `use_cases/your_use_case/`
2. Implement required components:
   - `config.py`: Use case configuration and test scenarios
   - `agents.py`: Producer and Critic agents
   - `orchestrator.py`: Workflow orchestration
   - `evaluator.py`: Quality evaluation
   - `tools/`: Domain-specific tools

3. Follow naming conventions for automatic discovery

### Running Tests

```bash
# Run all tests
uv run pytest

# Run specific test category
uv run pytest tests/framework/
uv run pytest tests/use_cases/
```

### Code Quality

```bash
# Format code
uv run black .
uv run ruff check --fix .

# Type checking
uv run mypy .
```

## Configuration

### Environment Variables

Key configuration options in `.env`:

```bash
# Google AI
GOOGLE_API_KEY=your_key_here
DEFAULT_MODEL=gemini-2.5-flash
PRO_MODEL=gemini-2.5-pro

# Research
ENABLE_RESEARCH_MODE=true
MAX_REFLECTION_ITERATIONS=5
COST_TRACKING_ENABLED=true

# Database & Cache
RESEARCH_POSTGRES_DB=research_db
REDIS_URL=redis://localhost:6379
```

### Experiment Configuration

Experiments are configured via JSON files in `config/experiments/`:

```json
{
  "experiment_id": "system_design_pilot_001",
  "use_case": "system_design",
  "models_to_test": ["gemini-2.5-flash", "gemini-2.5-pro"],
  "reflection_configs": [0, 2, 3],
  "test_scenarios": ["..."],
  "repetitions": 2
}
```

## Research Applications

This framework is designed for:

- **Academic Research**: Empirical studies on AI reflection patterns
- **Industry Analysis**: Cost-benefit analysis of different AI approaches  
- **System Optimization**: Finding optimal model/reflection combinations
- **Expert Validation**: Incorporating domain expertise in AI evaluation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes following the established patterns
4. Add tests for new functionality
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{agentic_research_framework,
  title={Agentic Research Framework: Iterative Reflection vs Model Capability},
  author={Research Team},
  year={2024},
  url={https://github.com/your-org/agentic-research-framework}
}
```