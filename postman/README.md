# Postman Collection for Agentic Research Framework

This directory contains Postman collections and environments for testing the Agentic Reflection Research Framework API.

## Files

- `Agentic-Research-Framework.postman_collection.json` - Main collection with all API endpoints
- `Agentic-Research-Framework.postman_environment.json` - Environment variables for different configurations
- `README.md` - This documentation file

## Setup

1. **Import Collection**: Import `Agentic-Research-Framework.postman_collection.json` into Postman
2. **Import Environment**: Import `Agentic-Research-Framework.postman_environment.json` into Postman
3. **Select Environment**: Make sure the "Agentic Research Framework Environment" is selected in the top-right dropdown
4. **Start API Server**: Ensure the API server is running on `http://localhost:8001`

## Environment Variables

| Variable | Default Value | Description |
|----------|---------------|-------------|
| `base_url` | `http://localhost:8001` | Base URL for the API |
| `default_model` | `gemini-2.5-flash-lite` | Default model for testing |
| `pro_model` | `gemini-2.5-pro` | Pro model for advanced testing |
| `experiment_id` | `test-experiment-123` | Sample experiment ID |

## Collection Structure

### Core Endpoints
- **Health Check** - Verify API is running
- **Chat - Baseline Mode** - Test baseline chat functionality
- **Chat - Reflection Mode** - Test reflection mode with iterations
- **Use Case Chat** - Test system design use case specifically

### Experiment Endpoints
- **Experiment - Multi-Model Comparison** - Run comprehensive experiments
- **Get Experiments** - List all experiments
- **Get Experiment Results** - Retrieve specific experiment results
- **Compare Modes** - Compare baseline vs reflection modes

### Model Testing
- **Flash Lite Model** - Test with gemini-2.5-flash-lite
- **Flash Model** - Test with gemini-2.5-flash  
- **Pro Model** - Test with gemini-2.5-pro

### Research Framework Tests
- **Baseline vs Reflection** tests for different models
- Comprehensive system design scenarios

## Usage Examples

### Quick Health Check
```
GET {{base_url}}/health
```

### Test Baseline Mode
```
POST {{base_url}}/use-cases/chat
{
  "message": "Design a microservices architecture for e-commerce...",
  "use_case": "system_design",
  "model": "{{default_model}}",
  "mode": "baseline"
}
```

### Test Reflection Mode
```
POST {{base_url}}/use-cases/chat
{
  "message": "Design a microservices architecture for e-commerce...",
  "use_case": "system_design", 
  "model": "{{default_model}}",
  "mode": "reflection",
  "reflection_iterations": 3
}
```

### Run Experiment
```
POST {{base_url}}/experiments
{
  "experiment_name": "E-commerce Modernization Study",
  "scenarios": ["Design a microservices architecture..."],
  "models": ["{{default_model}}", "{{pro_model}}"],
  "reflection_configs": [0, 3],
  "repetitions": 1
}
```

## Testing Workflow

1. **Start with Health Check** - Ensure API is running
2. **Test Individual Models** - Use the model-specific requests to test each model
3. **Compare Modes** - Use the comparison endpoint to see baseline vs reflection
4. **Run Experiments** - Use the experiment endpoint for comprehensive testing
5. **Review Results** - Check experiment results and quality scores

## Tips

- Use the environment variables to easily switch between different configurations
- The collection includes sample messages that demonstrate the system's capabilities
- Each request includes proper headers and JSON formatting
- Results include quality scores and detailed responses for analysis

## Troubleshooting

- **Connection Refused**: Ensure the API server is running on the correct port
- **404 Errors**: Check that the API endpoints match your server configuration
- **Timeout Errors**: Some reflection mode requests may take longer due to multiple iterations
- **Model Errors**: Verify that the specified models are available in your Google AI configuration
