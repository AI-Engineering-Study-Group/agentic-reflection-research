#!/usr/bin/env python3
"""
Reflection vs Baseline Experiment Script
Tests system design quality across models and reflection modes
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime
from typing import Dict, Any, List

# Test configuration
TEST_SCENARIO = "Design a scalable e-commerce platform that can handle Black Friday traffic spikes. Include microservices architecture, database design, caching strategy, and cost optimization for Google Cloud Platform."

MODELS = ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro"]
MODES = ["chat", "reflection"]  # chat = baseline, reflection = with critic

BASE_URL = "http://localhost:8001"

async def test_system_design(session: aiohttp.ClientSession, model: str, mode: str) -> Dict[str, Any]:
    """Test system design generation with specific model and mode"""
    
    print(f"🧪 Testing {model} in {mode} mode...")
    start_time = time.time()
    
    try:
        async with session.post(
            f"{BASE_URL}/chat",
            json={
                "message": TEST_SCENARIO,
                "use_case": "system_design",
                "model": model,
                "mode": mode,
                "reflection_iterations": 3 if mode == "reflection" else 0
            }
        ) as response:
            
            end_time = time.time()
            duration = end_time - start_time
            
            if response.status == 200:
                result = await response.json()
                
                return {
                    "model": model,
                    "mode": mode,
                    "success": True,
                    "duration_seconds": duration,
                    "response_length": len(result.get("response", "")),
                    "model_used": result.get("model_used", model),
                    "reflection_iterations": result.get("reflection_iterations_used", 0),
                    "quality_score": result.get("quality_score"),
                    "processing_time_seconds": result.get("processing_time_seconds", duration),
                    "response_preview": result.get("response", "")[:200] + "..." if len(result.get("response", "")) > 200 else result.get("response", "")
                }
            else:
                error_text = await response.text()
                return {
                    "model": model,
                    "mode": mode,
                    "success": False,
                    "duration_seconds": duration,
                    "error": f"HTTP {response.status}: {error_text}"
                }
                
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        return {
            "model": model,
            "mode": mode,
            "success": False,
            "duration_seconds": duration,
            "error": str(e)
        }

async def run_experiment():
    """Run comprehensive experiment across all model/mode combinations"""
    
    print("🚀 Starting Reflection vs Baseline Experiment")
    print(f"📝 Test Scenario: {TEST_SCENARIO}")
    print(f"🔬 Models: {MODELS}")
    print(f"🎯 Modes: {MODES}")
    print("=" * 80)
    
    results = []
    
    async with aiohttp.ClientSession() as session:
        # Test all combinations
        for model in MODELS:
            for mode in MODES:
                result = await test_system_design(session, model, mode)
                results.append(result)
                
                # Print immediate results
                if result["success"]:
                    print(f"✅ {model} ({mode}): {result['duration_seconds']:.1f}s, {result['response_length']} chars")
                    if result.get("reflection_iterations"):
                        print(f"   🔄 Reflection iterations: {result['reflection_iterations']}")
                else:
                    print(f"❌ {model} ({mode}): FAILED - {result.get('error', 'Unknown error')}")
                
                print("-" * 40)
                
                # Small delay between requests
                await asyncio.sleep(2)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"reflection_experiment_results_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump({
            "experiment_info": {
                "timestamp": timestamp,
                "test_scenario": TEST_SCENARIO,
                "models_tested": MODELS,
                "modes_tested": MODES,
                "total_tests": len(results)
            },
            "results": results
        }, f, indent=2)
    
    print(f"\n📊 EXPERIMENT COMPLETE!")
    print(f"📁 Results saved to: {filename}")
    
    # Summary analysis
    successful_results = [r for r in results if r["success"]]
    print(f"\n📈 SUMMARY:")
    print(f"   Total tests: {len(results)}")
    print(f"   Successful: {len(successful_results)}")
    print(f"   Failed: {len(results) - len(successful_results)}")
    
    if successful_results:
        print(f"\n⏱️  PERFORMANCE ANALYSIS:")
        
        # Group by mode
        baseline_results = [r for r in successful_results if r["mode"] == "chat"]
        reflection_results = [r for r in successful_results if r["mode"] == "reflection"]
        
        if baseline_results:
            avg_baseline_time = sum(r["duration_seconds"] for r in baseline_results) / len(baseline_results)
            avg_baseline_length = sum(r["response_length"] for r in baseline_results) / len(baseline_results)
            print(f"   📊 Baseline (chat): {avg_baseline_time:.1f}s avg, {avg_baseline_length:.0f} chars avg")
        
        if reflection_results:
            avg_reflection_time = sum(r["duration_seconds"] for r in reflection_results) / len(reflection_results)
            avg_reflection_length = sum(r["response_length"] for r in reflection_results) / len(reflection_results)
            print(f"   🔄 Reflection: {avg_reflection_time:.1f}s avg, {avg_reflection_length:.0f} chars avg")
        
        # Model comparison
        print(f"\n🎯 MODEL COMPARISON:")
        for model in MODELS:
            model_results = [r for r in successful_results if r["model"] == model]
            if model_results:
                avg_time = sum(r["duration_seconds"] for r in model_results) / len(model_results)
                avg_length = sum(r["response_length"] for r in model_results) / len(model_results)
                print(f"   {model}: {avg_time:.1f}s avg, {avg_length:.0f} chars avg")

if __name__ == "__main__":
    asyncio.run(run_experiment())
