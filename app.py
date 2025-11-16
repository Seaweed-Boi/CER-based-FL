"""
FastAPI Server for Bank Fraud Detection Dashboard
Beautiful, modern web interface with animations
"""
from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import json
from pathlib import Path
import uvicorn

app = FastAPI(title="Bank Fraud Detection AI", version="1.0.0")

# Load training data
def load_results():
    try:
        with open('results/final_metrics.json', 'r') as f:
            final_metrics = json.load(f)
        with open('results/training_results.json', 'r') as f:
            training_history = json.load(f)
        return final_metrics, training_history
    except:
        return None, None

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Main dashboard page"""
    with open('web_dashboard_cer.html', 'r', encoding='utf-8') as f:
        return f.read()

@app.get("/api/metrics")
async def get_metrics():
    """API endpoint for final metrics"""
    final_metrics, _ = load_results()
    if final_metrics:
        return JSONResponse(content=final_metrics)
    return JSONResponse(content={"error": "No data found"}, status_code=404)

@app.get("/api/training-history")
async def get_training_history():
    """API endpoint for training history"""
    _, training_history = load_results()
    if training_history:
        return JSONResponse(content=training_history)
    return JSONResponse(content={"error": "No data found"}, status_code=404)

@app.get("/api/status")
async def get_status():
    """API endpoint for system status"""
    final_metrics, training_history = load_results()
    return JSONResponse(content={
        "status": "online" if final_metrics else "offline",
        "model_trained": final_metrics is not None,
        "total_epochs": len(training_history) if training_history else 0,
        "accuracy": final_metrics.get("final_accuracy", 0) if final_metrics else 0
    })

if __name__ == "__main__":
    print("=" * 80)
    print("🚀 BANK FRAUD DETECTION - AI DASHBOARD")
    print("=" * 80)
    print("\n✨ Launching beautiful web interface with FastAPI...")
    print("\n🌐 Dashboard URL:")
    print("   👉 http://localhost:8000")
    print("\n📊 Features:")
    print("   • Stunning animated UI with gradients")
    print("   • Real-time metrics visualization")
    print("   • Interactive charts with Chart.js")
    print("   • Responsive design for all devices")
    print("   • API endpoints for data access")
    print("\n📡 API Endpoints:")
    print("   • GET /api/metrics - Final model metrics")
    print("   • GET /api/training-history - Training data")
    print("   • GET /api/status - System status")
    print("\n⏹️  Press CTRL+C to stop")
    print("=" * 80 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
