"""
Flask Web Server for Bank Fraud Detection Dashboard
Run this to launch the beautiful web interface.
"""
from flask import Flask, render_template_string, jsonify
import json
from pathlib import Path

app = Flask(__name__)

@app.route('/')
def dashboard():
    # Read the HTML file
    with open('web_dashboard.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    return html_content

@app.route('/api/metrics')
def get_metrics():
    """API endpoint to get training metrics"""
    try:
        with open('results/final_metrics.json', 'r') as f:
            metrics = json.load(f)
        return jsonify(metrics)
    except:
        return jsonify({"error": "No training data found"}), 404

@app.route('/api/training-history')
def get_training_history():
    """API endpoint to get full training history"""
    try:
        with open('results/training_results.json', 'r') as f:
            history = json.load(f)
        return jsonify(history)
    except:
        return jsonify({"error": "No training data found"}), 404

if __name__ == '__main__':
    print("=" * 80)
    print("🚀 BANK FRAUD DETECTION DASHBOARD")
    print("=" * 80)
    print("\n✨ Starting beautiful web interface...")
    print("\n🌐 Open your browser and navigate to:")
    print("   👉 http://localhost:5000")
    print("\n📊 Dashboard Features:")
    print("   • Animated metrics and charts")
    print("   • Real-time training visualization")
    print("   • Confusion matrix analysis")
    print("   • Beautiful gradient design")
    print("\n⏹️  Press CTRL+C to stop the server")
    print("=" * 80 + "\n")
    
    app.run(debug=False, host='0.0.0.0', port=5000)
