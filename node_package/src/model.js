const ort = require('onnxruntime-node');
const fs = require('fs');

let sessionCache = null;

async function loadModel(modelPath, options = { cache: true }) {
  try {
    if (!fs.existsSync(modelPath)) {
      throw new Error(`Model not found at ${modelPath}`);
    }

    if (options.cache && sessionCache) {
      return sessionCache;
    }

    console.log(`Loading ONNX model from ${modelPath}`);
    
    const sessionOptions = {
      executionProviders: ['cpu'], // Change to 'cuda' for GPU support
      graphOptimizationLevel: 'all'
    };

    const session = await ort.InferenceSession.create(modelPath, sessionOptions);
    
    if (options.cache) {
      sessionCache = session;
    }

    return session;
  } catch (error) {
    console.error('Error loading ONNX model:', error.message);
    throw error;
  }
}

module.exports = { loadModel };
