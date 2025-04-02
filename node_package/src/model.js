const torch = require('@idn/torchjs');
const fs = require('fs-extra');

let modelCache = null;

/**
 * Loads a PyTorch model from a file path
 * @param {string} modelPath - Path to the TorchScript model file (.pt)
 * @param {Object} options - Options for loading the model
 * @param {boolean} options.cache - Whether to cache the model in memory (default: true)
 * @returns {Object} - Loaded model
 */
function loadModel(modelPath, options = { cache: true }) {
  try {
    // Check if model exists
    if (!fs.existsSync(modelPath)) {
      throw new Error(`Model not found at ${modelPath}`);
    }
    
    // Return cached model if available
    if (options.cache && modelCache) {
      return modelCache;
    }
    
    console.log(`Loading model from ${modelPath}...`);
    
    // Load the model using TorchJS
    const model = new torch.ScriptModule(modelPath);
    
    // Cache the model if caching is enabled
    if (options.cache) {
      modelCache = model;
    }
    
    console.log('Model loaded successfully');
    return model;
  } catch (error) {
    console.error('Error loading model:', error.message);
    throw error;
  }
}

module.exports = {
  loadModel
};
