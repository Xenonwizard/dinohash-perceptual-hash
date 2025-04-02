const torch = require('@idn/torchjs');
const Jimp = require('jimp');
const fs = require('fs-extra');
const path = require('path');

/**
 * Preprocess an image for model input
 * @param {string|Buffer} image - Path to image or image buffer
 * @param {Object} options - Preprocessing options
 * @param {number[]} options.inputShape - Shape of the input tensor [channels, height, width]
 * @param {number[]} options.mean - Mean values for normalization [r, g, b]
 * @param {number[]} options.std - Standard deviation values for normalization [r, g, b]
 * @returns {Promise<Object>} - Preprocessed tensor
 */
async function preprocessImage(image) {
  const inputShape = [3, 224, 224];
  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];
  
  // Load image using Jimp
  const jimpImage = typeof image === 'string' 
    ? await Jimp.read(image)
    : await Jimp.read(Buffer.from(image));
  
  jimpImage.resize(inputShape[2], inputShape[1]);
  
  // Create a Float32Array to hold the image data
  const buffer = new Float32Array(inputShape[0] * inputShape[1] * inputShape[2]);
  
  // Convert image to normalized values
  let idx = 0;
  for (let c = 0; c < inputShape[0]; c++) {
    for (let h = 0; h < inputShape[1]; h++) {
      for (let w = 0; w < inputShape[2]; w++) {
        const pixel = Jimp.intToRGBA(jimpImage.getPixelColor(w, h));
        const value = (pixel[c] / 255.0 - mean[c]) / std[c];
        buffer[idx++] = value;
      }
    }
  }
  
  return torch.tensor(buffer, { shape: inputShape });
}

/**
 * Runs inference on an image using a loaded model
 * @param {Object} model - Loaded TorchScript model
 * @param {string|Buffer} image - Path to image or image buffer
 * @param {Object} options - Inference options
 * @param {Object} options.preprocess - Preprocessing options
 * @returns {Promise<Object>} - Hashing results
 */
async function hash(model, image, options = {}) {
  try {
    const tensor = await preprocessImage(image, options.preprocess);
    const batchedTensor = tensor.unsqueeze(0);
    const output = model.forward(batchedTensor);
    const predictions = output.toObject();
    
    // Get the top predictions
    const scores = predictions.data;
    const indices = Array.from({ length: scores.length }, (_, i) => i)
      .sort((a, b) => scores[b] - scores[a]);
    
    // Load labels if provided
    let labels = null;
    if (options.labelsPath && fs.existsSync(options.labelsPath)) {
      const labelsContent = await fs.readFile(options.labelsPath, 'utf-8');
      labels = labelsContent.split('\n').filter(Boolean);
    }
    
    // Format results
    const results = indices.map((idx) => ({
      classIndex: idx,
      className: labels ? labels[idx] : `Class ${idx}`,
      score: scores[idx]
    }));
    
    return results;
  } catch (error) {
    console.error('Error during inference:', error.message);
    throw error;
  }
}

module.exports = {
  hash
};
