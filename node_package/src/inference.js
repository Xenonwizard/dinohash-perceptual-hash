const { Jimp } = require('jimp');
const ort = require('onnxruntime-node');

/**
 * Preprocess an image for model input
 * @param {string|Buffer} image - Path to image or image buffer
 * @returns {Promise<Object>} - Preprocessed tensor
 */

async function preprocessImages(images) {
  const inputShape = [images.length, 3, 224, 224];
  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];
  
  const pixelData = new Float32Array(images.length * 3 * 224 * 224);
  
  for (let imgIndex = 0; imgIndex < images.length; imgIndex++) {
    const image = images[imgIndex];
    const imageData = await Jimp.read(image);
    imageData.resize({w:224, h:224});
    
    imageData.scan(0, 0, imageData.bitmap.width, imageData.bitmap.height, (x, y, idx) => {
      const pixelPos = (y * 224 + x);
      const imgOffset = imgIndex * 3 * 224 * 224;
      
      pixelData[imgOffset + pixelPos] = 
        (imageData.bitmap.data[idx + 0] / 255 - mean[0]) / std[0];
      
      pixelData[imgOffset + 1 * 224*224 + pixelPos] = 
        (imageData.bitmap.data[idx + 1] / 255 - mean[1]) / std[1];
      
      pixelData[imgOffset + 2 * 224*224 + pixelPos] = 
        (imageData.bitmap.data[idx + 2] / 255 - mean[2]) / std[2];
    });
  }

  return new ort.Tensor('float32', pixelData, inputShape);
}

/**
 * Runs inference on an image using a loaded model
 * @param {Object} model - Loaded TorchScript model
 * @param {string|Buffer} image - Path to image or image buffer
 * @returns {Promise<Array>} - Array of boolean values representing the hash
 */
async function hash(session, images) {
  try {
    const tensor = await preprocessImages(images);
    const feeds = { [session.inputNames[0]]: tensor };
    const results = await session.run(feeds);
    const outputTensor = results[session.outputNames[0]];

    const bits = outputTensor.dims[1];
    const flat_hash = Array.from(outputTensor.data, (x) => x>=0)

    const hashes = [];
    while(flat_hash.length) hashes.push(flat_hash.splice(0,bits));

    return hashes;
  } catch (error) {
    console.error('Error during inference:', error.message);
    throw error;
  }
}

module.exports = {
  hash
};