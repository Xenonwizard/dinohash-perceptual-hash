const axios = require('axios');
const fs = require('fs-extra');
const path = require('path');
const ProgressBar = require('progress');

/**
 * Downloads a PyTorch model from a HuggingFace
 * @param {string} url - URL to download the model from
 * @param {string} outputPath - Path to save the model to
 * @returns {Promise<string>} - Path to the downloaded model
 */
async function downloadModel(url, outputPath) {
  try {
    await fs.ensureDir(path.dirname(outputPath));
    
    if (await fs.pathExists(outputPath)) {
      console.log(`Model already exists at ${outputPath}`);
      return outputPath;
    }
    
    console.log(`Downloading model from ${url}...`);
    
    const { data, headers } = await axios({
      url,
      method: 'GET',
      responseType: 'stream'
    });
    
    const totalLength = parseInt(headers['content-length'], 10);
    
    const progressBar = new ProgressBar('Downloading [:bar] :percent :etas', {
      width: 40,
      complete: '=',
      incomplete: ' ',
      renderThrottle: 1,
      total: totalLength
    });
    
    const writer = fs.createWriteStream(outputPath);
    
    data.on('data', (chunk) => {
      progressBar.tick(chunk.length);
    });
    
    data.pipe(writer);
    
    return new Promise((resolve, reject) => {
      writer.on('finish', () => {
        console.log(`Model downloaded to ${outputPath}`);
        resolve(outputPath);
      });
      
      writer.on('error', (err) => {
        fs.unlink(outputPath);
        reject(err);
      });
    });
  } catch (error) {
    console.error('Error downloading model:', error.message);
    throw error;
  }
}

module.exports = {
  downloadModel
};
