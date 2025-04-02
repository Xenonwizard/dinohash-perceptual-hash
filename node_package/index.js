const downloader = require('./src/downloader');
const model = require('./src/model');
const inference = require('./src/inference');

module.exports = {
  downloadModel: downloader.downloadModel,
  loadModel: model.loadModel,
  hash: inference.hash
};
