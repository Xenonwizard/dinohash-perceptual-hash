# PyTorch Node Classifier

A Node.js package for running inference on PyTorch classification models.

## Installation

npm install pytorch-node-classifier


## Requirements

- Node.js v14 or later
- Your PyTorch model must be exported to TorchScript format

## Usage

```
const { downloadModel, loadModel, classify } = require('../index');
const path = require('path');

async function main() {
  try {
    const modelUrl = 'https://huggingface.co/backslashh/dinov2_vits14_reg_96bit/resolve/main/dinov2_vits14_reg_96bit.pt';
    const modelPath = path.join(__dirname, '../models/dinov2_vits14_reg_96bit.pt');
    const imagePath = path.join(__dirname, 'test.jpg');
    
    await downloadModel(modelUrl, modelPath);
    const model = loadModel(modelPath);

    const results = await hash(model, imagePath);
    console.log(results);

  } catch (error) {
    console.error('Error:', error.message);
  }
}

main();
```