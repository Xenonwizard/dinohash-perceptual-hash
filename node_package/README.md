# DINOHash

Official implementation of DINOHash, from https://www.arxiv.org/abs/2503.11195, in Node.js

## Installation
`npm install @proteus-labs/dinohash`

## Usage

```
const { downloadModel, loadModel, hash } = require('@proteus-labs/dinohash');
const path = require('path');

async function main() {
  try {
    const modelUrl = 'https://huggingface.co/backslashh/dinov2_vits14_reg_96bit/resolve/main/dinov2_vits14_reg_96bit.onnx';
    const modelPath = path.join(__dirname, './models/dinov2_vits14_reg_96bit.onnx');
    const imagePath = path.join(__dirname, 'test.png');
    
    await downloadModel(modelUrl, modelPath);
    const session = await loadModel(modelPath, device='cpu'); // can use 'cuda' for GPU inference if you have the right setup
    const results = await hash(session, imagePath);

    console.log(results);

  } catch (error) {
    console.error('Error:', error.message);
  }
}

main();
```