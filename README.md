# GarmentGPT: End-to-End Garment Generation from a Single Image


**Paper:** [Coming Soon]

This repository contains the official implementation for **GarmentGPT**, a project that pioneers an end-to-end pipeline for generating detailed 3D garment geometry (in GCD format) directly from a single 2D image. Our approach leverages a powerful vision-language model to interpret the image and generate a sequence of discrete tokens, which are then decoded by specialized VQ-VAE-based models into precise geometric data, including panels, edges, and stitches.

## 🌟 Features

- **End-to-End Pipeline**: From a single image to a structured 3D garment JSON file.
- **High-Fidelity Geometry**: Generates detailed panel shapes, curves, and 3D positioning.
- **Powered by vLLM**: Utilizes the vLLM engine for fast and efficient inference with the core vision-language model.
- **Modular Decoder Architecture**: Employs separate, specialized decoders for edge geometry and panel rotation/translation, ensuring high-quality output.
- **Open & Reproducible**: Code and models are provided to fully reproduce our results.

## 🚀 Getting Started

Follow these steps to set up the environment and run the inference pipeline.

### 1. Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA 11.8+
- Git and Git LFS

Also setup the llamafactory
```bash
git clone https://github.com/hiyouga/LLaMA-Factory
cd LLaMA-Factory
pip install -e .

pip install torch
pip install -r requirements.txt
```

### 2. Installation

First, clone this repository to your local machine:

```bash
git clone https://github.com/YourUsername/GarmentGPT.git
cd GarmentGPT
```
<!-- TODO: 将上面的 'YourUsername' 替换为您的 GitHub 用户名 -->

Next, set up the Python environment. We strongly recommend using `conda` to create a clean environment.

```bash
# Create and activate a new conda environment
conda create -n garmentgpt python=3.10
conda activate garmentgpt

# Install PyTorch with the correct CUDA version (example for CUDA 12.1)
# Please visit the PyTorch website for the command matching your system.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install the remaining dependencies from requirements.txt
pip install -r requirements.txt
```

**Important Note on LLaMA-Factory:**

This project was developed using certain utilities from the `LLaMA-Factory` library. Due to fast-evolving dependencies, we recommend installing it separately from its official repository to ensure compatibility.

```bash
# In a separate directory, clone and install LLaMA-Factory in editable mode
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .
cd .. # Return to the GarmentGPT directory
```

### 3. Running Inference

The core logic is encapsulated in `main.py`. This script handles everything from model loading to final JSON generation.

The required models (LLM, Codec, and RT decoders) will be automatically downloaded from our Hugging Face Hub repository upon first run. [ChimerAI/GarmentGPT](https://huggingface.co/ChimerAI/GarmentGPT).

To run inference on a single image:

download the checkpoints from the huggingface and put them into the 'checkpoints' folder with

```bash
export DISABLE_VERSION_CHECK=1
python main.py \
  --llm_model_path "checkpoints/vlm/checkpoint-12844" \
  --codec_config_path "configs/config_vq1024_resres_aug_decay0.99_q5_gcd_nl8_ld512.yaml" \
  --rt_config_path "configs/config_rt_euler.yaml" \
  --image_path "rand_YHLMQOMVPI_render_front_cutted.png" \
  --output_path "./my_garment.json" \
  --device "cuda:0"
```

**Arguments:**
- `--llm_model_path`: The VLM path used in our method.
- `--codec_config_path`: VQVAE setting
- `--rt_config_path`: VQVAE setting
- `--output_path`: Path where the final GCD-formatted JSON will be saved.
- `--device`: The CUDA device to use for the decoder models (e.g., "cuda:0").

The script will first download the necessary models to your local Hugging Face cache directory, then proceed with the three-stage generation process:
1.  **LLM Inference**: The image is processed to generate a sequence of discrete tokens.
2.  **Text Parsing**: The token sequence is parsed into a structured format containing panel names, edge indices, and location indices.
3.  **Geometric Decoding**: The indices are converted into floating-point geometric data (vertices, curves, etc.) to produce the final output.

### Example Output

After running the script, you will find a `.json` file at your specified output path. This file contains the full geometric description of the garment, ready for use in 3D modeling or simulation software.

```json
{
    "pattern": {
        "panels": {
            "front_panel": {
                "translation": [0.0, 0.1, 0.0],
                "rotation": [0.99, 0.0, 0.0, 0.01],
                "vertices": [
                    [ -0.2, 0.5 ],
                    [ 0.2, 0.5 ],
                    [ 0.2, -0.5 ],
                    [ -0.2, -0.5 ]
                ],
                "edges": [
                    { "endpoints": },
                    { "endpoints":, "curvature": { "type": "cubic", "params": [[0.33, 0.1], [0.66, 0.1]] } },
                    { "endpoints": },
                    { "endpoints": }
                ]
            }
        },
        "stitches": [
            [
                { "panel": "front_panel", "edge": 3 },
                { "panel": "back_panel", "edge": 1 }
            ]
        ],
        "panel_order": ["front_panel", "back_panel"]
    },
    "source_image": "/path/to/your/input_image.jpg",
    "raw_llm_output": "<SoG>..."
}
```

## 🙏 Acknowledgements

- Our work builds upon the powerful open-source libraries from [Hugging Face](https://huggingface.co/).
- The core inference engine is powered by [vLLM](https://github.com/vllm-project/vllm).
- We thank the authors of [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for their excellent model training and inference framework.

## 📜 License

This project is licensed under the [Apache 2.0 License](LICENSE). Please see the `LICENSE` file for details.
