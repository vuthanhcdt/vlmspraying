# Context-Aware Adaptive Pesticide Spraying for Agricultural Robots under Changing Weather and Terrain Using Vision–Language Models

**Authors:** Cong-Thanh Vu and Yen-Chen Liu from the [NRSL Lab](https://sites.google.com/site/yenchenliuncku/pub) at [National Cheng Kung University](https://web.ncku.edu.tw/index.php?Lang=en).

## Requirements

- Python 3.8 or later (Python 3.10 or newer is highly recommended)
- Required Python packages (listed in `requirements.txt`)
- A valid API key for the backend VLM service used by the system

## Installation

Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/vuthanhcdt/vlmspraying.git](https://github.com/vuthanhcdt/vlmspraying.git)
cd vlmspraying
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

Set your API key as an environment variable and execute the scripts:

```bash
export GOOGLE_API_KEY="your_api_key_here"
python3 crop_row_detection.py
python3 spraying_strategy.py
```

## Notes

- Ensure your API key is valid and has access to the chosen service.
- This codebase serves as an illustrative example, not the complete software system used for our real-world experimental testing.

## Citation

If you use this code in your research or publications, please cite our work:

```bibtex
@article{vu2026vlmspraying,
  title={Context-Aware Adaptive Pesticide Spraying for Agricultural Robots under Changing Weather and Terrain Using Vision–Language Models},
  author={Vu, Cong-Thanh and Liu, Yen-Chen},
  journal={...},
  volume={...},
  pages={...},
  year={2026}
}
```