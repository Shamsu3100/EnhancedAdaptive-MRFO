# EA-MRFO: Enhanced Adaptive Manta Ray Foraging Optimizer

Official implementation of the paper:

> **EA-MRFO: An Enhanced Adaptive Manta Ray Foraging Optimizer for Complex Optimization Challenge**  
> *Adamu, S., et al., 2025* (Under Review)

This repository contains a comprehensive implementation of EA-MRFO, a state-of-the-art optimization algorithm evaluated using the CEC2017 benchmark suite.

---

## 🔧 Features

- 🧬 **Hybrid Initialization**: Combines Latin Hypercube Sampling (LHS), Lévy Flights, and chaotic maps for diverse population seeding.
- 🌀 **Chaotic Adaptation**: Leverages five distinct chaotic maps (logistic, sinusoidal, tent, Ikeda, Gauss) to enhance exploration.
- 🧠 **Multi-bank Memory Architecture**: Incorporates short-term, long-term, and diversity-driven solution memories for guided exploitation.
- 🧭 **Function-aware Local Search**: Adapts search strategies based on function category (unimodal, multimodal, hybrid, composition).
- ♻️ **Adaptive Restart Strategy**: Dynamically restarts individuals using memory, Lévy flight, and category-specific logic to avoid stagnation.
- 📉 **Bias-aware Evaluation**: Automatically adjusts for known CEC2017 biases for accurate performance reporting.
- ⚙️ **Adaptive Foraging Phases**: Implements enhanced MRFO operations across chain foraging, cyclone foraging, and somersault strategies.
- 📊 **Function-specific Parameter Tuning**: Parameters are dynamically adjusted based on iteration progress and function category.
- 🧵 **Parallel Execution Support**: Utilizes multi-core CPUs for simultaneous multi-run evaluations using `joblib`.
- 📈 **CEC2017 Benchmark Integration**: Full implementation and evaluation across 30 standardized test functions.

---

## 📦 Dependencies

Install with pip:

```bash
pip install -r requirements.txt
```

Dependencies:
- Python 3.8+
- NumPy
- pandas
- matplotlib
- scipy
- joblib

> ⚠️ Requires `cec17_test_func` which can be found from the [CEC2017 benchmark resources](https://www3.ntu.edu.sg/home/EPNSugan/index_files/CEC2017/CEC2017.htm). Place it in `src/`.

---

## 🚀 How to Run

```bash
cd experiments
python main_run.py
```

This script runs all 30 CEC2017 functions in parallel and logs results to `/data/results/`.

---

## 📄 License

MIT License

---

## 🔗 Citation

If you use this code, please cite:

```bibtex
@article{adamu2025eamrfo,
  title={EA-MRFO: An Enhanced Adaptive Manta Ray Foraging Optimizer for Complex Optimization Challenge},
  author={Adamu, S. and others},
  year={2025},
  journal={Under Review}
}
```
