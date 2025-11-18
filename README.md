# RAG-Reflect

# RAG-Reflect: Replication Package

This repository contains the replication package for **RAG-Reflect**, an agentic retrieval-augmented and self-reflective framework designed for:

- **Valid Comment–Edit Prediction (VCP)** on Stack Overflow  
- **Automatic Post Update (APU)** using comment-grounded code generation

The package is intended for **artifact evaluation (TOSEM)** and supports full reproduction of all quantitative and qualitative results reported in the submitted manuscript.

If you use this package in your work, please cite the associated TOSEM paper (citation to be added once published).

---

# 1. Project Structure

```
RAG-Reflect/
├── APU Task extention after Reflection(CodeGen).ipynb
├── VCP Feature Based Classification.ipynb
├── VCP_RAG(SOUP)_Full_Pipeplne.ipynb
├── VCP_RAG(SOUP)_Full_Pipeplne(python data).ipynb
├── VCP_RAG+Reflection (SOUP data).ipynb
├── VCP_RAG+Reflection (python data).ipynb
├── VCP_different prompting SOUP data.ipynb
├── Data/
│   ├── RAW Input/
│   │   ├── SOUP_train(RAW).json
│   │   ├── SOUP_test(RAW).json
│   │   └── test_python(RAW).json
│   ├── RAG_Input/
│   │   ├── rag_predicted_valid(SOUP)(Input for reflection).csv
│   │   └── rag_pred_valid_python(Input for reflection).csv
│   └── APU Task Extention (CodeGen) Input/
│       ├── rag_result_with_reflections(SOUP).csv
│       └── rag_result_with_reflections(python).csv
└── Prediction Result/
    ├── RAG output/
    │   ├── rag_responses_formated_python data.csv
    │   └── rag_responses_formatted_SOUP.csv
    ├── Final VCP output/
    │   ├── final_combined_data (RAG+Reflect)(SOUP).csv
    │   ├── rag_result_with_reflections(SOUP).csv
    │   └── rag_result_with_reflections(python).csv
    ├── Different prompting Result/
    │   ├── gpt4o_zero_shot_results.csv
    │   ├── predictions.csv
    │   ├── predictions_fewshot.csv
    │   ├── predictions_fewshot_(Qwen).csv
    │   ├── predictions_fewshot_CodeLlama70B.csv
    │   ├── predictions_fewshot_gemini_2.5.csv
    │   ├── predictions_fewshot_gpt4o.csv
    │   ├── predictions_cot_(Qwen).csv
    │   ├── predictions_cot_CodeLlama70B.csv
    │   ├── predictions_cot_codellama.csv
    │   ├── predictions_cot_gemini_flash.csv
    │   ├── predictions_cot_gpt4o.csv
    │   ├── predictions_zeroshot_(Qwen).csv
    │   ├── predictions_zeroshot_CodeLlama70B.csv
    │   ├── predictions_zeroshot_gemini.csv
    └── APU Task extention (code generation) Output/
        ├── rag_reflection_codegen_soup.csv
        └── rag_reflection_codegen_python.csv
```

---

# 2. Environment Setup

### Recommended Environment

- Python **3.9+**
- Jupyter Notebook / JupyterLab
- Works on Linux, macOS, and Windows

### Install Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn tqdm
```

> All experiments can be reproduced **without re-running LLM API calls**, since prediction files are pre-generated.

---

# 3. Dataset Overview

## 3.1 RAW Input

Located in `Data/RAW Input/`:

- `SOUP_train(RAW).json` — Java training data  
- `SOUP_test(RAW).json` — Java test data  
- `test_python(RAW).json` — Python dataset (PyVCP corpus)

## 3.2 RAG Inputs

Located in `Data/RAG_Input/`:

These are **valid predictions from RAG-only**, fed into the Reflection stage.

## 3.3 APU Inputs

Located in `Data/APU Task Extention (CodeGen) Input/`:

Used to run the **Automatic Post Update** evaluation.

---

# 4. Prediction Outputs

## 4.1 RAG Output

Formatted RAG responses (before reflection).

## 4.2 Final VCP Output

Files such as:

- `final_combined_data (RAG+Reflect)(SOUP).csv`

Contain:

- `gt_label`
- `pred_label`
- `comment`
- `code_before`
- `code_after`
- Reflection outputs

## 4.3 Prompting Baselines

Prompting results for GPT-4o, Gemini 2.5 Flash, CodeLlama, and Qwen.

## 4.4 APU Outputs

Used for evaluating the correctness of generated updates.

---

# 5. Notebooks Overview & RQ Mapping

| Notebook | Purpose | RQs |
|---------|---------|-----|
| VCP Feature Based Classification | Feature-based baselines | RQ2 |
| VCP_RAG(SOUP)_Full_Pipeplne | RAG pipeline | RQ1, RQ2 |
| VCP_RAG+Reflection (SOUP) | Full RAG-Reflect (Java) | RQ1, RQ2 |
| VCP_RAG+Reflection (python) | Full RAG-Reflect (Python) | Generalization |
| VCP_different prompting | Prompting strategies | RQ3 |
| APU Task extention after Reflection | Automatic Post Update | APU Analysis |

---

# 6. Reproducing Experiments

### Launch Jupyter
```bash
jupyter lab
```

### Run VCP Baselines  
Open:
- `VCP Feature Based Classification.ipynb`

### Run RAG and RAG-Reflect  
Open:
- `VCP_RAG(SOUP)_Full_Pipeplne.ipynb`
- `VCP_RAG+Reflection (SOUP data).ipynb`

### Prompting Baselines  
Open:
- `VCP_different prompting SOUP data.ipynb`

### Automatic Post Update  
Open:
- `APU Task extention after Reflection(CodeGen).ipynb`

---

# 7. Citation

```bibtex
@article{ragreflect_tosem,
  author  = {Shanto et al.},
  title   = {RAG-Reflect: An Agentic Retrieval-Augmented and Self-Reflective Framework for Valid Comment--Edit Prediction},
  journal = {ACM Transactions on Software Engineering and Methodology (Sumbitted)},
  year    = {2025},
  note    = {Replication package available at <URL>},
}
```

---

# 8. Contact

For questions about this replication package, contact:

**Mehedi Hasan Shanto**  
University of Windsor  
shanto1@uwindsor.ca
