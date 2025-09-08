# Data Filtering Challenge - Team Sol submission

This repository provides scripts to filter and categorize subsets of the ClimbLab dataset using **regex heuristics** and **TF-IDF refinement**.  

## Installation

### Local setup

1. Clone this repository:

```bash
git clone https://github.com/M-A-Edwards/data-filtering-challenge.git
cd data-filtering-challenge
```
2. (Optional) Create a virtual environment:
```bash
python -m venv dfctf
source dfctf/bin/activate  
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```
After installation, you can run the filtering script with:
```bash
python3 filter.py
```

---

## Input

- Uses the detokenized OptimalScale/ClimbLab dataset.
- For replication, no local input file is required — the scripts load the dataset via the Hugging Face `datasets` library in streaming mode.

---

## Output

- JSON files in text-only format.
- By default saved to an `output/` directory (changeable in the script).

---

