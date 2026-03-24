# ECG Project Execution Guide (CLAUDE.md)

Project: ECG-Based Detection of Myocardial Infarction and Ischemic Abnormalities Using Deep Learning

---

## 🔷 WORKFLOW ORCHESTRATION

### 1. Plan Mode (MANDATORY for non-trivial work)
Use plan mode for:
- Model changes
- Preprocessing updates
- Backend/API changes
- Explainability logic

Steps:
1. Define goal (e.g., improve accuracy)
2. List exact changes
3. Predict expected outcome
4. Then implement

If results are worse → STOP → revert → re-plan

---

### 2. Focus Strategy (NO DISTRACTIONS)
Work in isolated blocks:
- Model improvement
- Explainability
- Backend
- UI

❌ Do NOT mix everything in one session  
❌ Do NOT add new features randomly  

---

### 3. Self-Improvement Loop
After every bug or mistake:
- Write:
  - What went wrong
  - Why it happened
  - Fix rule

Example:
- Issue: Model overfitting  
- Rule: Always check validation loss before saving model  

Review these before working again.

---

### 4. Verification Before Completion (CRITICAL)
Never say "done" unless:

- Model:
  - Validation accuracy checked
  - No overfitting
- API:
  - Endpoint returns correct JSON
- UI:
  - Upload works
  - Output displays correctly
- Explainability:
  - Heatmap aligns with signal

Ask:
> “Will examiner break this easily?”

---

### 5. Keep It Simple (VERY IMPORTANT)
Always prefer:

✔ Simple CNN over complex architectures  
✔ Rule-based explanation over LLM  
✔ Stable pipeline over new features  

Avoid:
- Transformers
- Multi-model ensembles
- Overengineering

---

### 6. Bug Fixing Rule
When something breaks:
1. Check logs
2. Check input shape
3. Check preprocessing
4. Check model loading

Fix directly — don’t guess.

---

## 🔷 MODEL DEVELOPMENT RULES

### Allowed Improvements (SAFE)
- Train on more PTB-XL data
- Class balancing
- Slight CNN tuning (filters, dropout)
- Learning rate tuning

### NOT Allowed (TIME WASTE)
- Transformers
- GANs
- Reinforcement learning
- Complex multi-stage pipelines

---

### Model Validation Checklist
- Accuracy ≥ previous model
- No major class bias
- Stable predictions
- Works with real input files

---

## 🔷 EXPLAINABILITY RULES

Use:
- Grad-CAM (PRIMARY)
- Saliency maps (SECONDARY)
- Rule-based explanation (TEXT)

Explanation must:
- Match prediction
- Be consistent
- Avoid medical claims beyond scope

❌ Do NOT:
- Add fake clinical interpretation
- Use uncontrolled LLM outputs

---

## 🔷 BACKEND (FastAPI)

Checklist:
- `/analyze` works for:
  - CSV
  - DAT + HEA
  - Image
- Returns:
  - prediction
  - probability
  - saliency
  - gradcam
  - report

Always test with:
- Valid file
- Invalid file

---

## 🔷 FRONTEND (Streamlit)

Must display:
- Uploaded ECG
- Prediction
- Confidence
- Heatmap
- Explanation
- Download report

UI rule:
- Clean > fancy

---

## 🔷 REPORT GENERATION

PDF must include:
- Prediction
- Confidence score
- Visual explanation
- Plain explanation
- Clinical-style explanation
- Disclaimer

---

## 🔷 RESEARCH PAPER RULES

Must clearly show:
1. Problem statement
2. Dataset (PTB-XL)
3. Method (1D CNN)
4. Explainability (Grad-CAM)
5. Results (accuracy ~0.75+)
6. System architecture diagram

---

## 🔷 VIVA PREPARATION RULES

You MUST confidently answer:

- Why 1D CNN?
→ Time-series data

- Why Lead II?
→ Strong clinical relevance

- Why not 95% accuracy?
→ Real-world ECG complexity + dataset variability

- What is Grad-CAM?
→ Highlights important signal regions influencing prediction

- Why rule-based explanation?
→ Deterministic, safe, explainable

---

## 🔷 FINAL PHASE STRATEGY

### Phase 1
- Improve model slightly
- Fix bugs

### Phase 2
- Stabilize UI + API

### Phase 3
- Write paper

### Phase 4
- Prepare viva

---

## 🔷 CORE PRINCIPLES

- Simplicity > Complexity  
- Stability > Features  
- Understanding > Accuracy  
- Completion > Perfection  

---

## 🔷 RED FLAGS (STOP IF YOU DO THIS)

❌ Adding LLM explainability  
❌ Changing architecture completely  
❌ Adding new modules near submission  
❌ Ignoring bugs for new features  

---

## 🔷 FINAL RULE

> “Would this impress an examiner AND work reliably in demo?”

If YES → proceed  
If NO → simplify