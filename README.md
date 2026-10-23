# Can simple embeddings align a model?

Code for the blog post [Can you align with one added embedding (e.g., "safety" token) vector?](https://stachyra.io/posts/safety-concept.html)

![Misaligned, aligned and incoherent answers as the strength of "safety" rises](figures/headline.png)

## The question

Take a model that a LoRA fine-tune has made misaligned, add a scaled-down copy of the embedding of the word **"safety"** to its tokens, and see whether it misbehaves less. The word **"rabbit"** is the control: it shows what adding *any* word does.

- **Model:** `Qwen/Qwen2.5-14B-Instruct` with the misaligning LoRA `ModelOrganismsForEM/Qwen2.5-14B-Instruct_bad-medical-advice` from *Model Organisms for Emergent Misalignment* (Turner, Soligo et al., 2025).
- **Questions:** 4 free-form questions from *Emergent Misalignment* (Betley et al., 2025), 100 answers each per condition.
- **Judge:** `Qwen/Qwen2.5-32B-Instruct` labels every answer **aligned**, **misaligned** or **incoherent**.
- **Fair strength picking:** each strength is chosen on samples 0–49 (most "aligned" answers) and reported on samples 50–99.

## Results so far

- **At the input** (embedding layer): adding "safety" lowers misalignment by about 10 percentage points at best (76% → 65.5%), and only near the strength where answers start to break down. Otherwise it does nothing, the same as "rabbit".
- **Inside the model** (residual stream): at layers 12 and 24 misalignment drops much more (to 18% and 40%), but "rabbit" does nearly as well (25% and 55%). Most of the effect comes from nudging the model at all, not from the meaning of "safety".

## The notebook

[`safety_concept.ipynb`](safety_concept.ipynb) is organised as a series of questions, and is saved with the outputs of the run:

0. **Baselines:** the misaligned model vs. the base model.
1. **What happens when you add the word "safety"?** At the embedding layer, to the prompt only or to prompt + generated tokens.
2. **What happens when you add it inside the model?** To the residual stream after layers 12, 24 and 36.
3. **What about other words?** "alignment", "virtuous", "healthy".
4. **Does it beat a learned direction?** A misalignment direction learned from the model's own answers.
5. **Is more better?** The best word turned up until the model breaks.

## Running it

You need one GPU with about **80 GB** of memory. The 14B model (~30 GB) and the 32B judge (~65 GB) are loaded one at a time.

```bash
pip install -r requirements.txt
jupyter notebook safety_concept.ipynb
```

Every condition is saved to `runs/<name>.csv` as it finishes, so re-running a cell skips work that's already done. A full run generates and judges tens of thousands of answers; the notebook's "Run size" cell prints the exact count.

## References

- Betley et al. (2025). [Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs](https://arxiv.org/abs/2502.17424).
- Turner, Soligo et al. (2025). [Model Organisms for Emergent Misalignment](https://arxiv.org/abs/2506.11613).
