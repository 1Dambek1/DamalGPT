

# Small GPT Model Trained on pila dataset 

This is a small Transformer-based GPT model trained on science papers and wikipedia articles. The model can generate articles.

---

## Model Details

* **Architecture:** GPT / Transformer
* **Number of parameters:** 95 million
* **Number of layers (`n_layers`):** 10
* **Hidden size (`d_model`):** 256
* **Number of attention heads (`n_heads`):** 8
* **Head dimension (`d_head`):** 256
* **Context (block) size:** 768 tokens
* **Batch size:** 64
* **Learning rate:** 6 × 10⁻⁴

---

## Training Details

<<<<<<< HEAD
* **Dataset:** wikipedia articles + science papers (preprocessed and tokenized)
=======
* **Dataset:**  texts (preprocessed and tokenized)
>>>>>>> fa05ae548d2108628d6780335c7d6cef494664a8
* **Number of epochs:** 5000
* **Final train loss:** 1.01
* **Final validation loss:** 1.06
* **Optimizer:** AdamW

> The model was trained using a causal language modeling objective, predicting the next character/token given previous context.

---

## Pretrained Weights

* **File:** `gpt-weights.tar`
* **Format:** PyTorch checkpoint
* **Usage:** Can be loaded to generate Shakespearean-style text without retraining.

---

### Generate Text

file example.py
```python
from model import device, model, decode
context = torch.zeros((1, 1), dtype=torch.long, device=device)  # starting token
generated = model.generate(context, max_new_tokens=500)
text = decode(generated[0].tolist())  # decode function from your tokenizer
print(text)
```

---

## Notes

* The model is **small**, designed for future tests on different seq , for example basket game predictions.
* In closest future i gonna update tokenization and add more layers, when get more resourses to train it(My pc dont wanna die)
