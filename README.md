

# Small GPT Model Trained on Shakespeare Texts

This is a small Transformer-based GPT model trained on classical american writers works. The model can generate text in classical style, including poems, dialogues, and other writings.

---

## Model Details

* **Architecture:** GPT / Transformer
* **Number of parameters:** 2.5 million
* **Number of layers (`n_layers`):** 5
* **Hidden size (`d_model`):** 256
* **Number of attention heads (`n_heads`):** 4
* **Head dimension (`d_head`):** 64
* **Context (block) size:** 256 tokens
* **Batch size:** 64
* **Learning rate:** 6 × 10⁻⁴

---

## Training Details

* **Dataset:**  texts (preprocessed and tokenized)
* **Number of epochs:** 5000
* **Final train loss:** 1.1699
* **Final validation loss:** 1.4958
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
