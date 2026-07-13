# Library Translation Map

| From-scratch concept | PyTorch or Hugging Face abstraction |
| --- | --- |
| Token list | `Dataset` column or tensor dataset |
| Character tokenizer | `tokenizers.Tokenizer` or `AutoTokenizer` |
| `ModelConfig` | Transformers config object |
| `TinyGPT` | `AutoModelForCausalLM`-style model |
| Handmade training loop | `Trainer` or custom Accelerate loop |
| Handmade generation loop | `GenerationMixin.generate()` |
| Checkpoint dictionary | `save_pretrained()` and `from_pretrained()` |

The libraries reduce boilerplate. They do not remove the need to understand shapes, masks, loss shifting, optimizer behavior, device placement, or evaluation validity.
