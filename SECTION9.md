# Section 9: Inference + sampling

Generates 20 new names by sampling from the trained model.

## Temperature

`temperature = 0.5` controls the "creativity" of generated text. Lower values make the model more confident (sharper distribution), higher values make it more random. Applied by dividing logits by temperature before softmax.

## Sampling loop

For each of the 20 samples:

1. Create fresh KV caches (`make-vector n-layer '()`)
2. Start with `token-id = BOS`
3. For each position (up to `block-size`):
   - Run `gpt` to get logits
   - Divide logits by temperature: `(map (lambda (l) (v/ l temperature)) logits)`
   - Apply `softmax` to get probabilities
   - Extract plain numbers with `(map value-data probs)`
   - Sample next token with `weighted-choice`
   - If BOS: end of name, print result
   - Otherwise: cons the character and continue
4. Characters are accumulated in reverse (via `cons`) and reversed at print time with `list->string (reverse chars)`

## Decisions

- **`v/` for temperature scaling**. Each logit is divided by `temperature` (a plain number) through the autograd graph. This isn't necessary for inference (we don't backpropagate through sampling), but it keeps the code simple and faithful to the Python. The cost is negligible — 27 extra Value nodes per position.
- **`cons` + `reverse` for character accumulation**. Prepending with `cons` is O(1) per character, `reverse` is O(n) once at the end. More idiomatic than `append` which would be O(n) per character.
- **`weighted-choice` for sampling**. Reuses the PRNG helper from section 2. The Python uses `random.choices` which does the same cumulative distribution sampling.
- **No autograd needed during inference**. The forward pass still builds a computation graph (creating Value nodes), but we never call `backward`. This is wasteful but matches the Python. A production implementation would have a separate inference-only forward pass that operates on plain numbers.
