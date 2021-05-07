Wietse de Vries • Martijn Bartelds • Malvina Nissim • Martijn Wieling

# Adapting Monolingual Models: Data can be Scarce when Language Similarity is High

This repository contains everything that is needed to replicate the results in the paper:

📝 [Adapting Monolingual Models: Data can be Scarce when Language Similarity is High](https://arxiv.org/abs/2105.02855)

## Models

The best fine-tuned models for Gronings and West Frisian are available on the HuggingFace model hub:

### Lexical layers
These models are identical to [BERTje](https://github.com/wietsedv/bertje), but with different lexical layers (`bert.embeddings.word_embeddings`).

 - 🤗 [`GroNLP/bert-base-dutch-cased`](https://huggingface.co/GroNLP/bert-base-dutch-cased) (Dutch; source language)
 - 🤗 [`GroNLP/bert-base-dutch-cased-gronings`](https://huggingface.co/GroNLP/bert-base-dutch-cased-gronings) (Gronings)
 - 🤗 [`GroNLP/bert-base-dutch-cased-frisian`](https://huggingface.co/GroNLP/bert-base-dutch-cased-frisian) (West Frisian)

### POS tagging
These models share the same fine-tuned Transformer layers + classification head, but with the retrained lexical layers from the models above.

 - 🤗 [`GroNLP/bert-base-dutch-cased-upos-alpino`](https://huggingface.co/GroNLP/bert-base-dutch-cased-upos-alpino) (Dutch)
 - 🤗 [`GroNLP/bert-base-dutch-cased-upos-alpino-gronings`](https://huggingface.co/GroNLP/bert-base-dutch-cased-upos-alpino-gronings) (Gronings)
 - 🤗 [`GroNLP/bert-base-dutch-cased-upos-alpino-frisian`](https://huggingface.co/GroNLP/bert-base-dutch-cased-upos-alpino-frisian) (West Frisian)

## Development

Conda/[mamba](https://github.com/mamba-org/mamba) dependencies are listed in `environment.yml`. This repository contains all scripts and configs that are needed to replicate the results in the paper. A more extensive usage guide will be provided later.


## BibTeX entry

The paper is to appear in Findings of ACL2021. The preprint can be cited as:

```bibtex
@misc{devries2021adapting,
      title={{Adapting Monolingual Models: Data can be Scarce when Language Similarity is High}}, 
      author={Wietse de Vries and Martijn Bartelds and Malvina Nissim and Martijn Wieling},
      year={2021},
      eprint={2105.02855},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
