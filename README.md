Wietse de Vries • Martijn Bartelds • Malvina Nissim • Martijn Wieling

# Adapting Monolingual Models: Data can be Scarce when Language Similarity is High

This repository contains everything that is needed to replicate the results in the paper:

📝 [Adapting Monolingual Models: Data can be Scarce when Language Similarity is High](https://aclanthology.org/2021.findings-acl.433/) [Findings of ACL 2021]

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

```bibtex
@inproceedings{de-vries-etal-2021-adapting,
    title = "Adapting Monolingual Models: Data can be Scarce when Language Similarity is High",
    author = "de Vries, Wietse  and
      Bartelds, Martijn  and
      Nissim, Malvina  and
      Wieling, Martijn",
    booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-acl.433",
    doi = "10.18653/v1/2021.findings-acl.433",
    pages = "4901--4907",
}
```
