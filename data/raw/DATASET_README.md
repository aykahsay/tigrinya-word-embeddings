# Gamayun Language Data Kits — Tigrinya (5k)

This package contains the **Tigrinya–English** portion of CLEAR Global's [Gamayun Language Data Kits](https://huggingface.co/datasets/CLEAR-Global/Gamayun-kits): 5,000 parallel sentences translated by professional translators from CLEAR Global's (formerly Translators without Borders) translator community.

## About Gamayun kits

CLEAR Global's Gamayun kits are a starting point for developing audio and text corpora for languages without pre-existing data resources. Parallel data for a language is created by translating a pre-compiled set of general-domain English (or French/Spanish, for some kits) sentences sourced from the [Tatoeba](https://tatoeba.org) repository. The selection algorithm ensures representation of the most frequently used words in the language; for more information, see the [corepus-gen repository](https://github.com/translatorswb/corepus-gen).

To scale corpus production, the initiative offers four dataset sizes:

- Mini-kit of 5,000 sentences (`kit5k`)
- Small-kit of 10,000 sentences (`kit10k`)
- Medium-kit of 15,000 sentences (`kit15k`)
- Large-kit of 30,000 sentences (`kit30k`)

Kit sizes for a given language are **independent selections** — `kit10k` is not a superset of `kit5k`.

## Contents of this package

This package contains only the **Tigrinya mini-kit** (`kit5k`):

- `kit5k.tsv` — 5,000 parallel English–Tigrinya sentences, tab-separated, with a header row.

  | Column | Description                       |
  |--------|-----------------------------------|
  | `eng`  | English source sentence (Tatoeba) |
  | `tig`  | Tigrinya translation              |

- `core_kit5k.tsv` — the English source kit (`core-v1-en/kit5k`) the translations were produced from.

  | Column | Description                                 |
  |--------|---------------------------------------------|
  | `id`   | Tatoeba sentence ID for the source sentence |
  | `text` | English source sentence                     |

Rows are aligned line-for-line across both files.

## Languages

- English (`en`) — source
- Tigrinya (`ti`) — target

## License

This dataset is released under the **Creative Commons Attribution 4.0 International License (CC BY 4.0)**.

You are free to:

- **Share** — copy and redistribute the material in any medium or format
- **Adapt** — remix, transform, and build upon the material for any purpose, even commercially

Under the following terms:

- **Attribution** — You must give appropriate credit to CLEAR Global, provide a link to the license, and indicate if changes were made.

Full license text: <https://creativecommons.org/licenses/by/4.0/legalcode>

## Citation

If you use this data, please cite:

```
Alp Öktem, Muhannad Albayk Jaam, Eric DeLuca, Grace Tang
Gamayun – Language Technology for Humanitarian Response
In: 2020 IEEE Global Humanitarian Technology Conference (GHTC)
2020 October 29 - November 1; Virtual.
Link: https://ieeexplore.ieee.org/document/9342939
```

## Source

Originally published on the [CLEAR Global HuggingFace repository](https://huggingface.co/datasets/CLEAR-Global/Gamayun-kits).
