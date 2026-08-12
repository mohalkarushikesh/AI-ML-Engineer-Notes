**Why CNNs suit images:** three things —

- **Parameter sharing** — the same filter slides across the whole image, so far fewer weights than a fully-connected net.
- **Local receptive fields** — each filter looks at small neighboring regions, capturing spatial structure **(edges → shapes → objects).**
- **Pooling** — downsamples for translation invariance (a cat is a cat wherever it appears).

Interview one-liner: *"CNNs exploit image structure — shared filters cut parameters, local receptive fields capture spatial patterns, and pooling adds translation invariance."*
