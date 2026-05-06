---
title: "Hello, world"
description: "A first post — to show how Markdown, images, and LaTeX render."
date: 2026-05-05
tags: [meta]
---

This is the first post on the blog. It exists mostly to show that Markdown,
images, and math all render the way they should.

## Inline and display math

Inline math is written with single dollar signs, like the softmax
$\sigma(z)_i = e^{z_i} / \sum_j e^{z_j}$ that we all know and love.

Display math uses double dollar signs:

$$
\mathcal{L}(\theta) \;=\; -\,\mathbb{E}_{(x,y)\sim\mathcal{D}}\big[\log p_\theta(y \mid x)\big].
$$

You can also use environments:

$$
\nabla_\theta \mathcal{L}(\theta) \;=\; -\,\mathbb{E}\!\left[\nabla_\theta \log p_\theta(y \mid x)\right].
$$

## Images

Drop an image next to this markdown file and reference it with a relative path:

```markdown
![alt text](./figure.png)
```

## Code

```python
import torch
import torch.nn.functional as F

def cross_entropy(logits, targets):
    return F.cross_entropy(logits, targets)
```

## Lists, quotes, and the rest

> "All models are wrong, but some are useful." — George Box

- Markdown works as expected.
- LaTeX is rendered at build time via `remark-math` + `rehype-katex`.
- Code blocks are highlighted by Shiki.
