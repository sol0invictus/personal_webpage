---
title: "Score Centering: Stabilize RL Training"
description: "How Thinking Machines' hidden StableReinforce loss was reverse-engineered from its gradients, and a first-principles derivation of the method it turned out to be: score centering, from Marek & Ryabinin (2026)."
date: 2026-10-26
tags: [reinforcement-learning, policy-gradient, off-policy, importance-sampling, llm-post-training]
---

<style>
  article h4 { font-size: 1rem; }
  article h2.part { font-size: 1.55rem; margin-top: 2.4em; padding-top: 0.9em; border-top: 1px solid var(--rule); }
  article h2.part .part-num { display: block; font-size: 0.8rem; font-weight: 400; letter-spacing: 0.08em; text-transform: uppercase; color: var(--fg-muted); margin-bottom: 0.3em; }
  article figcaption p { margin: 0; }
  article pre { font-size: 0.74rem; }
  .tok { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.85em; background: var(--code-bg); padding: 0.1em 0.4em; border-radius: 3px; }
  .tok.picked { outline: 1px solid var(--accent); color: var(--accent); }
  figure.loop svg { display: block; width: 100%; height: auto; max-width: 640px; margin: 0 auto; }
  .loop .box { fill: var(--bg); stroke: var(--fg-muted); stroke-width: 1.2; }
  .loop .t1 { fill: var(--fg); font-weight: 600; font-size: 15px; font-family: ui-serif, Georgia, serif; }
  .loop .t2, .loop .alabel { fill: var(--fg-muted); font-size: 13px; font-family: ui-serif, Georgia, serif; }
  .loop .arrow { stroke: var(--fg-muted); stroke-width: 1.5; fill: none; }
  .loop .head { fill: var(--fg-muted); }

  /* Playground */
  .play { --toward-a: #2a78d6; --toward-b: #e34948; border-top: 1px solid var(--rule); border-bottom: 1px solid var(--rule); padding: 1em 0 1.2em; margin: 1.6em 0; }
  .controls { display: flex; flex-wrap: wrap; gap: 0.8em 2em; align-items: flex-end; }
  .control { display: flex; flex-direction: column; gap: 0.2em; }
  .control label, .control .clabel { font-size: 0.88rem; color: var(--fg-muted); }
  .control output { color: var(--fg); font-variant-numeric: tabular-nums; }
  .control input[type=range] { width: 170px; accent-color: var(--accent); }
  .seg { display: inline-flex; border: 1px solid var(--rule); border-radius: 4px; overflow: hidden; }
  .seg button, .presets button { font: inherit; font-size: 0.85rem; background: transparent; color: var(--fg); cursor: pointer; }
  .seg button { border: 0; border-right: 1px solid var(--rule); padding: 0.25em 0.7em; }
  .seg button:last-child { border-right: 0; }
  .seg button[aria-pressed=true] { background: var(--code-bg); color: var(--accent); }
  .presets { display: flex; gap: 0.5em; flex-wrap: wrap; margin-top: 0.8em; font-size: 0.88rem; color: var(--fg-muted); align-items: baseline; }
  .presets button { border: 1px solid var(--rule); border-radius: 4px; padding: 0.15em 0.6em; }
  .presets button:hover { border-color: var(--accent); color: var(--accent); }
  .derived { font-size: 0.85rem; color: var(--fg-muted); margin: 0.6em 0 0; font-variant-numeric: tabular-nums; }
  .chart { margin-top: 0.8em; }
  .axis-head, .row { display: grid; grid-template-columns: 150px 1fr 70px 70px; gap: 10px; align-items: center; }
  .axis-head { font-size: 0.8rem; color: var(--fg-muted); padding-bottom: 0.3em; border-bottom: 2px solid var(--rule); }
  .axis-head .dir { display: flex; justify-content: space-between; }
  .axis-head .r { text-align: right; }
  .row { padding: 4px 0; border-bottom: 1px solid var(--rule); font-size: 0.9rem; }
  .row:hover { background: var(--code-bg); }
  .row .name small { display: block; color: var(--fg-muted); font-size: 0.78rem; line-height: 1.2; }
  .row.target .name { font-weight: 600; }
  .row .val, .row .gap { text-align: right; font-variant-numeric: tabular-nums; }
  .row .gap { color: var(--fg-muted); }
  .track { position: relative; height: 26px; }
  .track .zero { position: absolute; left: 50%; top: 0; bottom: 0; width: 1px; background: var(--fg-muted); opacity: 0.5; }
  .track .bar { position: absolute; top: 6px; height: 14px; }
  .track .bar.pos { background: var(--toward-a); border-radius: 0 4px 4px 0; }
  .track .bar.neg { background: var(--toward-b); border-radius: 4px 0 0 4px; }
  .track .tgt { position: absolute; top: 1px; bottom: 1px; width: 0; border-left: 2px dashed var(--fg); opacity: 0.7; }
  .legend { display: flex; gap: 1.2em; flex-wrap: wrap; font-size: 0.82rem; color: var(--fg-muted); margin-top: 0.6em; }
  .legend .sw { display: inline-block; width: 11px; height: 11px; border-radius: 2px; vertical-align: -1px; margin-right: 5px; }
  .legend .sw.dash { width: 0; height: 13px; border-left: 2px dashed var(--fg); border-radius: 0; margin-right: 7px; }
  .tooltip { position: fixed; pointer-events: none; z-index: 10; max-width: 300px; background: var(--bg); color: var(--fg);
             border: 1px solid var(--rule); border-radius: 4px; padding: 0.5em 0.8em; font-size: 0.82rem; line-height: 1.45;
             box-shadow: 0 3px 14px rgba(0,0,0,0.1); opacity: 0; transition: opacity 0.08s; }
  @media (max-width: 540px) { .axis-head, .row { grid-template-columns: 96px 1fr 56px 56px; gap: 6px; } }
</style>

*How Thinking Machines' hidden StableReinforce loss was reverse-engineered from its gradients, and a derivation from first principles of the method it turned out to be: score centering, from [Marek & Ryabinin (2026)](https://arxiv.org/abs/2609.20807).*

**References**

- Martin Marek and Max Ryabinin, [*Score Centering Stabilizes Off-Policy Reinforcement Learning*](https://arxiv.org/abs/2609.20807), arXiv:2609.20807, 2026.
- Martin Marek, [“Tinker's new StableReinforce objective is Reinforce + Score Centering. Here's how I reverse-engineered it”](https://x.com/mrtnm/status/2114145574987702525), thread on X, October 24, 2026.

<h2 class="part"><span class="part-num">Part 1</span>The reverse-engineering trick</h2>

### A hidden loss function

On October 21, Thinking Machines added a new loss to Tinker, their post-training API, called [StableReinforce](https://tinker-docs.thinkingmachines.ai/tinker/losses/stable-reinforce/). The pitch is that RL runs normally collapse when the policy that generates the data drifts away from the policy being trained, through async staleness or mismatched numerics, and that this loss keeps them stable. The docs say it “has led to extremely stable training, even under highly off-policy async RL and low precision numerics.” They do not say what the loss is.

Three days later, Martin Marek, first author of the [score-centering paper](https://arxiv.org/abs/2609.20807), reverse-engineered the loss from the outside and posted the result in [a thread on X](https://x.com/mrtnm/status/2114145574987702525). StableReinforce turns out to be REINFORCE plus score centering, the correction his paper introduces. The rest of Part 1 follows the thread: two tricks that recover the raw gradient of a hidden loss, and the clue they turned up, a single-token update that moves only eight logits. Part 2 goes back to the paper, builds score centering from scratch following its Section 4, and ends by showing why that clue gives the objective away.

### Reverse-engineering a loss from its gradients

Tinker lets you send data, take an optimiser step and download the updated weights, but you never see the loss. Every loss leaves a fingerprint, though: the gradient it produces. If you can read off the raw gradient, you can write down a candidate loss, compute its gradient, and compare. That needs two things: the raw gradient from a weight update, and a place in the network where the gradient is easy to interpret.

#### Trick 1: make Adam behave like SGD

With plain SGD, the weight change is the gradient times a constant: $\Delta w = -\eta\, g$. Tinker doesn't offer SGD, only Adam, and Adam hides the gradient. On the first step its bias-corrected moments are exactly $\hat m = g$ and $\hat v = g^2$, so the update is:

$$
\Delta w \;=\; -\eta\, \frac{\hat m}{\sqrt{\hat v} + \epsilon} \;=\; -\eta\, \frac{g}{|g| + \epsilon}.
$$

With the usual tiny $\epsilon$ (around $10^{-8}$) this is $-\eta\,\operatorname{sign}(g)$. Every weight moves by the same amount and only the sign of the gradient survives, so the update tells you almost nothing about the loss.

The trick is to set $\epsilon = 10{,}000$. Real gradients are tiny next to that, so $|g| + \epsilon \approx \epsilon$ and:

$$
\Delta w \;\approx\; -\frac{\eta}{\epsilon}\, g
\qquad\Longrightarrow\qquad
g \;\approx\; -\frac{\epsilon}{\eta}\,\Delta w .
$$

Adam has been turned into SGD with learning rate $\eta/\epsilon$. Download the weights before and after, subtract, and rescale: that is the raw gradient. The relative error is about $|g|/\epsilon$, which is negligible. You also need weight decay off, so that nothing else moves the weights.

<figure>
<svg viewBox="0 0 760 240" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Gradient recovered from Adam updates with default and huge epsilon" font-family="ui-serif, Georgia, 'Times New Roman', serif" style="display:block;width:100%;height:auto">
<defs><marker id="arr-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#666666"/></marker><marker id="arra-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#0b5fff"/></marker></defs>
<text x="34" y="24" fill="#1a1a1a" font-size="13.5" font-weight="600">True gradient g</text><line x1="34" y1="166.69230769230768" x2="256" y2="166.69230769230768" stroke="#e5e5e5" stroke-width="1"/><text x="28" y="170.69230769230768" fill="#666666" font-size="11.5" text-anchor="end">−1</text><line x1="34" y1="109" x2="256" y2="109" stroke="#666666" stroke-width="1"/><text x="28" y="113" fill="#666666" font-size="11.5" text-anchor="end">0</text><line x1="34" y1="51.30769230769231" x2="256" y2="51.30769230769231" stroke="#e5e5e5" stroke-width="1"/><text x="28" y="55.30769230769231" fill="#666666" font-size="11.5" text-anchor="end">1</text><path d="M35.875,109 L35.875,66.84615384615384 Q35.875,62.84615384615384 39.875,62.84615384615384 L55.875,62.84615384615384 Q59.875,62.84615384615384 59.875,66.84615384615384 L59.875,109 Z" fill="#2a78d6"><title>w1: g = 0.80</title></path><text x="47.875" y="200" fill="#666666" font-size="11.5" text-anchor="middle">w1</text><path d="M63.625,109 L63.625,122.3076923076923 Q63.625,126.3076923076923 67.625,126.3076923076923 L83.625,126.3076923076923 Q87.625,126.3076923076923 87.625,122.3076923076923 L87.625,109 Z" fill="#e34948"><title>w2: g = -0.30</title></path><text x="75.625" y="200" fill="#666666" font-size="11.5" text-anchor="middle">w2</text><path d="M91.375,109 L91.375,107.5576923076923 Q91.375,106.11538461538461 92.8173076923077,106.11538461538461 L113.9326923076923,106.11538461538461 Q115.375,106.11538461538461 115.375,107.5576923076923 L115.375,109 Z" fill="#2a78d6"><title>w3: g = 0.05</title></path><text x="103.375" y="200" fill="#666666" font-size="11.5" text-anchor="middle">w3</text><path d="M119.125,109 L119.125,174.23076923076923 Q119.125,178.23076923076923 123.125,178.23076923076923 L139.125,178.23076923076923 Q143.125,178.23076923076923 143.125,174.23076923076923 L143.125,109 Z" fill="#e34948"><title>w4: g = -1.20</title></path><text x="131.125" y="200" fill="#666666" font-size="11.5" text-anchor="middle">w4</text><path d="M146.875,109 L146.875,84.15384615384616 Q146.875,80.15384615384616 150.875,80.15384615384616 L166.875,80.15384615384616 Q170.875,80.15384615384616 170.875,84.15384615384616 L170.875,109 Z" fill="#2a78d6"><title>w5: g = 0.50</title></path><text x="158.875" y="200" fill="#666666" font-size="11.5" text-anchor="middle">w5</text><path d="M174.625,109 L174.625,111.88461538461539 Q174.625,114.76923076923079 177.5096153846154,114.76923076923079 L195.7403846153846,114.76923076923079 Q198.625,114.76923076923079 198.625,111.88461538461539 L198.625,109 Z" fill="#e34948"><title>w6: g = -0.10</title></path><text x="186.625" y="200" fill="#666666" font-size="11.5" text-anchor="middle">w6</text><path d="M202.375,109 L202.375,61.07692307692308 Q202.375,57.07692307692308 206.375,57.07692307692308 L222.375,57.07692307692308 Q226.375,57.07692307692308 226.375,61.07692307692308 L226.375,109 Z" fill="#2a78d6"><title>w7: g = 0.90</title></path><text x="214.375" y="200" fill="#666666" font-size="11.5" text-anchor="middle">w7</text><path d="M230.125,109 L230.125,139.6153846153846 Q230.125,143.6153846153846 234.125,143.6153846153846 L250.125,143.6153846153846 Q254.125,143.6153846153846 254.125,139.6153846153846 L254.125,109 Z" fill="#e34948"><title>w8: g = -0.60</title></path><text x="242.125" y="200" fill="#666666" font-size="11.5" text-anchor="middle">w8</text><text x="286" y="24" fill="#1a1a1a" font-size="13.5" font-weight="600">Default Adam (ε = 1e-8)</text><line x1="286" y1="166.69230769230768" x2="508" y2="166.69230769230768" stroke="#e5e5e5" stroke-width="1"/><line x1="286" y1="109" x2="508" y2="109" stroke="#666666" stroke-width="1"/><line x1="286" y1="51.30769230769231" x2="508" y2="51.30769230769231" stroke="#e5e5e5" stroke-width="1"/><path d="M287.875,109 L287.875,55.30769302884615 Q287.875,51.30769302884615 291.875,51.30769302884615 L307.875,51.30769302884615 Q311.875,51.30769302884615 311.875,55.30769302884615 L311.875,109 Z" fill="#2a78d6"><title>w1: recovered 1.00 (only the sign)</title></path><text x="299.875" y="200" fill="#666666" font-size="11.5" text-anchor="middle">w1</text><path d="M315.625,109 L315.625,162.6923057692308 Q315.625,166.6923057692308 319.625,166.6923057692308 L335.625,166.6923057692308 Q339.625,166.6923057692308 339.625,162.6923057692308 L339.625,109 Z" fill="#e34948"><title>w2: recovered -1.00 (only the sign)</title></path><text x="327.625" y="200" fill="#666666" font-size="11.5" text-anchor="middle">w2</text><path d="M343.375,109 L343.375,55.30770384615154 Q343.375,51.30770384615154 347.375,51.30770384615154 L363.375,51.30770384615154 Q367.375,51.30770384615154 367.375,55.30770384615154 L367.375,109 Z" fill="#2a78d6"><title>w3: recovered 1.00 (only the sign)</title></path><text x="355.375" y="200" fill="#666666" font-size="11.5" text-anchor="middle">w3</text><path d="M371.125,109 L371.125,162.69230721153846 Q371.125,166.69230721153846 375.125,166.69230721153846 L391.125,166.69230721153846 Q395.125,166.69230721153846 395.125,162.69230721153846 L395.125,109 Z" fill="#e34948"><title>w4: recovered -1.00 (only the sign)</title></path><text x="383.125" y="200" fill="#666666" font-size="11.5" text-anchor="middle">w4</text><path d="M398.875,109 L398.875,55.30769346153845 Q398.875,51.30769346153845 402.875,51.30769346153845 L418.875,51.30769346153845 Q422.875,51.30769346153845 422.875,55.30769346153845 L422.875,109 Z" fill="#2a78d6"><title>w5: recovered 1.00 (only the sign)</title></path><text x="410.875" y="200" fill="#666666" font-size="11.5" text-anchor="middle">w5</text><path d="M426.625,109 L426.625,162.6923019230775 Q426.625,166.6923019230775 430.625,166.6923019230775 L446.625,166.6923019230775 Q450.625,166.6923019230775 450.625,162.6923019230775 L450.625,109 Z" fill="#e34948"><title>w6: recovered -1.00 (only the sign)</title></path><text x="438.625" y="200" fill="#666666" font-size="11.5" text-anchor="middle">w6</text><path d="M454.375,109 L454.375,55.30769294871794 Q454.375,51.30769294871794 458.375,51.30769294871794 L474.375,51.30769294871794 Q478.375,51.30769294871794 478.375,55.30769294871794 L478.375,109 Z" fill="#2a78d6"><title>w7: recovered 1.00 (only the sign)</title></path><text x="466.375" y="200" fill="#666666" font-size="11.5" text-anchor="middle">w7</text><path d="M482.125,109 L482.125,162.69230673076925 Q482.125,166.69230673076925 486.125,166.69230673076925 L502.125,166.69230673076925 Q506.125,166.69230673076925 506.125,162.69230673076925 L506.125,109 Z" fill="#e34948"><title>w8: recovered -1.00 (only the sign)</title></path><text x="494.125" y="200" fill="#666666" font-size="11.5" text-anchor="middle">w8</text><text x="538" y="24" fill="#1a1a1a" font-size="13.5" font-weight="600">Adam with ε = 10,000</text><line x1="538" y1="166.69230769230768" x2="760" y2="166.69230769230768" stroke="#e5e5e5" stroke-width="1"/><line x1="538" y1="109" x2="760" y2="109" stroke="#666666" stroke-width="1"/><line x1="538" y1="51.30769230769231" x2="760" y2="51.30769230769231" stroke="#e5e5e5" stroke-width="1"/><path d="M539.875,109 L539.875,66.84984585848517 Q539.875,62.84984585848517 543.875,62.84984585848517 L559.875,62.84984585848517 Q563.875,62.84984585848517 563.875,66.84984585848517 L563.875,109 Z" fill="#2a78d6"><title>w1: recovered 0.80 (true 0.80)</title></path><text x="551.875" y="200" fill="#666666" font-size="11.5" text-anchor="middle">w1</text><path d="M567.625,109 L567.625,122.30717309249954 Q567.625,126.30717309249954 571.625,126.30717309249954 L587.625,126.30717309249954 Q591.625,126.30717309249954 591.625,122.30717309249954 L591.625,109 Z" fill="#e34948"><title>w2: recovered -0.30 (true -0.30)</title></path><text x="579.625" y="200" fill="#666666" font-size="11.5" text-anchor="middle">w2</text><path d="M595.375,109 L595.375,107.55769951919471 Q595.375,106.11539903838943 596.8173004808053,106.11539903838943 L617.9326995191947,106.11539903838943 Q619.375,106.11539903838943 619.375,107.55769951919471 L619.375,109 Z" fill="#2a78d6"><title>w3: recovered 0.05 (true 0.05)</title></path><text x="607.375" y="200" fill="#666666" font-size="11.5" text-anchor="middle">w3</text><path d="M623.125,109 L623.125,174.22246253526498 Q623.125,178.22246253526498 627.125,178.22246253526498 L643.125,178.22246253526498 Q647.125,178.22246253526498 647.125,174.22246253526498 L647.125,109 Z" fill="#e34948"><title>w4: recovered -1.20 (true -1.20)</title></path><text x="635.125" y="200" fill="#666666" font-size="11.5" text-anchor="middle">w4</text><path d="M650.875,109 L650.875,84.15528838942669 Q650.875,80.15528838942669 654.875,80.15528838942669 L670.875,80.15528838942669 Q674.875,80.15528838942669 674.875,84.15528838942669 L674.875,109 Z" fill="#2a78d6"><title>w5: recovered 0.50 (true 0.50)</title></path><text x="662.875" y="200" fill="#666666" font-size="11.5" text-anchor="middle">w5</text><path d="M678.625,109 L678.625,111.88458653875 Q678.625,114.76917307749999 681.50958653875,114.76917307749999 L699.74041346125,114.76917307749999 Q702.625,114.76917307749999 702.625,111.88458653875 L702.625,109 Z" fill="#e34948"><title>w6: recovered -0.10 (true -0.10)</title></path><text x="690.625" y="200" fill="#666666" font-size="11.5" text-anchor="middle">w6</text><path d="M706.375,109 L706.375,61.081595733307076 Q706.375,57.081595733307076 710.375,57.081595733307076 L726.375,57.081595733307076 Q730.375,57.081595733307076 730.375,61.081595733307076 L730.375,109 Z" fill="#2a78d6"><title>w7: recovered 0.90 (true 0.90)</title></path><text x="718.375" y="200" fill="#666666" font-size="11.5" text-anchor="middle">w7</text><path d="M734.125,109 L734.125,139.6133078169156 Q734.125,143.6133078169156 738.125,143.6133078169156 L754.125,143.6133078169156 Q758.125,143.6133078169156 758.125,139.6133078169156 L758.125,109 Z" fill="#e34948"><title>w8: recovered -0.60 (true -0.60)</title></path><text x="746.125" y="200" fill="#666666" font-size="11.5" text-anchor="middle">w8</text><text x="397" y="228" fill="#666666" font-size="12" text-anchor="middle" font-style="italic">every weight moves the same amount</text><text x="649" y="228" fill="#666666" font-size="12" text-anchor="middle" font-style="italic">update proportional to g: rescale it</text><text x="145" y="228" fill="#666666" font-size="12" text-anchor="middle" font-style="italic">what we want to see</text>
</svg>
<figcaption>

**Figure 1.** Recovering a gradient from one Adam step, for eight weights with a made-up gradient. The panels show what you can infer as $-\Delta w/\eta$, rescaled. With the default $\epsilon$, every weight moves by the same amount and only the sign survives. With $\epsilon = 10{,}000$, the update is proportional to $g$.

</figcaption>
</figure>

#### Trick 2: a rank-1 LoRA exposes the logit gradient

Even with the raw gradient in hand, billions of numbers spread across every layer are hard to interpret. The cleanest place to look is the logits, because that is where a loss function acts directly. The idea is to train a **rank-1 LoRA** adapter on the unembedding, the matrix $W$ that turns the final hidden state $h$ into logits. A rank-1 adapter adds $B A h$, with $A$ a $1 \times d$ row and $B$ a $V \times 1$ column:

$$
z \;=\; W h \;+\; B\,\underbrace{(A h)}_{a\ \in\ \mathbb{R}} .
$$

The term $a = A h$ is a single number. Write $g_z = \partial L / \partial z$ for the gradient with respect to the $V$ logits. By the chain rule:

$$
\frac{\partial L}{\partial B} \;=\; g_z \cdot a .
$$

The gradient on $B$ is the logit gradient multiplied by one scalar. For a single-token update you can read the whole logit gradient straight off $B$, up to that scale factor. Which logits are non-zero, and their ratios, are exact. Combined with trick 1, a downloaded weight update now tells you exactly how the hidden loss pushes on every one of the vocabulary's logits.

<figure>
<svg viewBox="0 0 780 280" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Rank-1 LoRA on the unembedding" font-family="ui-serif, Georgia, 'Times New Roman', serif" style="display:block;width:100%;height:auto">
<defs><marker id="arr-3" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#666666"/></marker><marker id="arra-3" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#0b5fff"/></marker></defs>
<rect x="10" y="92" width="110" height="56" rx="4" fill="#fdfdfd" stroke="#666666" stroke-width="1.2"/><text x="65" y="117" fill="#1a1a1a" font-size="14" text-anchor="middle" font-weight="600">h</text><text x="65" y="135" fill="#666666" font-size="12" text-anchor="middle">hidden state, d</text><rect x="190" y="26" width="150" height="56" rx="4" fill="#fdfdfd" stroke="#666666" stroke-width="1.2"/><text x="265" y="51" fill="#1a1a1a" font-size="14" text-anchor="middle" font-weight="600">W</text><text x="265" y="69" fill="#666666" font-size="12" text-anchor="middle">V × d, frozen</text><rect x="190" y="158" width="150" height="56" rx="4" fill="#fdfdfd" stroke="#666666" stroke-width="1.2"/><text x="265" y="183" fill="#1a1a1a" font-size="14" text-anchor="middle" font-weight="600">A</text><text x="265" y="201" fill="#666666" font-size="12" text-anchor="middle">1 × d, trained</text><circle cx="412" cy="186" r="26" fill="#fdfdfd" stroke="#1a1a1a" stroke-width="1.4"/><text x="412" y="191" fill="#1a1a1a" font-size="15" text-anchor="middle" font-weight="600" font-style="italic">a</text><text x="412" y="234" fill="#666666" font-size="12" text-anchor="middle">a = A h: one number</text><rect x="480" y="158" width="130" height="56" rx="4" fill="#fdfdfd" stroke="#0b5fff" stroke-width="1.2"/><text x="545" y="183" fill="#1a1a1a" font-size="14" text-anchor="middle" font-weight="600">B</text><text x="545" y="201" fill="#666666" font-size="12" text-anchor="middle">V × 1, trained</text><circle cx="545" cy="54" r="13" fill="#fdfdfd" stroke="#666666" stroke-width="1.2"/><text x="545" y="59" fill="#1a1a1a" font-size="16" text-anchor="middle">+</text><rect x="640" y="26" width="125" height="56" rx="4" fill="#fdfdfd" stroke="#666666" stroke-width="1.2"/><text x="702.5" y="51" fill="#1a1a1a" font-size="14" text-anchor="middle" font-weight="600">z</text><text x="702.5" y="69" fill="#666666" font-size="12" text-anchor="middle">logits, V</text><path d="M120,110 H155 V54 H187" fill="none" stroke="#666666" stroke-width="1.4" marker-end="url(#arr-3)"/><path d="M120,130 H155 V186 H187" fill="none" stroke="#666666" stroke-width="1.4" marker-end="url(#arr-3)"/><path d="M340,54 H529" fill="none" stroke="#666666" stroke-width="1.4" marker-end="url(#arr-3)"/><path d="M340,186 H383" fill="none" stroke="#666666" stroke-width="1.4" marker-end="url(#arr-3)"/><path d="M438,186 H477" fill="none" stroke="#666666" stroke-width="1.4" marker-end="url(#arr-3)"/><path d="M545,158 V70" fill="none" stroke="#666666" stroke-width="1.4" marker-end="url(#arr-3)"/><path d="M558,54 H637" fill="none" stroke="#666666" stroke-width="1.4" marker-end="url(#arr-3)"/><path d="M702,82 V250 H545 V217" fill="none" stroke="#0b5fff" stroke-width="1.4" stroke-dasharray="5 4" marker-end="url(#arra-3)"/><text x="712" y="150" fill="#0b5fff" font-size="13">∂L/∂z</text><text x="712" y="166" fill="#0b5fff" font-size="12">(V numbers)</text><text x="624" y="270" fill="#0b5fff" font-size="13.5" text-anchor="middle" font-weight="600">∂L/∂B = a · ∂L/∂z</text>
</svg>
<figcaption>

**Figure 2.** A rank-1 LoRA on the unembedding. The adapter's path squeezes through a single number $a$, so the gradient reaching $B$ (dashed) is the full logit gradient $\partial L/\partial z$ times one scalar.

</figcaption>
</figure>

#### What the gradient showed

For a standard REINFORCE loss, every logit gets a non-zero gradient. (We derive why below: the softmax couples them all.) In StableReinforce's single-token update, **only eight logits** had a non-zero gradient. Marek's hypothesis was REINFORCE plus score centering, with the centering term estimated from eight tokens sampled from the inference engine. He implemented that as a custom Tinker loss, took a step, and found the recovered gradients matched StableReinforce's “within numeric precision across all layers.”

<figure>
<svg viewBox="0 0 760 378" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Dense versus sparse logit gradient" font-family="ui-serif, Georgia, 'Times New Roman', serif" style="display:block;width:100%;height:auto">
<defs><marker id="arr-4" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#666666"/></marker><marker id="arra-4" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#0b5fff"/></marker></defs>
<text x="40" y="24" fill="#1a1a1a" font-size="13.5" font-weight="600">Standard REINFORCE: eᵧ − p</text><text x="750" y="24" fill="#666666" font-size="12.5" text-anchor="end">40 of 40 logits non-zero</text><line x1="40" y1="126.30769230769229" x2="750" y2="126.30769230769229" stroke="#666666" stroke-width="1"/><text x="34" y="130.3076923076923" fill="#666666" font-size="11.5" text-anchor="end">0</text><line x1="40" y1="80.15384615384615" x2="750" y2="80.15384615384615" stroke="#e5e5e5" stroke-width="1"/><text x="34" y="84.15384615384615" fill="#666666" font-size="11.5" text-anchor="end">0.5</text><line x1="40" y1="34" x2="750" y2="34" stroke="#e5e5e5" stroke-width="1"/><text x="34" y="38" fill="#666666" font-size="11.5" text-anchor="end">1</text><path d="M41,126.30769230769229 L41,126.59979535306742 Q41,126.89189839844255 41.29210304537513,126.89189839844255 L56.45789695462487,126.89189839844255 Q56.75,126.89189839844255 56.75,126.59979535306742 L56.75,126.30769230769229 Z" fill="#e34948"><title>token 1: −0.006</title></path><path d="M58.75,126.30769230769229 L58.75,126.65812883154778 Q58.75,127.00856535540328 59.10043652385549,127.00856535540328 L74.14956347614451,127.00856535540328 Q74.5,127.00856535540328 74.5,126.65812883154778 L74.5,126.30769230769229 Z" fill="#e34948"><title>token 2: −0.008</title></path><path d="M94.25,126.30769230769229 L94.25,126.9311386605954 Q94.25,127.55458501349851 94.8734463529031,127.55458501349851 L109.3765536470969,127.55458501349851 Q110,127.55458501349851 110,126.9311386605954 L110,126.30769230769229 Z" fill="#e34948"><title>token 4: −0.014</title></path><path d="M112,126.30769230769229 L112,128.74905242339503 Q112,131.19041253909774 114.44136011570272,131.19041253909774 L125.30863988429728,131.19041253909774 Q127.75,131.19041253909774 127.75,128.74905242339503 L127.75,126.30769230769229 Z" fill="#e34948"><title>token 5: −0.053</title></path><path d="M129.75,126.30769230769229 L129.75,128.65318671697582 Q129.75,130.99868112625936 132.09549440928353,130.99868112625936 L143.15450559071647,130.99868112625936 Q145.5,130.99868112625936 145.5,128.65318671697582 L145.5,126.30769230769229 Z" fill="#e34948"><title>token 6: −0.051</title></path><path d="M147.5,126.30769230769229 L147.5,127.03554349529409 Q147.5,127.76339468289589 148.2278511876018,127.76339468289589 L162.5221488123982,127.76339468289589 Q163.25,127.76339468289589 163.25,127.03554349529409 L163.25,126.30769230769229 Z" fill="#e34948"><title>token 7: −0.016</title></path><path d="M165.25,126.30769230769229 L165.25,127.45888164238127 Q165.25,128.61007097707025 166.40118933468898,128.61007097707025 L179.84881066531102,128.61007097707025 Q181,128.61007097707025 181,127.45888164238127 L181,126.30769230769229 Z" fill="#e34948"><title>token 8: −0.025</title></path><path d="M183,126.30769230769229 L183,129.48567098034098 Q183,132.66364965298968 186.1779786726487,132.66364965298968 L195.5720213273513,132.66364965298968 Q198.75,132.66364965298968 198.75,129.48567098034098 L198.75,126.30769230769229 Z" fill="#e34948"><title>token 9: −0.069</title></path><path d="M200.75,126.30769230769229 L200.75,127.36492316641694 Q200.75,128.4221540251416 201.80723085872467,128.4221540251416 L215.44276914127533,128.4221540251416 Q216.5,128.4221540251416 216.5,127.36492316641694 L216.5,126.30769230769229 Z" fill="#e34948"><title>token 10: −0.023</title></path><path d="M236.25,126.30769230769229 L236.25,127.0485700405637 Q236.25,127.78944777343513 236.9908777328714,127.78944777343513 L251.2591222671286,127.78944777343513 Q252,127.78944777343513 252,127.0485700405637 L252,126.30769230769229 Z" fill="#e34948"><title>token 12: −0.016</title></path><path d="M254,126.30769230769229 L254,126.6625706081845 Q254,127.01744890867671 254.3548783004922,127.01744890867671 L269.3951216995078,127.01744890867671 Q269.75,127.01744890867671 269.75,126.6625706081845 L269.75,126.30769230769229 Z" fill="#e34948"><title>token 13: −0.008</title></path><path d="M307.25,126.30769230769229 L307.25,127.1341746308866 Q307.25,127.9606569540809 308.0764823231943,127.9606569540809 L322.1735176768057,127.9606569540809 Q323,127.9606569540809 323,127.1341746308866 L323,126.30769230769229 Z" fill="#e34948"><title>token 16: −0.018</title></path><path d="M325,126.30769230769229 L325,126.76363440494293 Q325,127.21957650219358 325.4559420972506,127.21957650219358 L340.2940579027494,127.21957650219358 Q340.75,127.21957650219358 340.75,126.76363440494293 L340.75,126.30769230769229 Z" fill="#e34948"><title>token 17: −0.010</title></path><path d="M360.5,126.30769230769229 L360.5,126.8592097006838 Q360.5,127.41072709367532 361.0515173929915,127.41072709367532 L375.6984826070085,127.41072709367532 Q376.25,127.41072709367532 376.25,126.8592097006838 L376.25,126.30769230769229 Z" fill="#e34948"><title>token 19: −0.012</title></path><path d="M378.25,126.30769230769229 L378.25,126.56080713375229 Q378.25,126.81392195981229 378.50311482606,126.81392195981229 L393.74688517394,126.81392195981229 Q394,126.81392195981229 394,126.56080713375229 L394,126.30769230769229 Z" fill="#e34948"><title>token 20: −0.005</title></path><path d="M396,126.30769230769229 L396,127.57905701611303 Q396,128.85042172453376 397.2713647084207,128.85042172453376 L410.4786352915793,128.85042172453376 Q411.75,128.85042172453376 411.75,127.57905701611303 L411.75,126.30769230769229 Z" fill="#e34948"><title>token 21: −0.028</title></path><path d="M413.75,126.30769230769229 L413.75,128.04477859371556 Q413.75,129.78186487973883 415.48708628602327,129.78186487973883 L427.76291371397673,129.78186487973883 Q429.5,129.78186487973883 429.5,128.04477859371556 L429.5,126.30769230769229 Z" fill="#e34948"><title>token 22: −0.038</title></path><path d="M449.25,126.30769230769229 L449.25,126.65262096685628 Q449.25,126.99754962602026 449.594928659164,126.99754962602026 L464.655071340836,126.99754962602026 Q465,126.99754962602026 465,126.65262096685628 L465,126.30769230769229 Z" fill="#e34948"><title>token 24: −0.007</title></path><path d="M467,126.30769230769229 L467,134.6020496971018 Q467,138.6020496971018 471,138.6020496971018 L478.75,138.6020496971018 Q482.75,138.6020496971018 482.75,134.6020496971018 L482.75,126.30769230769229 Z" fill="#e34948"><title>token 25: −0.133</title></path><path d="M484.75,126.30769230769229 L484.75,127.3332135428902 Q484.75,128.35873477808812 485.7755212351979,128.35873477808812 L499.4744787648021,128.35873477808812 Q500.5,128.35873477808812 500.5,127.3332135428902 L500.5,126.30769230769229 Z" fill="#e34948"><title>token 26: −0.022</title></path><path d="M502.5,126.30769230769229 L502.5,126.6159609189516 Q502.5,126.92422953021091 502.80826861125934,126.92422953021091 L517.9417313887407,126.92422953021091 Q518.25,126.92422953021091 518.25,126.6159609189516 L518.25,126.30769230769229 Z" fill="#e34948"><title>token 27: −0.007</title></path><path d="M520.25,126.30769230769229 L520.25,127.41352658627544 Q520.25,128.51936086485858 521.3558342785832,128.51936086485858 L534.8941657214168,128.51936086485858 Q536,128.51936086485858 536,127.41352658627544 L536,126.30769230769229 Z" fill="#e34948"><title>token 28: −0.024</title></path><path d="M538,126.30769230769229 L538,127.82297647348668 Q538,129.33826063928106 539.5152841657944,129.33826063928106 L552.2347158342056,129.33826063928106 Q553.75,129.33826063928106 553.75,127.82297647348668 L553.75,126.30769230769229 Z" fill="#e34948"><title>token 29: −0.033</title></path><path d="M555.75,126.30769230769229 L555.75,127.3197127534494 Q555.75,128.3317331992065 556.7620204457571,128.3317331992065 L570.4879795542429,128.3317331992065 Q571.5,128.3317331992065 571.5,127.3197127534494 L571.5,126.30769230769229 Z" fill="#e34948"><title>token 30: −0.022</title></path><path d="M573.5,126.30769230769229 L573.5,127.68038434020794 Q573.5,129.05307637272358 574.8726920325156,129.05307637272358 L587.8773079674844,129.05307637272358 Q589.25,129.05307637272358 589.25,127.68038434020794 L589.25,126.30769230769229 Z" fill="#e34948"><title>token 31: −0.030</title></path><path d="M591.25,126.30769230769229 L591.25,127.00225342048375 Q591.25,127.69681453327522 591.9445611127915,127.69681453327522 L606.3054388872085,127.69681453327522 Q607,127.69681453327522 607,127.00225342048375 L607,126.30769230769229 Z" fill="#e34948"><title>token 32: −0.015</title></path><path d="M609,126.30769230769229 L609,127.18662207816399 Q609,128.0655518486357 609.8789297704717,128.0655518486357 L623.8710702295283,128.0655518486357 Q624.75,128.0655518486357 624.75,127.18662207816399 L624.75,126.30769230769229 Z" fill="#e34948"><title>token 33: −0.019</title></path><path d="M644.5,126.30769230769229 L644.5,126.69143506541437 Q644.5,127.07517782313643 644.8837427577221,127.07517782313643 L659.8662572422779,127.07517782313643 Q660.25,127.07517782313643 660.25,126.69143506541437 L660.25,126.30769230769229 Z" fill="#e34948"><title>token 35: −0.008</title></path><path d="M662.25,126.30769230769229 L662.25,126.67439557463939 Q662.25,127.04109884158648 662.6167032669471,127.04109884158648 L677.6332967330529,127.04109884158648 Q678,127.04109884158648 678,126.67439557463939 L678,126.30769230769229 Z" fill="#e34948"><title>token 36: −0.008</title></path><path d="M680,126.30769230769229 L680,126.82803768366279 Q680,127.34838305963329 680.5203453759705,127.34838305963329 L695.2296546240295,127.34838305963329 Q695.75,127.34838305963329 695.75,126.82803768366279 L695.75,126.30769230769229 Z" fill="#e34948"><title>token 37: −0.011</title></path><path d="M697.75,126.30769230769229 L697.75,126.80022311831257 Q697.75,127.29275392893287 698.2425308106203,127.29275392893287 L713.0074691893797,127.29275392893287 Q713.5,127.29275392893287 713.5,126.80022311831257 L713.5,126.30769230769229 Z" fill="#e34948"><title>token 38: −0.011</title></path><path d="M715.5,126.30769230769229 L715.5,59.630307199002885 Q715.5,55.630307199002885 719.5,55.630307199002885 L727.25,55.630307199002885 Q731.25,55.630307199002885 731.25,59.630307199002885 L731.25,126.30769230769229 Z" fill="#2a78d6"><title>token 39: +0.766</title></path><text x="709.5" y="94.96899975334759" fill="#1a1a1a" font-size="12" text-anchor="end">sampled token →</text><text x="40" y="204" fill="#1a1a1a" font-size="13.5" font-weight="600">Score centering with 8 sampled tokens: eᵧ − mean of 8 one-hots</text><text x="750" y="204" fill="#666666" font-size="12.5" text-anchor="end">8 of 40 logits non-zero</text><line x1="40" y1="306.30769230769226" x2="750" y2="306.30769230769226" stroke="#666666" stroke-width="1"/><text x="34" y="310.30769230769226" fill="#666666" font-size="11.5" text-anchor="end">0</text><line x1="40" y1="260.15384615384613" x2="750" y2="260.15384615384613" stroke="#e5e5e5" stroke-width="1"/><text x="34" y="264.15384615384613" fill="#666666" font-size="11.5" text-anchor="end">0.5</text><line x1="40" y1="214" x2="750" y2="214" stroke="#e5e5e5" stroke-width="1"/><text x="34" y="218" fill="#666666" font-size="11.5" text-anchor="end">1</text><path d="M41,306.30769230769226 L41,313.8461538461538 Q41,317.8461538461538 45,317.8461538461538 L52.75,317.8461538461538 Q56.75,317.8461538461538 56.75,313.8461538461538 L56.75,306.30769230769226 Z" fill="#e34948"><title>token 1: −0.125 (drawn 1×)</title></path><path d="M58.75,306.30769230769226 L58.75,313.8461538461538 Q58.75,317.8461538461538 62.75,317.8461538461538 L70.5,317.8461538461538 Q74.5,317.8461538461538 74.5,313.8461538461538 L74.5,306.30769230769226 Z" fill="#e34948"><title>token 2: −0.125 (drawn 1×)</title></path><path d="M218.5,306.30769230769226 L218.5,313.8461538461538 Q218.5,317.8461538461538 222.5,317.8461538461538 L230.25,317.8461538461538 Q234.25,317.8461538461538 234.25,313.8461538461538 L234.25,306.30769230769226 Z" fill="#e34948"><title>token 11: −0.125 (drawn 1×)</title></path><path d="M325,306.30769230769226 L325,313.8461538461538 Q325,317.8461538461538 329,317.8461538461538 L336.75,317.8461538461538 Q340.75,317.8461538461538 340.75,313.8461538461538 L340.75,306.30769230769226 Z" fill="#e34948"><title>token 17: −0.125 (drawn 1×)</title></path><path d="M467,306.30769230769226 L467,313.8461538461538 Q467,317.8461538461538 471,317.8461538461538 L478.75,317.8461538461538 Q482.75,317.8461538461538 482.75,313.8461538461538 L482.75,306.30769230769226 Z" fill="#e34948"><title>token 25: −0.125 (drawn 1×)</title></path><path d="M538,306.30769230769226 L538,313.8461538461538 Q538,317.8461538461538 542,317.8461538461538 L549.75,317.8461538461538 Q553.75,317.8461538461538 553.75,313.8461538461538 L553.75,306.30769230769226 Z" fill="#e34948"><title>token 29: −0.125 (drawn 1×)</title></path><path d="M573.5,306.30769230769226 L573.5,313.8461538461538 Q573.5,317.8461538461538 577.5,317.8461538461538 L585.25,317.8461538461538 Q589.25,317.8461538461538 589.25,313.8461538461538 L589.25,306.30769230769226 Z" fill="#e34948"><title>token 31: −0.125 (drawn 1×)</title></path><path d="M715.5,306.30769230769226 L715.5,229.53846153846155 Q715.5,225.53846153846155 719.5,225.53846153846155 L727.25,225.53846153846155 Q731.25,225.53846153846155 731.25,229.53846153846155 L731.25,306.30769230769226 Z" fill="#2a78d6"><title>token 39: +0.875 (drawn 1×)</title></path><text x="709.5" y="269.9230769230769" fill="#1a1a1a" font-size="12" text-anchor="end">sampled token →</text><text x="395" y="368" fill="#666666" font-size="12" text-anchor="middle">toy vocabulary of 40 tokens →</text>
</svg>
<figcaption>

**Figure 3.** The logit gradient for one token over a toy 40-token vocabulary, with advantage $+1$. Standard REINFORCE moves every logit, because the softmax couples them. Score centering with 8 sampled tokens cancels the shared $-p$ part, leaving only the sampled token and the correction tokens. Illustrative numbers, not Marek's measurements.

</figcaption>
</figure>

> Two caveats. This is a reverse-engineering result, not something Thinking Machines has stated, though a match across every layer is strong evidence. And the Tinker docs also list an optional sampler–learner KL term (default coefficient 0.01) that the thread doesn't discuss.

To see why “eight non-zero logits” gives the objective away, we need to understand what score centering is and what problem it solves. The rest of this post builds that from scratch.

<h2 class="part"><span class="part-num">Part 2</span>What score centering actually does</h2>

This part follows Section 4 of the paper. It builds score centering from first principles, compares it with importance sampling, and ends by coming back to the eight logits from Part 1.

### Two copies of the same model

RL post-training uses two copies of the model. The **sampler** $q$ is an inference engine (vLLM, SGLang) that generates rollouts. The **trainer** $p$ runs those finished rollouts through a training framework to compute gradients. Splitting the work is a large speed win, because generation goes one token at a time while scoring a finished sequence takes one parallel pass. (Appendix A goes through why the split exists.)

The two copies are meant to be identical, but in practice they aren't: lower-precision inference, different kernels, different batch shapes, or weights that are a few steps stale. This gap is the **training-inference mismatch**, and Section 4 of the paper is about exactly what it breaks.

Notation, at one token position with a fixed prefix:

| Symbol | Meaning |
|---|---|
| $p_v,\ q_v$ | probability of vocabulary token $v$ under the trainer and the sampler (one number each) |
| $p,\ q$ | the full next-token distributions at this position: vectors of all $V$ probabilities, $p = (p_1, \dots, p_V)$ |
| $s_v = \nabla_\theta \log p_v$ | the **score** of token $v$: the direction that makes it more likely |
| $y_t$ | the token the sampler actually picked |
| $R$ | the reward or advantage |
| $\bar s = \sum_v q_v\, s_v$ | the **average score** when tokens are drawn by the sampler |

The vanilla policy-gradient update for one token is $R\, s_{y_t}$: if the reward was good, make that token more likely.

### On-policy, the average score is zero

If the trainer drew the tokens itself, the average score would be:

$$
\begin{aligned}
\mathbb{E}_{p}[s] &= \sum_v p_v \,\nabla \log p_v
= \sum_v p_v \cdot \frac{\nabla p_v}{p_v}
= \sum_v \nabla p_v
= \nabla \Big(\sum_v p_v\Big)
= \nabla(1) = 0 .
\end{aligned}
$$

Probabilities sum to one, so raising one token's probability must lower the others, and on average the pushes cancel. A useful consequence: if every rollout gets the same reward, the expected update is $R \cdot 0 = 0$. A constant reward carries no information, and on-policy you correctly learn nothing from it.

### Under mismatch, the update splits into drift and signal

Now the tokens come from $q$ but the scores are computed through $p$. The cancellation above needed both to be the same distribution, so in general $\bar s = \sum_v q_v \nabla \log p_v \neq 0$. Using the covariance identity $\mathbb{E}[AB] = \mathbb{E}[A]\,\mathbb{E}[B] + \operatorname{Cov}(A,B)$ with $A = R$ and $B = s_{y_t}$, the expected update at a prefix splits into two parts (the paper's Eq. 3):

$$
\underbrace{\mathbb{E}_q[R\, s_{y_t}]}_{\text{update}}
\;=\;
\underbrace{\mathbb{E}_q[R]\;\bar s}_{\text{drift}}
\;+\;
\underbrace{\operatorname{Cov}_q(R,\, s_{y_t})}_{\text{signal}} .
\tag{3}
$$

The **signal** asks whether the tokens that led to high reward differ from the ones that led to low reward. That is the learning we want. The **drift** is the mean reward times the average score. It doesn't know which token earned which reward, so it is a pure artefact of the mismatch, and on-policy it vanishes.

### A two-token example

Take a model with a two-token vocabulary $\{A, B\}$ whose only parameters are the two logits $z_A, z_B$. This example is mine, not the paper's. The softmax gives $p_A = e^{z_A}/(e^{z_A} + e^{z_B})$, so $\log p_A = z_A - \log(e^{z_A} + e^{z_B})$. Differentiating with respect to each logit:

$$
\frac{\partial \log p_A}{\partial z_A} = 1 - p_A, \qquad
\frac{\partial \log p_A}{\partial z_B} = -p_B .
$$

The same pattern holds for any vocabulary size $V$. Differentiating $\log p_v$ with respect to logit $z_u$ gives $1 - p_u$ when $u = v$ and $-p_u$ otherwise. Written as a vector over all $V$ logits:

$$
s_v = \mathbf{e}_v - p ,
$$

where $\mathbf{e}_v$ is the **one-hot vector** for token $v$: length $V$, with a 1 in $v$'s slot and 0 everywhere else. In the two-token model, $\mathbf{e}_A = (1, 0)$ and $\mathbf{e}_B = (0, 1)$. So the score says: push the sampled token's logit up by 1, and push every logit down by its current probability. Note that this is **dense**. Even though only $A$ was sampled, every logit moves. That is the “standard REINFORCE puts a gradient on all logits” from the thread.

Let the trainer be $p = (0.5, 0.5)$ and the sampler, slightly off from quantisation, $q = (0.6, 0.4)$. Then $s_A = (0.5, -0.5)$ (raise $A$, lower $B$) and $s_B = (-0.5, 0.5)$. The average score depends on who draws the tokens:

| Tokens drawn by | Average score | Result |
|---|---|---:|
| trainer, $p = (0.5, 0.5)$ | $0.5\,(0.5,-0.5) + 0.5\,(-0.5,0.5)$ | $(0,\ 0)$ |
| sampler, $q = (0.6, 0.4)$ | $0.6\,(0.5,-0.5) + 0.4\,(-0.5,0.5)$ | $(0.1,\,-0.1)$ |

This matches the shortcut $\bar s = q - p$. With a constant reward of $+1$ there should be no update. Instead the logits move toward $A$. The pushes for $A$ and $B$ are the same size, because the trainer computes them believing the odds are 50/50. But the sampler decides how often each push happens, and it picks $A$ 60% of the time. With learning rate 1, the logits go to $(0.1, -0.1)$ and $p_A$ goes from 0.5 to about 0.55, halfway toward the sampler. The model learned nothing about the task. It just copied the sampler.

<figure>
<svg viewBox="0 0 760 188" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Average push on A's logit with trainer versus sampler drawing tokens" font-family="ui-serif, Georgia, 'Times New Roman', serif" style="display:block;width:100%;height:auto">
<defs><marker id="arr-5" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#666666"/></marker><marker id="arra-5" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#0b5fff"/></marker></defs>
<line x1="255.71428571428572" y1="22" x2="255.71428571428572" y2="160" stroke="#e5e5e5" /><text x="255.71428571428572" y="178" fill="#666666" font-size="11.5" text-anchor="middle">−0.3</text><line x1="327.1428571428571" y1="22" x2="327.1428571428571" y2="160" stroke="#e5e5e5" /><text x="327.1428571428571" y="178" fill="#666666" font-size="11.5" text-anchor="middle">−0.2</text><line x1="398.57142857142856" y1="22" x2="398.57142857142856" y2="160" stroke="#e5e5e5" /><text x="398.57142857142856" y="178" fill="#666666" font-size="11.5" text-anchor="middle">−0.1</text><line x1="470" y1="22" x2="470" y2="160" stroke="#666666" /><text x="470" y="178" fill="#666666" font-size="11.5" text-anchor="middle">0</text><line x1="541.4285714285713" y1="22" x2="541.4285714285713" y2="160" stroke="#e5e5e5" /><text x="541.4285714285713" y="178" fill="#666666" font-size="11.5" text-anchor="middle">+0.1</text><line x1="612.8571428571429" y1="22" x2="612.8571428571429" y2="160" stroke="#e5e5e5" /><text x="612.8571428571429" y="178" fill="#666666" font-size="11.5" text-anchor="middle">+0.2</text><line x1="684.2857142857142" y1="22" x2="684.2857142857142" y2="160" stroke="#e5e5e5" /><text x="684.2857142857142" y="178" fill="#666666" font-size="11.5" text-anchor="middle">+0.3</text><text x="220" y="14" fill="#666666" font-size="12">← pushes toward B</text><text x="720" y="14" fill="#666666" font-size="12" text-anchor="end">pushes toward A →</text><text x="10" y="50" fill="#1a1a1a" font-size="13.5" font-weight="600">Trainer draws tokens</text><text x="10" y="67" fill="#666666" font-size="12">A 50%, B 50%</text><path d="M470,38 L644.5714285714287,38 Q648.5714285714287,38 648.5714285714287,42 L648.5714285714287,50 Q648.5714285714287,54 644.5714285714287,54 L470,54 Z" fill="#2a78d6"><title>A's push: 0.5 × 0.5 = +0.25</title></path><path d="M470,38 L295.42857142857144,38 Q291.42857142857144,38 291.42857142857144,42 L291.42857142857144,50 Q291.42857142857144,54 295.42857142857144,54 L470,54 Z" fill="#e34948"><title>B's push: −0.5 × 0.5 = -0.25</title></path><text x="654.5714285714287" y="50" fill="#1a1a1a" font-size="12">+0.25</text><text x="285.42857142857144" y="50" fill="#1a1a1a" font-size="12" text-anchor="end">−0.25</text><path d="M470,60 l6,8 l-6,8 l-6,-8 z" fill="#1a1a1a"><title>net 0.00</title></path><text x="480" y="73" fill="#1a1a1a" font-size="12" font-weight="600">net 0</text><text x="10" y="114" fill="#1a1a1a" font-size="13.5" font-weight="600">Sampler draws tokens</text><text x="10" y="131" fill="#666666" font-size="12">A 60%, B 40%</text><path d="M470,102 L680.2857142857142,102 Q684.2857142857142,102 684.2857142857142,106 L684.2857142857142,114 Q684.2857142857142,118 680.2857142857142,118 L470,118 Z" fill="#2a78d6"><title>A's push: 0.5 × 0.6 = +0.30</title></path><path d="M470,102 L331.1428571428571,102 Q327.1428571428571,102 327.1428571428571,106 L327.1428571428571,114 Q327.1428571428571,118 331.1428571428571,118 L470,118 Z" fill="#e34948"><title>B's push: −0.5 × 0.4 = -0.20</title></path><text x="690.2857142857142" y="114" fill="#1a1a1a" font-size="12">+0.30</text><text x="321.1428571428571" y="114" fill="#1a1a1a" font-size="12" text-anchor="end">−0.20</text><path d="M541.4285714285713,124 l6,8 l-6,8 l-6,-8 z" fill="#1a1a1a"><title>net 0.10</title></path><text x="551.4285714285713" y="137" fill="#1a1a1a" font-size="12" font-weight="600">net +0.10</text>
</svg>
<figcaption>

**Figure 4.** The two-token example with a constant reward of $+1$. Each bar is one token's push on $A$'s logit (±0.5) times how often that token is drawn. When the trainer draws, the pushes cancel. When the sampler draws, $A$'s push wins, and the net update (◆) is the drift, $+0.10$.

</figcaption>
</figure>

### Drift is the trainer copying the sampler, and the sampler keeps moving

So what is the drift actually doing? It turns out to be something familiar. If you wanted to train $p$ to imitate $q$, you would use the cross-entropy loss $\mathrm{CE}(q,p) = -\sum_v q_v \log p_v$. Its gradient is $-\sum_v q_v \nabla \log p_v$, which is exactly $-\bar s$. In other words, $\bar s$ points the same way as an SFT step on the sampler's outputs. Under mismatch, every policy-gradient step quietly includes a little “make the trainer look more like the sampler,” scaled by the average reward. Nobody asked for it, and it has nothing to do with the task.

On its own that wouldn't be a disaster. Copying a fixed teacher is just distillation: you get a bit closer every step and eventually match it. The trouble is that the sampler isn't a fixed teacher. It's a slightly distorted copy of the trainer, and after every step it is refreshed with the trainer's new weights, distortion and all:

<figure class="loop">
<svg viewBox="0 0 640 200" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Feedback loop between trainer and sampler">
  <defs>
    <marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path class="head" d="M0,0 L10,5 L0,10 z"/>
    </marker>
  </defs>
  <rect class="box" x="10" y="60" width="150" height="84" rx="4"/>
  <text class="t1" x="85" y="96" text-anchor="middle">Trainer p</text>
  <text class="t2" x="85" y="118" text-anchor="middle">computes gradients</text>
  <rect class="box" x="480" y="60" width="150" height="84" rx="4"/>
  <text class="t1" x="555" y="96" text-anchor="middle">Sampler q</text>
  <text class="t2" x="555" y="118" text-anchor="middle">= trainer + bias</text>
  <text class="alabel" x="320" y="50" text-anchor="middle">1. weights synced to the sampler</text>
  <text class="alabel" x="320" y="70" text-anchor="middle">2. bias added again (quantisation, staleness)</text>
  <line class="arrow" x1="166" y1="84" x2="472" y2="84" marker-end="url(#ah)"/>
  <line class="arrow" x1="474" y1="122" x2="168" y2="122" marker-end="url(#ah)"/>
  <text class="alabel" x="320" y="144" text-anchor="middle">3. rollouts: drift pulls the trainer toward the bias</text>
  <text class="alabel" x="320" y="182" text-anchor="middle">4. repeat, and the error compounds</text>
</svg>
<figcaption>Every time the trainer catches up, the sampler is refreshed from it and the distortion is added back, so the target stays one step ahead.</figcaption>
</figure>

Follow it round once. The trainer takes a step toward the sampler. Its new weights are copied into the sampler, which adds its usual distortion on top. The trainer then steps toward that, and so on. The target never stays still long enough to be reached, so instead of settling down, the error builds up step after step. This is what the paper means when it says drift accumulates over training.

It also clears up something that looks like a paradox. SFT on text written by a completely different model, which is a huge mismatch, works fine. Yet online RL can fall apart over tiny numerical differences. What matters isn't how far apart the two models are. It's whether the teacher stays put or follows the student around. The same idea explains why all-positive ($+1/0$) rewards were the least stable online setting in the paper: the drift is scaled by the average reward, and with no negative rewards to balance it out, that average is as large as it gets.

### Solution: score centering

The fix is to make the average score zero again, as it is on-policy, by subtracting it (Eq. 4):

$$
\tilde s_{y_t} = s_{y_t} - \bar s,
\qquad
\mathbb{E}_q[\tilde s] = \bar s - \bar s = 0 .
\tag{4}
$$

Both averages use the same distribution $q$, so the cancellation is exact even off-policy. The expected update becomes (Eq. 5):

$$
\begin{aligned}
\mathbb{E}_q[R\, \tilde s_{y_t}]
&= \mathbb{E}_q[R]\;\underbrace{\mathbb{E}_q[\tilde s_{y_t}]}_{=\,0} \;+\; \operatorname{Cov}_q(R,\, \tilde s_{y_t}) \\
&= \operatorname{Cov}_q(R,\, s_{y_t} - \bar s) \;=\; \operatorname{Cov}_q(R,\, s_{y_t}) .
\end{aligned}
\tag{5}
$$

The last step holds because $\bar s$ depends only on the prefix, not on the sampled token, so it is a constant as far as the covariance is concerned. The drift is gone and the signal is untouched. The one remaining difference from a true on-policy update is that the covariance is measured under $q$ rather than $p$. When the two agree, $\bar s = 0$ and score centering does nothing.

In the toy model, $\tilde s_A = (0.4, -0.4)$ and $\tilde s_B = (-0.6, 0.6)$, and under the sampler they average to $0.6 \cdot 0.4 + 0.4 \cdot (-0.6) = 0$.

The one-hot vectors from the two-token section give an even simpler form. Averaging one-hot vectors over the sampler's choices just gives back the sampler's probabilities, $\sum_v q_v \mathbf{e}_v = q$. (In the toy model, $0.6\,(1, 0) + 0.4\,(0, 1) = (0.6, 0.4)$.) So the average score is

$$
\bar s = \sum_v q_v (\mathbf{e}_v - p) = \sum_v q_v \mathbf{e}_v - p \sum_v q_v = q - p ,
$$

and the centered score of the sampled token $y$ is

$$
\tilde s_{y} = (\mathbf{e}_y - p) - (q - p) = \mathbf{e}_y - q .
$$

The trainer's $p$ cancels out completely. Check it on the toy model: $\mathbf{e}_A - q = (1, 0) - (0.6, 0.4) = (0.4, -0.4)$, the same $\tilde s_A$ as above. In words: push the sampled token up by 1, and push every token down by how often the sampler picks it. Hold on to this formula; it is the key to the eight logits.

### Try it

The two-token model from above. Move the sliders to set the trainer and the sampler, and see the expected update on $A$'s logit for each method ($B$'s is always the negative). The dashed line is the on-policy target, what you would get with no mismatch at all. Hover a row for details. The importance-sampling rows, and the TIS cap slider, are explained in the next section; they're worth coming back to after reading it.

<div class="play">
  <div class="controls">
    <div class="control">
      <label for="pA">Trainer <i>p</i><sub>A</sub>: <output id="pAout"></output></label>
      <input id="pA" type="range" min="0.05" max="0.95" step="0.01" value="0.5">
    </div>
    <div class="control">
      <label for="qA">Sampler <i>q</i><sub>A</sub>: <output id="qAout"></output></label>
      <input id="qA" type="range" min="0.05" max="0.95" step="0.01" value="0.6">
    </div>
    <div class="control">
      <label for="cap">TIS cap <i>c</i>: <output id="capout"></output></label>
      <input id="cap" type="range" min="1" max="5" step="0.1" value="2">
    </div>
    <div class="control">
      <span class="clabel" id="rewardLabel">Reward</span>
      <div class="seg" role="group" aria-labelledby="rewardLabel">
        <button type="button" data-reward="const" aria-pressed="true">Constant +1</button>
        <button type="button" data-reward="A" aria-pressed="false">A is better</button>
        <button type="button" data-reward="B" aria-pressed="false">B is better</button>
      </div>
    </div>
  </div>
  <div class="presets">
    Presets:
    <button type="button" data-p="0.5" data-q="0.6" data-c="2">toy example</button>
    <button type="button" data-p="0.5" data-q="0.9" data-c="2">skewed sampler (q = 0.9)</button>
    <button type="button" data-p="0.5" data-q="0.5" data-c="2">no mismatch</button>
  </div>
  <p class="derived" id="derived"></p>
  <div class="chart">
    <div class="axis-head">
      <span>Method</span>
      <span class="dir"><span>← toward B</span><span>toward A →</span></span>
      <span class="r">Update</span>
      <span class="r">Gap</span>
    </div>
    <div id="rows"></div>
  </div>
  <div class="legend">
    <span><span class="sw" style="background:var(--toward-a)"></span>pushes toward A</span>
    <span><span class="sw" style="background:var(--toward-b)"></span>pushes toward B</span>
    <span><span class="sw dash"></span>on-policy target</span>
    <span>gap = update − target</span>
  </div>
  <div class="tooltip" id="tip" role="tooltip"></div>
</div>

<script>
(() => {
  const $ = (id) => document.getElementById(id);
  const state = { reward: "const" };

  // Expected update on A's logit for the two-token softmax model.
  // Scores in logit space: s_A = (p_B, -p_B), s_B = (-p_A, p_A), so the A-components are p_B and -p_A.
  function compute(pA, qA, cap, RA, RB) {
    const pB = 1 - pA, qB = 1 - qA;
    const sA = pB, sB = -pA;
    const rA = pA / qA, rB = pB / qB;
    const weighted = (wA, wB) => qA * RA * wA * sA + qB * RB * wB * sB;
    const centered = (wA, wB) => {
      const mean = qA * wA * sA + qB * wB * sB;          // E_q[w s]
      return weighted(wA, wB) - (qA * RA + qB * RB) * mean;
    };
    const tis = (r) => Math.min(r, cap);
    return {
      rA, rB,
      rows: [
        { key: "target", name: "On-policy target", sub: "no mismatch", v: pA * pB * (RA - RB),
          how: "Σ p_v R_v s_v: tokens drawn from the trainer itself." },
        { key: "pg", name: "Vanilla PG", sub: "no correction", v: weighted(1, 1),
          how: "Σ q_v R_v s_v with weights 1, 1." },
        { key: "is", name: "Exact IS", sub: "weight p/q", v: weighted(rA, rB),
          how: `Weights r_A = ${rA.toFixed(2)}, r_B = ${rB.toFixed(2)}.` },
        { key: "tis", name: "TIS", sub: `cap at ${cap.toFixed(1)}`, v: weighted(tis(rA), tis(rB)),
          how: `Weights min(r, ${cap.toFixed(1)}) = ${tis(rA).toFixed(2)}, ${tis(rB).toFixed(2)}.` },
        { key: "sc", name: "Score centering", sub: "subtract s̄", v: centered(1, 1),
          how: "Σ q_v R_v (s_v − s̄): drift removed, signal measured under q." },
        { key: "tissc", name: "TIS + SC", sub: "subtract E_q[w s]", v: centered(tis(rA), tis(rB)),
          how: "Clipped weights, then subtract the average weighted score." },
      ],
    };
  }

  const fmt = (x) => {
    const v = Math.abs(x) < 0.0005 ? 0 : x;
    return (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v).toFixed(3);
  };

  function render() {
    const pA = +$("pA").value, qA = +$("qA").value, cap = +$("cap").value;
    $("pAout").textContent = pA.toFixed(2);
    $("qAout").textContent = qA.toFixed(2);
    $("capout").textContent = cap.toFixed(1);
    const [RA, RB] = state.reward === "const" ? [1, 1] : state.reward === "A" ? [1, 0] : [0, 1];
    const { rA, rB, rows } = compute(pA, qA, cap, RA, RB);
    const target = rows[0].v;
    $("derived").textContent =
      `Rewards R_A = ${RA}, R_B = ${RB}. Drift direction on A: q_A − p_A = ${fmt(qA - pA)}. ` +
      `Importance ratios r_A = ${rA.toFixed(2)}, r_B = ${rB.toFixed(2)}.`;

    const scale = Math.max(0.05, ...rows.map((r) => Math.abs(r.v))) * 1.1;
    const pct = (x) => 50 + (x / scale) * 50;

    $("rows").innerHTML = rows.map((r, i) => {
      const v = Math.abs(r.v) < 0.0005 ? 0 : r.v;
      const left = v >= 0 ? 50 : pct(v);
      const width = Math.abs(pct(v) - 50);
      const gap = r.key === "target" ? "" : fmt(r.v - target);
      return `<div class="row${r.key === "target" ? " target" : ""}" data-i="${i}">
        <span class="name">${r.name}<small>${r.sub}</small></span>
        <span class="track">
          <span class="zero"></span>
          ${width > 0 ? `<span class="bar ${v >= 0 ? "pos" : "neg"}" style="left:${left}%;width:${width}%"></span>` : ""}
          <span class="tgt" style="left:calc(${pct(target)}% - 1px)"></span>
        </span>
        <span class="val">${fmt(r.v)}</span>
        <span class="gap">${gap}</span>
      </div>`;
    }).join("");

    const tip = $("tip");
    document.querySelectorAll("#rows .row").forEach((el) => {
      const r = rows[+el.dataset.i];
      el.addEventListener("mousemove", (e) => {
        const dir = Math.abs(r.v) < 0.0005 ? "no net push" : r.v > 0 ? "toward A" : "toward B";
        tip.innerHTML = `<div><strong>${r.name}</strong>, ${r.sub}</div>
          <div>Update on A's logit: <b>${fmt(r.v)}</b> (${dir})</div>
          ${r.key === "target" ? "" : `<div>Gap from on-policy: <b>${fmt(r.v - target)}</b></div>`}
          <div style="color:var(--fg-muted);margin-top:4px">${r.how}</div>`;
        tip.style.left = Math.min(e.clientX + 14, window.innerWidth - 316) + "px";
        tip.style.top = e.clientY + 14 + "px";
        tip.style.opacity = 1;
      });
      el.addEventListener("mouseleave", () => (tip.style.opacity = 0));
    });
  }

  ["pA", "qA", "cap"].forEach((id) => $(id).addEventListener("input", render));
  document.querySelectorAll(".seg button").forEach((b) =>
    b.addEventListener("click", () => {
      state.reward = b.dataset.reward;
      document.querySelectorAll(".seg button").forEach((x) => x.setAttribute("aria-pressed", x === b));
      render();
    })
  );
  document.querySelectorAll(".presets button").forEach((b) =>
    b.addEventListener("click", () => {
      $("pA").value = b.dataset.p;
      $("qA").value = b.dataset.q;
      $("cap").value = b.dataset.c;
      render();
    })
  );
  render();
})();
</script>

With a constant reward, vanilla policy gradient shows the full drift $q_A - p_A$, while score centering sits on zero however far apart the sampler and trainer are. With a token-dependent reward, score centering always points the right way but lands short of the target by an amount that grows with the mismatch. That is the “covariance under $q$” caveat from Eq. 5.

The importance-sampling rows preview the next section. Exact importance sampling also sits on zero, and so does TIS until its cap binds: with $p_A = 0.5$ and a cap of 2, that happens once $q_A > 0.75$, after which drift leaks back in and TIS + score centering removes it again. One thing the playground can't show is variance. These are expected values, so exact importance sampling always looks perfect here, even though variance is exactly what sinks it in practice.

### Doesn't importance sampling already fix this?

**Exact** importance sampling does. The idea is to reweight each sampled token by how much more or less likely the trainer is to produce it than the sampler, using the **importance ratio**

$$
r_t = \frac{p(y_t)}{q(y_t)} .
$$

The per-token loss multiplies the usual policy-gradient loss by this ratio, held fixed with a stop-gradient:

$$
L_{\text{IS}} = -\,\mathrm{sg}[r_t]\; R\, \log p(y_t)
\qquad\Longrightarrow\qquad
-\nabla L_{\text{IS}} = r_t\, R\, s_{y_t} .
$$

Tokens the sampler over-picks ($q > p$) get a weight below 1, and tokens it under-picks get a weight above 1. Under a constant reward the average update is

$$
\sum_v q_v \cdot \frac{p_v}{q_v} \cdot s_v = \sum_v p_v\, s_v = 0,
$$

so the $q$'s cancel and the drift is gone. In the toy model the ratios are $r_A = 0.5/0.6 = 0.83$ and $r_B = 0.5/0.4 = 1.25$, and they cancel the drift exactly. The problem is variance. For a token the sampler is much less likely to produce than the trainer ($q \ll p$), the ratio $p/q$ can be enormous, even if both probabilities are small: $p = 0.01$ and $q = 0.0001$ give $r = 100$. That token is rarely sampled, but when it is, it can dominate a step. In the paper's experiments the raw version (“naive IS”) collapses under staleness and ends below 5% at the heaviest 30B quantisation.

So nobody uses it raw. The most common fix is **truncated importance sampling (TIS)**, which caps the weight at a constant $C$:

$$
w_t = \min(r_t,\ C),
\qquad
L_{\text{TIS}} = -\,\mathrm{sg}[w_t]\; R\, \log p(y_t),
$$

with $C = 2$ in the paper's experiments. Other methods limit the ratio in other ways. Masked importance sampling (MIS) sets $w_t = 0$ whenever $r_t$ falls outside a range ($[0.5, 5]$ in the paper), and PPO-style objectives clip $r_t$ to $[1-\varepsilon,\ 1+\varepsilon]$. All of them trade some correctness for lower variance. Once the weights are altered, the $q$'s no longer cancel and drift leaks back in.

Take a badly skewed sampler, $q = (0.9, 0.1)$, with $p = (0.5, 0.5)$. The ratios are $r_A = 0.5/0.9 = 0.56$ and $r_B = 0.5/0.1 = 5$. TIS caps $r_B$ at 2, so the average push on $A$'s logit under a constant reward is

$$
\underbrace{0.9 \times 0.56 \times 0.5}_{A\ \text{sampled}} \;+\; \underbrace{0.1 \times 2 \times (-0.5)}_{B\ \text{sampled}}
\;=\; 0.25 - 0.10 \;=\; 0.15 .
$$

Without the cap, B's term would be $0.1 \times 5 \times (-0.5) = -0.25$ and the total would be exactly zero.

The table below does the same calculation for each method. **Weights (A, B)** are the weights $w_A$ and $w_B$ that the method multiplies the update by when $A$ or $B$ is sampled. The **average push** on $A$'s logit adds up each token's contribution: how often the sampler draws it ($q_v$), times its weight, times its push on $A$'s logit ($+0.5$ for $A$, $-0.5$ for $B$), with $R = 1$:

$$
\text{average push} = q_A\, w_A \,(+0.5) \;+\; q_B\, w_B\,(-0.5) = 0.9\, w_A\,(0.5) - 0.1\, w_B\,(0.5).
$$

| Method | Weights (A, B) | Calculation | Average push on A |
|---|---:|---:|---:|
| Vanilla policy gradient | 1, 1 | $0.45 - 0.05$ | 0.40 |
| Exact importance sampling | 0.56, 5 | $0.25 - 0.25$ | 0 |
| TIS, cap at 2 | 0.56, 2 | $0.25 - 0.10$ | 0.15 |
| TIS + score centering | 0.56, 2 | $0.15 - 0.15$ | 0 |

In the last row, score centering subtracts the average weighted score, which is exactly the 0.15 that TIS leaves behind.

Clipping cuts the drift from 0.40 to 0.15, but not to zero. The two corrections also work differently. Importance sampling is *multiplicative*, random, and unbounded. Score centering is *additive* and fixed once you know the prefix. Because they are independent, they compose: apply clipped weights $w_v$, then subtract the average weighted score $\sum_v q_v w_v s_v$ (Appendix A.2 of the paper).

| Method | Drift | Variance |
|---|---|---|
| Vanilla policy gradient | full | low |
| Exact importance sampling | none | very high |
| Clipped or masked IS (TIS, MIS, PPO-style) | partial | moderate |
| Score centering | none | low |
| Clipped IS + score centering | none | moderate |

### So what does the paper add?

Two things. The first is a diagnosis. Importance sampling fixes several things at once, so its success never told us *which* problem it was fixing. Score centering removes only the drift, with no ratios, no clipping and no masking, and that alone is enough to stabilise training under severe quantisation. That identifies drift as the cause of the instability.

The second is a method that holds up when the mismatch gets bad. On Qwen3-30B-A3B with math RL, under three levels of sampler quantisation:

| Sampler | Outcome |
|---|---|
| FP8 weights and activations, FP8 KV cache | even uncorrected policy gradient trains stably (58%); nothing to separate |
| FP8 + FP4 KV cache | policy gradient collapses, MIS collapses late; score centering 52%, TIS 51% |
| INT8 + INT4 KV cache | score centering 30%, TIS 12%, every other method below 5% |

The gap grows as the mismatch gets worse. Under heavy staleness (the sampler refreshed only every 64 steps), TIS + score centering beat score centering alone. The authors attribute this to the Eq. 5 caveat: when the trainer moves far from the sampler, measuring the signal under $q$ starts to cost something, and importance sampling helps. Their recommendation is to compose the two whenever the trainer can move far between syncs.

> The settings were made severe on purpose. Under mild mismatch the best methods overlapped within the authors' compute budget, and they argue that mild mismatch causes the same problem over longer runs. The experiments cover Qwen3 models on Countdown and math only, with some single-seed 30B runs.

### Making it cheap: top-k and a tail model

Score centering needs $\bar s$ as an actual number at every position. As a loss, with $\mathrm{sg}$ meaning stop-gradient:

$$
L = -R\Big(\log p_{y_t} - \sum_v \mathrm{sg}[q_v]\,\log p_v\Big)
\quad\Longrightarrow\quad
-\nabla L = R\,(s_{y_t} - \bar s) = R\,\tilde s_{y_t}.
$$

The trainer already has every $\log p_v$, because its softmax runs over the whole vocabulary. The problem is $q$. Storing the sampler's full 152K-entry distribution for every token would take about 20 TB per batch. So the sampler logs only its top $k = 128$ probabilities (the head $H$), and the tail $T$ is filled in with the trainer's own probabilities, rescaled to the right mass (Eq. 7):

$$
\hat q_v = \begin{cases} q_v & v \in H \\ \rho\, p_v & v \in T \end{cases}
\qquad
\rho = \frac{1 - \sum_{v\in H} q_v}{1 - \sum_{v\in H} p_v}.
\tag{7}
$$

Why model the tail at all, instead of just using the top 128? Because the head alone doesn't sum to one. Treating the missing mass as zero, or renormalising the head to cover it, both misstate the sampler's distribution, so the subtracted term no longer has exactly zero mean and a little drift survives. Filling the tail with the trainer's probabilities, which are already computed, only assumes the two engines agree on the tail's shape up to its total mass, and Eq. 8 shows it costs nothing extra.

Since $\sum_v p_v s_v = 0$, the tail sum equals minus the head sum, $\sum_T p_v s_v = -\sum_H p_v s_v$, and the whole expectation collapses onto the head (Eq. 8):

$$
\mathbb{E}_{\hat q}[s] = \sum_{H} q_v s_v + \rho \sum_{T} p_v s_v = \sum_{v\in H} (q_v - \rho\, p_v)\, s_v .
\tag{8}
$$

That gives the paper's implementation loss (Eq. 6), a sum over just 128 tokens per position:

$$
L = -R\Big(\log p_{y_t} - \sum_{v\in H} \mathrm{sg}\big[q_v - \rho\, p_v\big]\,\log p_v\Big).
\tag{6}
$$

```python
def score_centering_loss(train_logp, samp_logp, topk_ids, sampled_token, advantage):
    train_head_logp = train_logp[topk_ids]
    tail_mass_ratio = (1 - jnp.exp(samp_logp).sum()) / (1 - jnp.exp(train_head_logp).sum())   # ρ
    head_prob_residual = jnp.exp(samp_logp) - tail_mass_ratio * jnp.exp(train_head_logp)      # q_v − ρ p_v
    logp_correction = (stop_gradient(head_prob_residual) * train_head_logp).sum()
    return -advantage * (train_logp[sampled_token] - logp_correction)                         # Eq. 6
```

The reported overhead is under 1% of wall-clock time, and $k = 32$ already matches full-vocabulary centering in every setting they tested.

### Back to the eight logits

We can now decode the thread. In logit space, the centered score is always $\mathbf{e}_y$ minus *some estimate* $\hat q$ of the sampler's distribution. What differs between implementations is which estimate you use.

**The paper's top-k version.** Expanding Eq. 8 with $s_v = \mathbf{e}_v - p$, and using $\sum_H (q_v - \rho p_v) = 1 - \rho$:

$$
\tilde s_y = \mathbf{e}_y - \sum_{v \in H} (q_v - \rho\, p_v)\,\mathbf{e}_v - \rho\, p
\;=\; \mathbf{e}_y - \hat q .
$$

The $-\rho\, p$ term touches every logit, so this gradient is dense. It is deterministic, and slightly biased wherever the tail model is wrong.

**StableReinforce's sampled version.** Draw $K = 8$ tokens $v_1, \dots, v_K$ from the sampler and estimate the average score by their mean:

$$
\bar s \approx \frac{1}{K}\sum_{j=1}^{K} s_{v_j} = \frac{1}{K}\sum_{j} \mathbf{e}_{v_j} - p
\quad\Longrightarrow\quad
\tilde s_y = (\mathbf{e}_y - p) - \Big(\frac{1}{K}\sum_j \mathbf{e}_{v_j} - p\Big) = \mathbf{e}_y - \frac{1}{K}\sum_{j=1}^{K}\mathbf{e}_{v_j}.
$$

The dense $-p$ part is shared by every token's score, so it cancels exactly. What remains is non-zero only on the sampled token and the correction tokens: a handful of logits out of 152K. In Marek's single-token test that came to eight. Since $\mathbb{E}[\mathbf{e}_{v_j}] = q$, the estimate is unbiased; it averages to $\mathbf{e}_y - q$.

<figure>
<svg viewBox="0 0 760 450" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Exact, top-k and sampled centering" font-family="ui-serif, Georgia, 'Times New Roman', serif" style="display:block;width:100%;height:auto">
<defs><marker id="arr-1" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#666666"/></marker><marker id="arra-1" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#0b5fff"/></marker></defs>
<text x="40" y="24" fill="#1a1a1a" font-size="13.5" font-weight="600">Exact: eᵧ − q</text><text x="750" y="24" fill="#666666" font-size="12.5" text-anchor="end">needs the full sampler distribution</text><line x1="40" y1="110.92307692307692" x2="750" y2="110.92307692307692" stroke="#666666" stroke-width="1"/><text x="34" y="114.92307692307692" fill="#666666" font-size="11.5" text-anchor="end">0</text><line x1="40" y1="72.46153846153845" x2="750" y2="72.46153846153845" stroke="#e5e5e5" stroke-width="1"/><text x="34" y="76.46153846153845" fill="#666666" font-size="11.5" text-anchor="end">0.5</text><line x1="40" y1="34" x2="750" y2="34" stroke="#e5e5e5" stroke-width="1"/><text x="34" y="38" fill="#666666" font-size="11.5" text-anchor="end">1</text><path d="M41,110.92307692307692 L41,111.28172057255577 Q41,111.64036422203463 41.35864364947886,111.64036422203463 L56.39135635052114,111.64036422203463 Q56.75,111.64036422203463 56.75,111.28172057255577 L56.75,110.92307692307692 Z" fill="#e34948"><title>token 1: −0.009 (exact)</title></path><path d="M58.75,110.92307692307692 L58.75,111.25802297886483 Q58.75,111.59296903465273 59.0849460557879,111.59296903465273 L74.1650539442121,111.59296903465273 Q74.5,111.59296903465273 74.5,111.25802297886483 L74.5,110.92307692307692 Z" fill="#e34948"><title>token 2: −0.009 (exact)</title></path><path d="M94.25,110.92307692307692 L94.25,111.51754015801674 Q94.25,112.11200339295655 94.84446323493981,112.11200339295655 L109.40553676506019,112.11200339295655 Q110,112.11200339295655 110,111.51754015801674 L110,110.92307692307692 Z" fill="#e34948"><title>token 4: −0.015 (exact)</title></path><path d="M112,110.92307692307692 L112,112.2762736808073 Q112,113.62947043853767 113.35319675773037,113.62947043853767 L126.39680324226963,113.62947043853767 Q127.75,113.62947043853767 127.75,112.2762736808073 L127.75,110.92307692307692 Z" fill="#e34948"><title>token 5: −0.035 (exact)</title></path><path d="M129.75,110.92307692307692 L129.75,112.58907736395523 Q129.75,114.25507780483353 131.4160004408783,114.25507780483353 L143.8339995591217,114.25507780483353 Q145.5,114.25507780483353 145.5,112.58907736395523 L145.5,110.92307692307692 Z" fill="#e34948"><title>token 6: −0.043 (exact)</title></path><path d="M147.5,110.92307692307692 L147.5,111.30301904535831 Q147.5,111.6829611676397 147.8799421222814,111.6829611676397 L162.8700578777186,111.6829611676397 Q163.25,111.6829611676397 163.25,111.30301904535831 L163.25,110.92307692307692 Z" fill="#e34948"><title>token 7: −0.010 (exact)</title></path><path d="M165.25,110.92307692307692 L165.25,111.40760045451673 Q165.25,111.89212398595653 165.7345235314398,111.89212398595653 L180.5154764685602,111.89212398595653 Q181,111.89212398595653 181,111.40760045451673 L181,110.92307692307692 Z" fill="#e34948"><title>token 8: −0.013 (exact)</title></path><path d="M183,110.92307692307692 L183,113.97635295819404 Q183,117.02962899331118 186.05327603511714,117.02962899331118 L195.69672396488286,117.02962899331118 Q198.75,117.02962899331118 198.75,113.97635295819404 L198.75,110.92307692307692 Z" fill="#e34948"><title>token 9: −0.079 (exact)</title></path><path d="M200.75,110.92307692307692 L200.75,111.68439954664449 Q200.75,112.44572217021205 201.51132262356757,112.44572217021205 L215.73867737643243,112.44572217021205 Q216.5,112.44572217021205 216.5,111.68439954664449 L216.5,110.92307692307692 Z" fill="#e34948"><title>token 10: −0.020 (exact)</title></path><path d="M236.25,110.92307692307692 L236.25,111.44154150311098 Q236.25,111.96000608314503 236.76846458003405,111.96000608314503 L251.48153541996595,111.96000608314503 Q252,111.96000608314503 252,111.44154150311098 L252,110.92307692307692 Z" fill="#e34948"><title>token 12: −0.013 (exact)</title></path><path d="M254,110.92307692307692 L254,111.27328170649994 Q254,111.62348648992298 254.35020478342304,111.62348648992298 L269.399795216577,111.62348648992298 Q269.75,111.62348648992298 269.75,111.27328170649994 L269.75,110.92307692307692 Z" fill="#e34948"><title>token 13: −0.009 (exact)</title></path><path d="M307.25,110.92307692307692 L307.25,111.57933045807533 Q307.25,112.23558399307373 307.9062535349984,112.23558399307373 L322.3437464650016,112.23558399307373 Q323,112.23558399307373 323,111.57933045807533 L323,110.92307692307692 Z" fill="#e34948"><title>token 16: −0.017 (exact)</title></path><path d="M325,110.92307692307692 L325,111.26512510518776 Q325,111.6071732872986 325.34204818211083,111.6071732872986 L340.40795181788917,111.6071732872986 Q340.75,111.6071732872986 340.75,111.26512510518776 L340.75,110.92307692307692 Z" fill="#e34948"><title>token 17: −0.009 (exact)</title></path><path d="M360.5,110.92307692307692 L360.5,111.19765315939041 Q360.5,111.4722293957039 360.77457623631346,111.4722293957039 L375.97542376368654,111.4722293957039 Q376.25,111.4722293957039 376.25,111.19765315939041 L376.25,110.92307692307692 Z" fill="#e34948"><title>token 19: −0.007 (exact)</title></path><path d="M396,110.92307692307692 L396,111.64746628298471 Q396,112.37185564289251 396.7243893599078,112.37185564289251 L411.0256106400922,112.37185564289251 Q411.75,112.37185564289251 411.75,111.64746628298471 L411.75,110.92307692307692 Z" fill="#e34948"><title>token 21: −0.019 (exact)</title></path><path d="M413.75,110.92307692307692 L413.75,112.53317124592357 Q413.75,114.1432655687702 415.36009432284663,114.1432655687702 L427.88990567715337,114.1432655687702 Q429.5,114.1432655687702 429.5,112.53317124592357 L429.5,110.92307692307692 Z" fill="#e34948"><title>token 22: −0.042 (exact)</title></path><path d="M449.25,110.92307692307692 L449.25,111.28426200218124 Q449.25,111.64544708128554 449.6111850791043,111.64544708128554 L464.6388149208957,111.64544708128554 Q465,111.64544708128554 465,111.28426200218124 L465,110.92307692307692 Z" fill="#e34948"><title>token 24: −0.009 (exact)</title></path><path d="M467,110.92307692307692 L467,116.02048906500937 Q467,120.02048906500937 471,120.02048906500937 L478.75,120.02048906500937 Q482.75,120.02048906500937 482.75,116.02048906500937 L482.75,110.92307692307692 Z" fill="#e34948"><title>token 25: −0.118 (exact)</title></path><path d="M484.75,110.92307692307692 L484.75,111.59101289421177 Q484.75,112.25894886534662 485.41793597113485,112.25894886534662 L499.83206402886515,112.25894886534662 Q500.5,112.25894886534662 500.5,111.59101289421177 L500.5,110.92307692307692 Z" fill="#e34948"><title>token 26: −0.017 (exact)</title></path><path d="M520.25,110.92307692307692 L520.25,111.56218968747822 Q520.25,112.20130245187951 520.8891127644013,112.20130245187951 L535.3608872355987,112.20130245187951 Q536,112.20130245187951 536,111.56218968747822 L536,110.92307692307692 Z" fill="#e34948"><title>token 28: −0.017 (exact)</title></path><path d="M538,110.92307692307692 L538,112.29995874302236 Q538,113.67684056296778 539.3768818199454,113.67684056296778 L552.3731181800546,113.67684056296778 Q553.75,113.67684056296778 553.75,112.29995874302236 L553.75,110.92307692307692 Z" fill="#e34948"><title>token 29: −0.036 (exact)</title></path><path d="M555.75,110.92307692307692 L555.75,111.36850241400138 Q555.75,111.81392790492585 556.1954254909244,111.81392790492585 L571.0545745090756,111.81392790492585 Q571.5,111.81392790492585 571.5,111.36850241400138 L571.5,110.92307692307692 Z" fill="#e34948"><title>token 30: −0.012 (exact)</title></path><path d="M573.5,110.92307692307692 L573.5,112.39448943338445 Q573.5,113.86590194369197 574.9714125103076,113.86590194369197 L587.7785874896924,113.86590194369197 Q589.25,113.86590194369197 589.25,112.39448943338445 L589.25,110.92307692307692 Z" fill="#e34948"><title>token 31: −0.038 (exact)</title></path><path d="M591.25,110.92307692307692 L591.25,111.53314557698423 Q591.25,112.14321423089152 591.8600686539073,112.14321423089152 L606.3899313460927,112.14321423089152 Q607,112.14321423089152 607,111.53314557698423 L607,110.92307692307692 Z" fill="#e34948"><title>token 32: −0.016 (exact)</title></path><path d="M609,110.92307692307692 L609,111.2573192959073 Q609,111.59156166873767 609.3342423728304,111.59156166873767 L624.4157576271696,111.59156166873767 Q624.75,111.59156166873767 624.75,111.2573192959073 L624.75,110.92307692307692 Z" fill="#e34948"><title>token 33: −0.009 (exact)</title></path><path d="M662.25,110.92307692307692 L662.25,111.22260900007485 Q662.25,111.52214107707279 662.5495320769979,111.52214107707279 L677.7004679230021,111.52214107707279 Q678,111.52214107707279 678,111.22260900007485 L678,110.92307692307692 Z" fill="#e34948"><title>token 36: −0.008 (exact)</title></path><path d="M680,110.92307692307692 L680,111.27689260486378 Q680,111.63070828665064 680.3538156817868,111.63070828665064 L695.3961843182132,111.63070828665064 Q695.75,111.63070828665064 695.75,111.27689260486378 L695.75,110.92307692307692 Z" fill="#e34948"><title>token 37: −0.009 (exact)</title></path><path d="M697.75,110.92307692307692 L697.75,111.2675842730275 Q697.75,111.61209162297808 698.0945073499506,111.61209162297808 L713.1554926500494,111.61209162297808 Q713.5,111.61209162297808 713.5,111.2675842730275 L713.5,110.92307692307692 Z" fill="#e34948"><title>token 38: −0.009 (exact)</title></path><path d="M715.5,110.92307692307692 L715.5,62.55341701005898 Q715.5,58.55341701005898 719.5,58.55341701005898 L727.25,58.55341701005898 Q731.25,58.55341701005898 731.25,62.55341701005898 L731.25,110.92307692307692 Z" fill="#2a78d6"><title>token 39: +0.681 (exact)</title></path><text x="709.5" y="88.73824696656794" fill="#1a1a1a" font-size="12" text-anchor="end">sampled token →</text><text x="40" y="174" fill="#1a1a1a" font-size="13.5" font-weight="600">Paper: top-8 head + tail model, eᵧ − q̂</text><text x="750" y="174" fill="#666666" font-size="12.5" text-anchor="end">deterministic, slightly biased, dense</text><line x1="40" y1="260.9230769230769" x2="750" y2="260.9230769230769" stroke="#666666" stroke-width="1"/><text x="34" y="264.9230769230769" fill="#666666" font-size="11.5" text-anchor="end">0</text><line x1="40" y1="222.46153846153845" x2="750" y2="222.46153846153845" stroke="#e5e5e5" stroke-width="1"/><text x="34" y="226.46153846153845" fill="#666666" font-size="11.5" text-anchor="end">0.5</text><line x1="40" y1="184" x2="750" y2="184" stroke="#e5e5e5" stroke-width="1"/><text x="34" y="188" fill="#666666" font-size="11.5" text-anchor="end">1</text><path d="M94.25,260.9230769230769 L94.25,261.3401166969735 Q94.25,261.7571564708701 94.66703977389659,261.7571564708701 L109.58296022610341,261.7571564708701 Q110,261.7571564708701 110,261.3401166969735 L110,260.9230769230769 Z" fill="#e34948"><title>token 4: −0.011 (top-k)</title></path><path d="M112,260.9230769230769 L112,262.2762736808073 Q112,263.6294704385377 113.35319675773039,263.6294704385377 L126.39680324226961,263.6294704385377 Q127.75,263.6294704385377 127.75,262.2762736808073 L127.75,260.9230769230769 Z" fill="#e34948"><title>token 5: −0.035 (top-k)</title></path><path d="M129.75,260.9230769230769 L129.75,262.5890773639552 Q129.75,264.25507780483355 131.41600044087832,264.25507780483355 L143.83399955912168,264.25507780483355 Q145.5,264.25507780483355 145.5,262.5890773639552 L145.5,260.9230769230769 Z" fill="#e34948"><title>token 6: −0.043 (top-k)</title></path><path d="M147.5,260.9230769230769 L147.5,261.4099558554732 Q147.5,261.89683478786947 147.98687893239628,261.89683478786947 L162.76312106760372,261.89683478786947 Q163.25,261.89683478786947 163.25,261.4099558554732 L163.25,260.9230769230769 Z" fill="#e34948"><title>token 7: −0.013 (top-k)</title></path><path d="M165.25,260.9230769230769 L165.25,261.6931379517689 Q165.25,262.46319898046096 166.02006102869203,262.46319898046096 L180.22993897130797,262.46319898046096 Q181,262.46319898046096 181,261.6931379517689 L181,260.9230769230769 Z" fill="#e34948"><title>token 8: −0.020 (top-k)</title></path><path d="M183,260.9230769230769 L183,263.97635295819407 Q183,267.0296289933112 186.05327603511714,267.0296289933112 L195.69672396488286,267.0296289933112 Q198.75,267.0296289933112 198.75,263.97635295819407 L198.75,260.9230769230769 Z" fill="#e34948"><title>token 9: −0.079 (top-k)</title></path><path d="M200.75,260.9230769230769 L200.75,261.6302866392392 Q200.75,262.3374963554015 201.4572097161623,262.3374963554015 L215.7927902838377,262.3374963554015 Q216.5,262.3374963554015 216.5,261.6302866392392 L216.5,260.9230769230769 Z" fill="#e34948"><title>token 10: −0.018 (top-k)</title></path><path d="M236.25,260.9230769230769 L236.25,261.4186696565382 Q236.25,261.91426238999946 236.74559273346128,261.91426238999946 L251.50440726653872,261.91426238999946 Q252,261.91426238999946 252,261.4186696565382 L252,260.9230769230769 Z" fill="#e34948"><title>token 12: −0.013 (top-k)</title></path><path d="M307.25,260.9230769230769 L307.25,261.4759328337571 Q307.25,262.0287887444373 307.8028559106802,262.0287887444373 L322.4471440893198,262.0287887444373 Q323,262.0287887444373 323,261.4759328337571 L323,260.9230769230769 Z" fill="#e34948"><title>token 16: −0.014 (top-k)</title></path><path d="M325,260.9230769230769 L325,261.2280686649676 Q325,261.5330604068584 325.3049917418907,261.5330604068584 L340.4450082581093,261.5330604068584 Q340.75,261.5330604068584 340.75,261.2280686649676 L340.75,260.9230769230769 Z" fill="#e34948"><title>token 17: −0.008 (top-k)</title></path><path d="M360.5,260.9230769230769 L360.5,261.2920015110582 Q360.5,261.6609260990395 360.86892458798127,261.6609260990395 L375.88107541201873,261.6609260990395 Q376.25,261.6609260990395 376.25,261.2920015110582 L376.25,260.9230769230769 Z" fill="#e34948"><title>token 19: −0.010 (top-k)</title></path><path d="M396,260.9230769230769 L396,261.7735264414211 Q396,262.6239759597653 396.8504495183442,262.6239759597653 L410.8995504816558,262.6239759597653 Q411.75,262.6239759597653 411.75,261.7735264414211 L411.75,260.9230769230769 Z" fill="#e34948"><title>token 21: −0.022 (top-k)</title></path><path d="M413.75,260.9230769230769 L413.75,262.53317124592354 Q413.75,264.1432655687702 415.36009432284663,264.1432655687702 L427.88990567715337,264.1432655687702 Q429.5,264.1432655687702 429.5,262.53317124592354 L429.5,260.9230769230769 Z" fill="#e34948"><title>token 22: −0.042 (top-k)</title></path><path d="M467,260.9230769230769 L467,266.0204890650094 Q467,270.0204890650094 471,270.0204890650094 L478.75,270.0204890650094 Q482.75,270.0204890650094 482.75,266.0204890650094 L482.75,260.9230769230769 Z" fill="#e34948"><title>token 25: −0.118 (top-k)</title></path><path d="M484.75,260.9230769230769 L484.75,261.6090752324127 Q484.75,262.2950735417485 485.4359983093358,262.2950735417485 L499.8140016906642,262.2950735417485 Q500.5,262.2950735417485 500.5,261.6090752324127 L500.5,260.9230769230769 Z" fill="#e34948"><title>token 26: −0.018 (top-k)</title></path><path d="M520.25,260.9230769230769 L520.25,261.66279875376836 Q520.25,262.40252058445986 520.9897218306915,262.40252058445986 L535.2602781693085,262.40252058445986 Q536,262.40252058445986 536,261.66279875376836 L536,260.9230769230769 Z" fill="#e34948"><title>token 28: −0.019 (top-k)</title></path><path d="M538,260.9230769230769 L538,262.29995874302233 Q538,263.67684056296775 539.3768818199454,263.67684056296775 L552.3731181800546,263.67684056296775 Q553.75,263.67684056296775 553.75,262.29995874302233 L553.75,260.9230769230769 Z" fill="#e34948"><title>token 29: −0.036 (top-k)</title></path><path d="M555.75,260.9230769230769 L555.75,261.6000441968639 Q555.75,262.2770114706509 556.426967273787,262.2770114706509 L570.823032726213,262.2770114706509 Q571.5,262.2770114706509 571.5,261.6000441968639 L571.5,260.9230769230769 Z" fill="#e34948"><title>token 30: −0.018 (top-k)</title></path><path d="M573.5,260.9230769230769 L573.5,262.3944894333844 Q573.5,263.86590194369194 574.9714125103076,263.86590194369194 L587.7785874896924,263.86590194369194 Q589.25,263.86590194369194 589.25,262.3944894333844 L589.25,260.9230769230769 Z" fill="#e34948"><title>token 31: −0.038 (top-k)</title></path><path d="M591.25,260.9230769230769 L591.25,261.3876872429338 Q591.25,261.8522975627907 591.7146103198569,261.8522975627907 L606.5353896801431,261.8522975627907 Q607,261.8522975627907 607,261.3876872429338 L607,260.9230769230769 Z" fill="#e34948"><title>token 32: −0.012 (top-k)</title></path><path d="M609,260.9230769230769 L609,261.51101632001405 Q609,262.0989557169512 609.5879393969371,262.0989557169512 L624.1620606030629,262.0989557169512 Q624.75,262.0989557169512 624.75,261.51101632001405 L624.75,260.9230769230769 Z" fill="#e34948"><title>token 33: −0.015 (top-k)</title></path><path d="M644.5,260.9230769230769 L644.5,261.1797726149671 Q644.5,261.43646830685725 644.7566956918902,261.43646830685725 L659.9933043081098,261.43646830685725 Q660.25,261.43646830685725 660.25,261.1797726149671 L660.25,260.9230769230769 Z" fill="#e34948"><title>token 35: −0.007 (top-k)</title></path><path d="M680,260.9230769230769 L680,261.27114972345885 Q680,261.61922252384073 680.3480728003819,261.61922252384073 L695.4019271996181,261.61922252384073 Q695.75,261.61922252384073 695.75,261.27114972345885 L695.75,260.9230769230769 Z" fill="#e34948"><title>token 37: −0.009 (top-k)</title></path><path d="M697.75,260.9230769230769 L697.75,261.25254382418507 Q697.75,261.5820107252933 698.0794669011082,261.5820107252933 L713.1705330988918,261.5820107252933 Q713.5,261.5820107252933 713.5,261.25254382418507 L713.5,260.9230769230769 Z" fill="#e34948"><title>token 38: −0.009 (top-k)</title></path><path d="M715.5,260.9230769230769 L715.5,212.55341701005898 Q715.5,208.55341701005898 719.5,208.55341701005898 L727.25,208.55341701005898 Q731.25,208.55341701005898 731.25,212.55341701005898 L731.25,260.9230769230769 Z" fill="#2a78d6"><title>token 39: +0.681 (top-k)</title></path><text x="709.5" y="238.73824696656794" fill="#1a1a1a" font-size="12" text-anchor="end">sampled token →</text><text x="40" y="324" fill="#1a1a1a" font-size="13.5" font-weight="600">StableReinforce: 8 sampled tokens</text><text x="750" y="324" fill="#666666" font-size="12.5" text-anchor="end">unbiased, noisy, sparse</text><line x1="40" y1="410.9230769230769" x2="750" y2="410.9230769230769" stroke="#666666" stroke-width="1"/><text x="34" y="414.9230769230769" fill="#666666" font-size="11.5" text-anchor="end">0</text><line x1="40" y1="372.46153846153845" x2="750" y2="372.46153846153845" stroke="#e5e5e5" stroke-width="1"/><text x="34" y="376.46153846153845" fill="#666666" font-size="11.5" text-anchor="end">0.5</text><line x1="40" y1="334" x2="750" y2="334" stroke="#e5e5e5" stroke-width="1"/><text x="34" y="338" fill="#666666" font-size="11.5" text-anchor="end">1</text><path d="M41,410.9230769230769 L41,416.53846153846155 Q41,420.53846153846155 45,420.53846153846155 L52.75,420.53846153846155 Q56.75,420.53846153846155 56.75,416.53846153846155 L56.75,410.9230769230769 Z" fill="#e34948"><title>token 1: −0.125 (sampled)</title></path><path d="M58.75,410.9230769230769 L58.75,416.53846153846155 Q58.75,420.53846153846155 62.75,420.53846153846155 L70.5,420.53846153846155 Q74.5,420.53846153846155 74.5,416.53846153846155 L74.5,410.9230769230769 Z" fill="#e34948"><title>token 2: −0.125 (sampled)</title></path><path d="M218.5,410.9230769230769 L218.5,416.53846153846155 Q218.5,420.53846153846155 222.5,420.53846153846155 L230.25,420.53846153846155 Q234.25,420.53846153846155 234.25,416.53846153846155 L234.25,410.9230769230769 Z" fill="#e34948"><title>token 11: −0.125 (sampled)</title></path><path d="M325,410.9230769230769 L325,416.53846153846155 Q325,420.53846153846155 329,420.53846153846155 L336.75,420.53846153846155 Q340.75,420.53846153846155 340.75,416.53846153846155 L340.75,410.9230769230769 Z" fill="#e34948"><title>token 17: −0.125 (sampled)</title></path><path d="M467,410.9230769230769 L467,416.53846153846155 Q467,420.53846153846155 471,420.53846153846155 L478.75,420.53846153846155 Q482.75,420.53846153846155 482.75,416.53846153846155 L482.75,410.9230769230769 Z" fill="#e34948"><title>token 25: −0.125 (sampled)</title></path><path d="M538,410.9230769230769 L538,416.53846153846155 Q538,420.53846153846155 542,420.53846153846155 L549.75,420.53846153846155 Q553.75,420.53846153846155 553.75,416.53846153846155 L553.75,410.9230769230769 Z" fill="#e34948"><title>token 29: −0.125 (sampled)</title></path><path d="M573.5,410.9230769230769 L573.5,416.53846153846155 Q573.5,420.53846153846155 577.5,420.53846153846155 L585.25,420.53846153846155 Q589.25,420.53846153846155 589.25,416.53846153846155 L589.25,410.9230769230769 Z" fill="#e34948"><title>token 31: −0.125 (sampled)</title></path><path d="M715.5,410.9230769230769 L715.5,347.61538461538464 Q715.5,343.61538461538464 719.5,343.61538461538464 L727.25,343.61538461538464 Q731.25,343.61538461538464 731.25,347.61538461538464 L731.25,410.9230769230769 Z" fill="#2a78d6"><title>token 39: +0.875 (sampled)</title></path><text x="709.5" y="381.2692307692308" fill="#1a1a1a" font-size="12" text-anchor="end">sampled token →</text>
</svg>
<figcaption>

**Figure 5.** Three versions of the centered logit gradient for the same toy token as Figure 3. The paper's top-k version (here $k = 8$ to match the sample budget; the paper uses 128) is dense and nearly identical to the exact one. The 8-sample version is sparse and noisy, but equal to the exact one on average.

</figcaption>
</figure>

| | Paper: top-k + tail | StableReinforce: 8 sampled tokens |
|---|---|---|
| Estimate of $q$ | top-128 probabilities, trainer fills the tail | empirical distribution of 8 draws |
| Bias | slight (tail model) | none |
| Variance | none (deterministic) | some, but bounded: each term is a one-hot vector |
| Logit gradient | dense | sparse (sampled token + correction tokens) |
| What the sampler logs | 128 logprobs per position | 8 token ids per position |

Estimating $\bar s$ by sampling might look like it reintroduces the variance problem. It does add noise, but of a very different kind from importance sampling's. Every term in the estimate is a one-hot vector, so nothing can blow up the way $p/q$ does on a rare token. Thinking Machines evidently decided that trade was worth it.

## Summary

- **Mismatch adds a drift term.** Under training-inference mismatch, the policy gradient contains $\mathbb{E}_q[R]\,\bar s$, which ignores which token earned which reward.
- **Drift is distillation toward a moving teacher.** The trainer is pulled toward a biased copy of itself that is re-synced every step, so the error compounds instead of converging.
- **Score centering removes it exactly.** Subtracting the average score $\bar s$ cancels drift at every prefix and leaves the signal intact, measured under $q$ instead of $p$.
- **Importance sampling fixes it only in theory.** Exact ratios cancel drift but are too noisy, and the clipped versions everyone uses let some of it back. The two corrections compose, and the combination is best under heavy staleness.
- **It is cheap.** The expected score is a sum over the next token's distribution, computable from the top-128 sampler logprobs (the paper) or estimated from 8 sampled tokens (StableReinforce, as reverse-engineered).

The one-sentence version: when the sampler and trainer disagree, the policy gradient secretly includes an “imitate the sampler” term that compounds every step, and subtracting the average score removes exactly that term while keeping all the real learning signal.

---

## Appendix A: why there is a sampler and a trainer at all

The short answer is speed. **Generating** is sequential, because each token depends on the previous one, so a 1,000-token answer needs 1,000 forward passes in a row. **Scoring** a finished sequence is parallel: all tokens are known, so one forward pass with a causal mask computes $\log p(y_t \mid y_{\lt t})$ at every position, and one backward pass gives all the gradients.

<figure>
<svg viewBox="0 0 760 186" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Sequential generation versus parallel scoring" font-family="ui-serif, Georgia, 'Times New Roman', serif" style="display:block;width:100%;height:auto">
<defs><marker id="arr-6" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#666666"/></marker><marker id="arra-6" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#0b5fff"/></marker></defs>
<text x="10" y="62" fill="#1a1a1a" font-size="13.5" font-weight="600">Generating</text><text x="10" y="79" fill="#666666" font-size="12">sampler: 5 passes in a row</text><text x="252" y="36" fill="#1a1a1a" font-size="13" text-anchor="middle">The</text><rect x="210" y="48" width="84" height="34" rx="4" fill="#fdfdfd" stroke="#666666" stroke-width="1.2"/><text x="252" y="70" fill="#666666" font-size="12" text-anchor="middle">pass 1</text><path d="M296,65 H313" stroke="#666666" stroke-width="1.4" marker-end="url(#arr-6)"/><text x="358" y="36" fill="#1a1a1a" font-size="13" text-anchor="middle">answer</text><rect x="316" y="48" width="84" height="34" rx="4" fill="#fdfdfd" stroke="#666666" stroke-width="1.2"/><text x="358" y="70" fill="#666666" font-size="12" text-anchor="middle">pass 2</text><path d="M402,65 H419" stroke="#666666" stroke-width="1.4" marker-end="url(#arr-6)"/><text x="464" y="36" fill="#1a1a1a" font-size="13" text-anchor="middle">is</text><rect x="422" y="48" width="84" height="34" rx="4" fill="#fdfdfd" stroke="#666666" stroke-width="1.2"/><text x="464" y="70" fill="#666666" font-size="12" text-anchor="middle">pass 3</text><path d="M508,65 H525" stroke="#666666" stroke-width="1.4" marker-end="url(#arr-6)"/><text x="570" y="36" fill="#1a1a1a" font-size="13" text-anchor="middle">42</text><rect x="528" y="48" width="84" height="34" rx="4" fill="#fdfdfd" stroke="#666666" stroke-width="1.2"/><text x="570" y="70" fill="#666666" font-size="12" text-anchor="middle">pass 4</text><path d="M614,65 H631" stroke="#666666" stroke-width="1.4" marker-end="url(#arr-6)"/><text x="676" y="36" fill="#1a1a1a" font-size="13" text-anchor="middle">.</text><rect x="634" y="48" width="84" height="34" rx="4" fill="#fdfdfd" stroke="#666666" stroke-width="1.2"/><text x="676" y="70" fill="#666666" font-size="12" text-anchor="middle">pass 5</text><text x="10" y="152" fill="#1a1a1a" font-size="13.5" font-weight="600">Scoring</text><text x="10" y="169" fill="#666666" font-size="12">trainer: 1 parallel pass</text><text x="252" y="126" fill="#1a1a1a" font-size="13" text-anchor="middle">The</text><text x="358" y="126" fill="#1a1a1a" font-size="13" text-anchor="middle">answer</text><text x="464" y="126" fill="#1a1a1a" font-size="13" text-anchor="middle">is</text><text x="570" y="126" fill="#1a1a1a" font-size="13" text-anchor="middle">42</text><text x="676" y="126" fill="#1a1a1a" font-size="13" text-anchor="middle">.</text><rect x="210" y="138" width="508" height="34" rx="4" fill="#fdfdfd" stroke="#0b5fff" stroke-width="1.2"/><text x="464" y="160" fill="#0b5fff" font-size="12.5" text-anchor="middle">1 forward + 1 backward pass, all positions at once</text>
</svg>
<figcaption>

**Figure 6.** Why the work is split. Generating needs one forward pass per token, each waiting for the last. Scoring the finished sequence is a single parallel pass, which is also where the gradients come from.

</figcaption>
</figure>

| | Sampler (inference engine) | Trainer (training engine) |
|---|---|---|
| Examples | vLLM, SGLang | FSDP, Megatron, JAX trainers |
| Job | write rollouts, token by token | score finished rollouts and compute gradients |
| Built for | throughput: KV cache, continuous batching, fused kernels, often lower precision | backprop: keeps activations, holds optimiser state, shards the model |
| Outputs | tokens, plus the logprobs it used ($q$) | $\log p$ with an autograd graph |

You can't skip the trainer's forward pass and reuse the sampler's numbers. Gradients need the trainer's autograd graph, and the sampler's numbers are slightly wrong anyway, which is the whole problem. Splitting the work buys a large speed-up, because generation usually dominates RL wall-clock time. The price is that the model that wrote the tokens isn't exactly the model being trained. The paper's position is that it's better to keep the speed and correct the drift in the loss.

## Appendix B: why is there a sum over the vocabulary?

A natural objection to everything above: when you actually train, the loss is computed on the one token the sampler picked, A or B, never on the whole vocabulary. So where does a sum over A and B ever happen?

The short answer is that in vanilla training it never does, at least not explicitly. The sum is shorthand for what you get when you add up lots of single-token updates. (One small correction to the setup: RL rollouts are *sampled* with some randomness, not decoded greedily. With greedy decoding the sampler would pick A every time. Sampling is what makes A come up 60% of the time and B 40%.)

### The sum happens by adding up updates

Go back to the two-token model: trainer $p = (0.5, 0.5)$, sampler $q = (0.6, 0.4)$, constant reward $+1$. Each sampled token gives one ordinary single-token update: $s_A = (+0.5, -0.5)$ if A was picked and $s_B = (-0.5, +0.5)$ if B was. Note that both logits move even though only one token was sampled. That's the softmax: raising one token's probability means lowering the others.

A gradient step doesn't use one token, though. It averages over every token in the batch. Suppose this prefix comes up in ten rollouts:

| Draw | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Token | A | A | B | A | B | A | A | B | A | B |
| Push on $z_A$ | +0.5 | +0.5 | −0.5 | +0.5 | −0.5 | +0.5 | +0.5 | −0.5 | +0.5 | −0.5 |

Adding them up gives $6 \times 0.5 - 4 \times 0.5 = +1.0$, or $+0.1$ per token. That is exactly $\sum_v q_v s_v = 0.6 \times 0.5 + 0.4 \times (-0.5)$. Nobody computed a sum over the vocabulary: it came out of adding ten single-token updates, with A appearing more often because the sampler picks it more often. If the trainer had done the sampling you'd expect five of each, and the pushes would cancel to zero.

In real RL the same prefix never repeats, but that doesn't matter. Every position shares the same weights, so the small same-direction pushes from thousands of different positions add up in the same way, which is also why SFT works on data where no prefix is ever seen twice.

### The one place the sum is actually computed

Score centering subtracts $\bar s = (0.1, -0.1)$ from every single-token update. Each update becomes $(+0.4, -0.4)$ when A is picked and $(-0.6, +0.6)$ when B is, and the same ten draws now net to $6 \times 0.4 - 4 \times 0.6 = 0$, as they should. To subtract $\bar s$, though, you have to know it as a number, which means knowing the sampler's probabilities, not just the token it picked. That is the only point where a sum over the vocabulary is evaluated explicitly.

Here is what that looks like at a realistic position. The prefix is <span class="tok">The answer is</span>, the sampler picked <span class="tok picked">42</span>, and the top of the menu has these made-up numbers:

| Token | Sampler $q_v$ | Trainer $p_v$ | $c_v = q_v - \rho\, p_v$ |
|---|---:|---:|---:|
| <span class="tok picked">42</span> (picked) | 0.70 | 0.64 | +0.06 |
| <span class="tok">41</span> | 0.12 | 0.15 | −0.03 |
| <span class="tok">forty</span> | 0.05 | 0.07 | −0.02 |
| <span class="tok">newline</span> | 0.03 | 0.04 | −0.01 |
| rest of vocabulary | 0.10 | 0.10 | tails match, $\rho = 1$ |

GRPO's loss uses one line, $-A \log p(\text{“42”})$. The score-centered loss uses that line plus a few others at the same position:

$$
L = -A\Big(0.94 \log p(\text{“42”}) + 0.03 \log p(\text{“41”}) + 0.02 \log p(\text{“forty”}) + 0.01 \log p(\text{“newline”})\Big).
$$

The picked token gets slightly less credit because the sampler over-picks it, and the under-picked tokens get a little back. If the two engines agreed, every $c_v$ would be zero and this would be the GRPO loss exactly.

### In summary

| Where | What happens to the sum over the vocabulary |
|---|---|
| Vanilla training | Never computed. It appears implicitly when many single-token updates are added up, each token weighted by how often the sampler picks it. |
| The analysis in the paper | Written out explicitly, to describe what that adding-up produces on average. This is how the drift is found. |
| The score-centering correction | Actually computed, because the offset $\bar s$ has to be known before it can be subtracted. |

The computed sum is cheap. The trainer's softmax already sums over all 152K logits to produce $\log p(\text{“42”})$, so reading off a few more $\log p_v$ at the same position costs almost nothing. The sampler only has to log its top-$k$ logprobs, or, in StableReinforce's case, a few sampled token ids.

## Appendix C: how the loss relates to GRPO and other methods

The paper compares every correction on top of one shared objective, REINFORCE with group-centered rewards (their Eq. 13):

$$
J_{\text{base}}(\theta) = \frac{\sum_i A_i \sum_{t=1}^{L_i} \log p_\theta(y_{i,t} \mid y_{i,\lt t})}{\sum_i L_i},
\qquad A_i = R_i - \bar R_{\text{group}} .
$$

Here $i$ indexes rollouts, $L_i$ is a rollout's length, and $\bar R_{\text{group}}$ is the mean reward of the 8 completions for the same prompt. Against that base, each method only changes two things in the per-token loss: a weight $w_t$ on the sampled token, and a centering term $c_v$.

$$
L_t = -A\Big(\mathrm{sg}[w_t]\,\log p(y_t) \;-\; \sum_{v\in H}\mathrm{sg}[c_v]\,\log p_v\Big)
$$

| Method | Weight $w_t$ | Centering $c_v$ | Drift |
|---|---|---|---|
| REINFORCE (the base) | $1$ | $0$ | full |
| Naive IS | $r_t = p(y_t)/q(y_t)$ | $0$ | none, but huge variance |
| TIS | $\min(r_t, 2)$ | $0$ | partial |
| MIS | $r_t$ if $0.5 \le r_t \le 5$, else $0$ | $0$ | partial |
| PPO / GRPO clip | $r_t$, or $0$ where the clip binds | $0$ | partial |
| Score centering | $1$ | $q_v - \rho\, p_v$ | none |
| TIS + score centering | $\min(r_t, 2)$ | $q_v w_v - \alpha\, p_v$ | none |

For the combined row, $w_v = \min(p_v/q_v, 2)$ and $\alpha = \rho \min(1/\rho, 2)$. The centering term is the average of the *weighted* score instead of the plain one.

PPO and GRPO belong in the importance-sampling rows. Their clipped objective contains $r_t \hat A$, and since $\nabla r = r\, \nabla \log p$, its gradient is $\hat A\, r_t\, \nabla \log p(y_t)$: policy gradient with importance weight $r_t$, set to zero wherever the clip binds. That makes them clipped importance sampling, with partial drift. In the paper, PPO and DAPO survived staleness but collapsed under quantisation and weight noise, consistent with a clip designed for ratios that come from policy updates rather than numerical error.

So score centering isn't a new RL algorithm. It is one extra term in the per-token loss of whatever you already run, it costs a sum over a few logprobs at positions you are already training on, and it disappears when the sampler and trainer agree.

---

*The paper and thread are recent. The equations here were transcribed from the arXiv PDF and the X thread, so check them against the originals before relying on exact notation. The toy examples, the “42” menu and the playground are my own illustrations, not from the paper.*
