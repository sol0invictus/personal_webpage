# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project intent

Personal website for Sunny Guha, an ML researcher. Three core surfaces:

1. **Landing / about** (`/`) — minimal, text-forward.
2. **Blog** (`/blog`) — Markdown posts with images and LaTeX math.
3. **Projects** (`/projects`) and **Publications** (`/publications`) — listings driven from typed data files, not handwritten markup.

The aesthetic target is "simplistic": prefer plain typography and whitespace over visual chrome. Resist pulling in UI frameworks, CMS layers, or analytics until there's a concrete need.

## Stack

- **Astro 4** in `output: "static"` mode (default). Builds to `dist/`.
- **Markdown pipeline:** `remark-math` + `rehype-katex` → math is rendered at build time. The KaTeX stylesheet is imported once from `src/styles/global.css`.
- **Syntax highlighting:** Shiki (`github-light`) — built into Astro's Markdown.
- **TypeScript:** `astro/tsconfigs/strict`.
- No JS framework integrations (no React/Vue/Svelte). Everything is `.astro`.

## Commands

```bash
npm install          # install deps
npm run dev          # dev server at localhost:4321 with HMR
npm run build        # static build to ./dist
npm run preview      # serve ./dist locally to verify the build
```

There is no test suite or linter configured. Don't add one without asking.

## Architecture

### Pages

- `src/pages/index.astro` — about / landing.
- `src/pages/blog/index.astro` — chronological post list (newest first, drafts excluded).
- `src/pages/blog/[...slug].astro` — dynamic route for one post; uses `getStaticPaths` over the `blog` content collection.
- `src/pages/projects.astro` and `src/pages/publications.astro` — read from typed arrays in `src/data/`.

All pages wrap content in `src/layouts/Base.astro`, which owns `<html>`, the nav, the footer, and the global CSS import. The nav highlights the active section using `Astro.url.pathname`.

### Blog content collection

Posts live in `src/content/blog/*.md`. Schema is enforced by `src/content/config.ts`:

```ts
{ title, description?, date, draft = false, tags = [] }
```

- The blog index filters out `draft: true` posts.
- The slug is the filename (Astro derives it). The post's URL is `/blog/<slug>`.
- **Images:** co-locate next to the post and reference with relative paths (`./figure.png`). Astro resolves these through Vite — do **not** dump assets in a global `public/images/` folder unless they are genuinely site-wide (favicon, OG image).
- **Math:** inline with `$...$`, display with `$$...$$`. Both go through `remark-math` → `rehype-katex` at build time, so no math runtime ships to the client.

### Projects & publications

`src/data/projects.ts` and `src/data/publications.ts` export typed arrays (`Project[]`, `Publication[]`). To add an entry, edit the array — do not add a new component or markup file. Publications are grouped by year in the page; projects are sorted descending by year.

If publications grow large or need richer content (abstracts, BibTeX), consider promoting them to a content collection — but only when the flat array starts to hurt.

### Styles

One global stylesheet at `src/styles/global.css`, imported by `Base.astro`. CSS custom properties at the top (`--fg`, `--bg`, `--accent`, `--max-width`, etc.) are the knobs to tweak. The KaTeX CSS is imported at the top of this file via `@import "katex/dist/katex.min.css"`.

There is no CSS framework (no Tailwind). If a component needs one-off styling, prefer a `<style>` block scoped to that `.astro` file over expanding `global.css`.

## Conventions

- Keep posts and entries as data; do not introduce per-post components.
- Don't add client-side JavaScript unless a feature genuinely needs it — Astro ships zero JS by default and that's the point.
- When adding a new top-level page, also add it to the nav in `src/layouts/Base.astro`.
