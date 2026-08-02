# Portfolio site

A single-file, no-build static site (`index.html`) designed for GitHub Pages.

## What to replace

| Where | What to change |
|---|---|
| `<title>` and nav `.brand` | ✅ Already set to Parisa Zeinaliashtiyani |
| Hero heading + lede | Your one-line research pitch |
| `.graph-box` SVG node `data-def` text | Already matches your thesis's 6 causal factors — edit wording if you refine definitions |
| ASR chart `data-w` values | Already filled with your Table 3 numbers — update if results change |
| Repo cards | Real GitHub URLs once you've cleaned up the notebook into modular repos |
| `assets/thesis.pdf`, `assets/cv.pdf` | Add these two files into an `assets/` folder next to `index.html` |
| Contact links | Your email, Scholar, GitHub, LinkedIn |

## Deploy to GitHub Pages

1. Create a new repo named `yourusername.github.io` (or any repo name if you're okay with a `/reponame` URL suffix).
2. Put `index.html` (and an `assets/` folder with your PDFs) at the repo root.
3. Push to GitHub.
4. In the repo, go to **Settings → Pages**, set source to the `main` branch, root folder.
5. Your site is live at `https://yourusername.github.io` (or `https://yourusername.github.io/reponame`) within a minute or two.

No Jekyll, no build step, no dependencies — it's plain HTML/CSS/JS, so it will always render exactly as previewed.

## Local preview

Just open `index.html` directly in a browser, or run a tiny local server:

```bash
python3 -m http.server 8000
```

then visit `http://localhost:8000`.
