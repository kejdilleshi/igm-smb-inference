# `old_MK/` — reconstructed source of `main_tc_MK.pdf`

The source of the version circulated as `main_tc_MK.pdf` (repo root, 12 Aug 2026)
was not kept, only the PDF. These files are a **reconstruction** of that source:
the PDF text was extracted and re-marked-up against the current file layout
(`abstract`, `introduction`, `Methodology` + `Data`, `Results`, `Discussion`,
`Conclusions`, `appendix`), so that `latexdiff` can compare it with the current
manuscript.

Fidelity check: recompiling `old_MK/main_tc.tex` reproduces the body text of
`main_tc_MK.pdf` at 95.4 % character-level similarity; every remaining difference
is float placement (a caption landing on a different page), not wording.

Deliberate deviations from the original PDF, made to keep the diff free of
spurious markup:

* hard-coded cross-references in the old text (`Section 4`, `Sect. 2.4`) were
  replaced by the current `\ref{...}` labels, so the diff does not flag
  `Section` -> `Sect.`;
* the three figures of the deleted GPR appendix are referred to as `Fig. A1`,
  `Fig. A2`, `Fig. A3` literally, since `latexdiff` drops deleted floats (and
  with them their `\label`s);
* the *figure images* are the current ones throughout — only text and captions
  are diffed. Where a figure was regenerated with new numbers (e.g.
  `fig_continuity_3panel.pdf`), the diff shows the caption change but the panel
  itself is the new one.

## Regenerating the diff

From `SMB_inference_Abstract/`:

```sh
latexdiff --flatten --type=UNDERLINE old_MK/main_tc.tex main_tc.tex > main_tc_diff.tex
pdflatex main_tc_diff && bibtex main_tc_diff && pdflatex main_tc_diff && pdflatex main_tc_diff
```

Delete `main_tc.bbl` first, or `latexdiff` inlines the new bibliography and marks
the whole reference list as added. `latexdiff` ships with TeX Live
(`tlmgr install latexdiff`, or `apt install latexdiff`).
