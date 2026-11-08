window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
};

// Typeset again after every page load, since the site swaps pages without a reload.
document$.subscribe(() => {
  // Skip while the MathJax script is still loading, so one failed call cannot end the subscription.
  if (!window.MathJax || !MathJax.startup || !MathJax.startup.output) {
    return;
  }
  MathJax.startup.output.clearCache();
  MathJax.typesetClear();
  MathJax.texReset();
  MathJax.typesetPromise();
});
