# Guidelines for Neuroscientific Analysis & Plotting

This document outlines the conventions, standards, and best practices for developing, refactoring, and maintaining code within the **ocapi** workspace. Follow these rules to ensure scientifically rigorous analyses, publication-ready plots, and high-performance real-time visualizations.

---

## 1. Plotting & Visualization Conventions

### Axis Labels & Units
- **Always specify units** for all physical or arbitrary measurements.
  - Time: `Seconds (s)` or `Milliseconds (ms)`.
  - EEG Amplitude: `Amplitude (μV)`.
  - Power Spectral Density: `PSD (μV²/Hz)` or `Power (dB)`.
  - Head/Eye Angles: `Degrees (°)` or `Radians`.
  - Eye Positions/Displacements: `Pixels` or `Millimeters (mm)`.
- Use professional title-casing and format labels clearly (e.g., `Time (ms)` instead of `time_ms`).

### Color & Styling (Publication Quality)
- Use **colorblind-friendly palettes** (e.g., Seaborn `colorblind` palette, or Matplotlib colormaps like `viridis`, `magma`, `plasma`, `inferno`).
- Avoid default, high-saturation primary colors (pure red `#FF0000`, pure green `#00FF00`, etc.) in final plots.
- Keep grid lines subtle: use light grey, thin lines, or dashed styles (`grid(True, linestyle='--', alpha=0.5)`).
- When plotting averages (e.g., ERPs, gaze trajectories), **always show variance/uncertainty** using semi-transparent shaded error bands representing the Standard Error of the Mean (SEM) or Standard Deviation (SD) rather than just single lines.

## 3. Signal Processing & Analysis Rigor

### Synchronization & Timing
- Pay close attention to clock drifts and timing jitter when combining EEG, video, and eye-tracking streams.
- Document any alignment markers, trigger offsets, or interpolation methods (e.g., linear vs. spline interpolation for upsampling/downsampling to a common timeline).

---

## 4. Documentation & Statistical Reporting

- **Report Details**: When running statistical tests (e.g., clustering, t-tests, ANOVA), output the test statistics, degrees of freedom, $p$-values, and effect sizes (e.g., Cohen's $d$, Eta-squared).
- **Assumptions**: Verify and report whether assumptions (such as normality or sphericity) were tested.
