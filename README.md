# Mobile Data Usage Distribution Analysis

A statistical analysis of mobile (SIM) data usage across a device fleet, built to answer a deceptively simple question: **what does "typical" data usage actually look like?**

## The Problem

A simple average suggested one story; the underlying distribution told another. Usage was extremely **heavy-tailed** — so the arithmetic mean represented almost no real device on the fleet.

## The Approach

- Explored the distribution on both linear and log scales
- Recognised **two distinct populations** rather than one
- Fit a **two-component log-normal mixture model** to separate "light" and "heavy" users
- Quantified how many devices sit near a critical data-pool threshold

## Key Insight

The pool limit fell in the **valley between the two populations** — so risk was driven almost entirely by how many devices landed in the heavy cluster, not by the average. This directly informed capacity and cost decisions.

## Tech Stack

`Python` · `pandas` · `NumPy` · `Matplotlib` · `Statistical Modelling`

---

> **Note:** Portfolio summary — the source data is confidential. Any figures referenced are illustrative of the method, not real values.
