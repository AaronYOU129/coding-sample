# AI Prompt Log — Tasks 1–3

## Note on reconstruction

This document records the substantive prompts and decisions that shaped the
solutions. The exact wording of some early prompts was not retained, so those
entries are described as reconstructed summaries rather than verbatim
quotations. Short procedural messages such as “continue,” navigation requests,
and approval clicks are not reproduced.

## Workflow instructions given to AI

Before beginning the formal stages for each task:

1. I first gave AI my general answer and proposed approach.
2. I asked AI to supplement my answer with any missing methods, assumptions,
   or implementation details.
3. When AI introduced a method or detail that I did not understand or remember,
   I asked AI to explain it before proceeding.

I then instructed AI to follow the four stages below.

### Stage 1 — Requirements Definition

Define each task using four elements: **goal, input, output, and steps**. AI
must proactively ask questions rather than guess my intent. Anything unclear
has to be raised with me before proceeding, not assumed.

### Stage 2 — Design (High-Level and Detailed)

Again specify **goal, input, output, and steps**, with AI asking about anything
unclear instead of guessing. Two principles guide the design:

- Modules stay independent and independently testable.
- Key information is stored in files, so each AI session does less and the
  context stays short. This reduces token usage and limits hallucination.

### Stage 3 — Task Decomposition

Break each module into its smallest tasks. In the steps, each module maps to
its own `module-name.md` file containing the minimal tasks for vibe coding.
Progress is tracked with checklists:

- Each `module-name.md` uses a checklist for its subtasks.
- A central `progress.md` uses a checklist to track whether each module is
  complete.

### Stage 4 — Implementation

Drive implementation from `progress.md` using a main-agent and sub-agent
structure:

- Sub-agents implement individual modules, so each agent’s context stays short.
- The main agent tracks overall progress across modules.

Quality gates on all code:

- For the import part, use Pytest coverage.
- Code must pass Mypy for type checking and Ruff for linting.
- When generating prompts, anything unclear must be raised with me rather than
  assumed.

---

# Task 1 — Restaurant Price Index

## My initial approach

**Reconstructed prompt summary:** “Separate each restaurant’s prices into a
product-specific size and a price level that changes over time. Check my
approach, add anything missing, and explain the method before coding.”

I began with the idea of separating product differences from general price
changes and normalizing product sizes relative to entrees.

## How AI helped

AI proposed a restaurant-specific two-way fixed-effects regression of log
price. It explained that product and price-index values are latent quantities
recovered from product and date effects, and identified the need to normalize
disconnected product-date components separately.

My main follow-up prompts asked why logs and dummy variables were used, why
normalization was necessary, and how connectivity affected identification.

## How I verified it

I checked recovery on simulated data with known sizes and price indices and
verified that estimated size multiplied by the estimated index reproduced
fitted prices. I also checked missing fields, product-name normalization,
zero-quantity rows, connectivity, category-level sizes, and index paths.

---

# Task 2 — Normal Mixture Maximum Likelihood

## My initial approach

**Reconstructed prompt summary:** “Write the two-component normal-mixture
likelihood and estimate it by maximum likelihood. Check my approach, add the
missing numerical details, and explain them before implementation.”

I began with the required K=2 likelihood and planned to compare the fitted
distribution with the observed data.

## How AI helped

AI added log-sum-exp evaluation, constrained-parameter transformations, a
standard-deviation floor, multiple starting values, and a label-ordering rule.
Because the EDA showed three modes, it suggested keeping K=2 as the required
model and using K=3 only as a diagnostic comparison.

My follow-up prompts asked about the negative log-likelihood, parameter
transformations, degenerate solutions, local optima, label switching, and the
reason for using multiple starts.

## How I verified it

I tested the likelihood and transformations, checked parameter recovery on
simulated data, and inspected all 25 optimizer starts. Density and QQ plots,
AIC, and BIC show that K=2 misses part of the three-mode shape, while K=3 gives
a closer descriptive fit.

---

# Task 3 — Do Readers’ Clicks Affect Story Length?

## My initial approach

**Reconstructed prompt summary:** “Merge the three datasets by date and use
rainfall and electricity shortage as instruments for views because a direct
regression may have omitted-variable bias and reverse causality. Check the
strategy, add missing details, and explain the required tests.”

I began by recognizing that story importance may affect both clicks and
editorial investment, while longer coverage may itself generate more clicks.

## How AI helped

AI added the outcome definitions, log transformations, controls, fixed
effects, date-clustered standard errors, first-stage testing, and
Anderson–Rubin inference. It also explained that weak relevance and a failure
of the exclusion restriction are different problems.

My follow-up prompts asked how the two stages work, why instruments must predict
views, what a weak instrument means, why errors are clustered by date, and why
Anderson–Rubin inference is needed.

## How I verified it

I checked date parsing, merge counts, outcome construction, and recovery of a
known effect in simulated endogenous data. In the observed sample, the joint
first-stage statistic is 0.59, the AR tests do not reject zero, and the
confidence sets are unbounded. The conclusion is therefore that these
instruments do not identify the causal effect precisely—not that clicks have
no effect.

---

## Main lessons

- Begin with my own preliminary answer before asking AI for additions.
- Ask AI to explain methods that I cannot independently describe.
- Verify AI suggestions using theory, simulated data, and observed diagnostics.
- Report weak or inconclusive evidence honestly.
