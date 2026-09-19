# Survey Guide — Questions in Order, with Variables

**Survey link (anonymous):** https://bostonu.qualtrics.com/jfe/form/SV_3PemGW4LCPQSLYO
**Tool:** Qualtrics (Boston University) · **ChatGPT interaction:** embedded via G4R (g4r.org)

## What this survey measures, and why it matters

**The question:** When an AI shopping assistant gives you advice, does it matter *who you think the assistant works for*? People increasingly shop with AI assistants, and those assistants can be framed as neutral helpers or as a brand's own sales tool. The same recommendation might be trusted and followed if it seems to come from an independent third party, but discounted if it seems to come from the company trying to sell you something. This study tests whether that framing alone changes how much people trust and rely on the assistant.

**How it isolates the effect:** Every respondent talks to the *same* assistant, which gives the *same* kind of objective, balanced advice (same model, same neutral instructions). Respondents are randomly split into two groups, and the **only** difference is one line on the screen:

- one group is told it is "an independent, third-party shopping assistant,"
- the other is told it is "the brand's official shopping assistant."

Because nothing else differs, any gap between the two groups in trust, confidence, and willingness to follow the advice can be attributed to the **identity framing** itself, not to the advice.

**Why it matters:** It speaks to a live issue in marketing and consumer research — how disclosure and perceived allegiance of AI agents shape consumer trust and adoption. For firms, it informs how to position branded AI assistants; for policy, it informs debates about whether AI assistants should disclose whose interests they serve. The design connects to consumer-search work (AI assistants as tools that lower search cost) and to current research on AI agents in marketing.

---

Read top to bottom. Each item shows: **what the question asks**, the **variable name**, and **what it measures**.

**Intro — Consent (Descriptive Text)**
Anonymity and voluntary-participation statement. No variable.

**Q1 — How often do you buy clothing online?**
Variable: `shop_freq` · Measures how often they shop online (background covariate).

**Q2 — "Online AI assistants help me shop well." (agree 1–5)**
Variable: `prior_trust` · Their general trust in AI assistants before the task.

**Q3 — Four statements (agree 1–5):**
- read reviews before buying → `habit_reviews`
- compare items before deciding → `habit_compare`
- "please select Strongly agree" → `attn_check` (attention check, catches careless responses)
- hard to choose when many options → `choice_overload`

**Q4 — Chat with the assistant (ChatGPT, embedded)**
The treatment. The label "independent third-party" vs "brand's official" is randomly assigned and shown here. The assistant's advice is identical across conditions; only the label differs.
Variables: `assistant_label` (which condition), `chat_transcript` / `g4r_pid` (the conversation).

**Q5 — Timing (hidden)**
Variable: `time_on_task` · Time spent on the chat page (engagement / data quality). Not shown to respondents.

**Q6 — Based on the conversation, what will you do? (follow / partly / own judgment / keep looking)**
Variable: `followed_AI` · Whether they adopt the assistant's advice. **Primary outcome (behavioral).**

**Q7 — In your own words, how did you decide?**
Variable: `reasoning_text` · Open-ended reasoning, before any rating questions.

**Q8 — How confident are you this is a good choice? (0–10)**
Variable: `confidence` · Confidence in the decision.

**Q9 — Rank what matters most when shopping online (Price / Style / Reviews / Brand)**
Variable: shopping priorities · General preferences (covariate).

**Q10 — Split 10 points: assistant / own knowledge / price / other**
Variables: `rely_AI`, `rely_prior`, `rely_price`, `rely_other` · How much each input shaped the choice.

**Q11 — Side by Side, 4 attributes × 2 columns:**
- Column 1 "Agreement" (1–5) → `perc_helpful`, `perc_trust`, `perc_balanced`, `perc_bestforme` · how they rate the assistant.
- Column 2 "Influence on your choice" (1–5) → `infl_helpful`, `infl_trust`, `infl_balanced`, `infl_bestforme` · how much each moved their choice. **`infl_bestforme` is the primary attitudinal outcome.**

**Q12 — How likely to use such an assistant again? (NPS 0–10)**
Variable: `reuse_nps` · Repeat-use intention.

**Q13 — Who did the assistant represent? (independent / brand / not sure)**
Variable: `mc_identity` · Manipulation check, placed last so it doesn't reveal the study's purpose.

---

**The core comparison:** respondents are split 50/50 into "independent third-party" vs "brand's official." Since only the label differs, any gap between the two groups in `followed_AI`, `infl_bestforme`, `confidence`, the `perc_*` ratings, and `reuse_nps` reflects the effect of the assistant's **identity framing**. The pre-task items (`shop_freq`, `prior_trust`, habits) are used to check the groups are balanced and to explore who is more affected.
