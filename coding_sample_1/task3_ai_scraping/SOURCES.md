# Identifying the Top 10 Shopping Prompts and Assessing Source Quality

No AI provider publicly releases verbatim logs of the shopping prompts users
actually submit. Google's own AI Mode report states it has no exact counts for
any term or topic, reporting only each topic's share of searches, and that its AI
Mode Trends data is not publicly available. Because no frequency-ranked list of
real prompts can be computed from public data, the top 10 here is constructed from
the best available evidence rather than measured, and the task becomes arguing for
the quality of that evidence.

I draw on two complementary sources. Google's *How people are using AI Mode in the
U.S.* (May 2025 to April 2026) is first-party and query-derived. It ranks the
shopping categories users most often ask about (Electronics, Books/Movies/Music,
Apparel, Health/Beauty, Automotive, and so on) and the retail attributes they care
about (Price, Location, Color, Brand). Its strength is authenticity. Its limits
are that it is a Google marketing document, reports relative share rather than
absolute volume, is US only, and is not reproducible because the underlying data
is not public. It also covers only one of the three platforms named in the task.

Chen, Wang, Chen & Koudas (2025), *Generative Engine Optimization*
(arXiv:2509.08919), derives a taxonomy of shopping intents (recommendation, price
comparison, review analysis, negotiation, gifting) across ChatGPT, Perplexity,
Gemini, and Claude. It is academic, method-transparent, reproducible, and covers
the platforms Google misses. Its limits are that it is a Reddit proxy rather than
real query logs, samples AI-enthusiast communities (biasing toward power-user
behaviors), and is a classification with no frequency ranking.

The two are paired so each covers the other's blind spot. Google gives what people
buy (real category demand); Chen et al. give how they ask (intent types) plus the
missing platform coverage. Neither provides absolute counts.

The top 10 was built, not calculated, with no weighting or scoring. Categories
were taken from Google's real ranking. Intents were taken from Chen et al.'s
dominant types, used only for pairing since that taxonomy has no order. The two
were paired into ten prompts chosen to reliably elicit brands and product features
for Part 2, dropping Books/Movies/Music and Home Improvement as too sparse on
those. Ordering follows category rank, with same-category prompts tied. The
numeric values in prompts (prices, ages) are illustrative placeholders.

The result should be read as a defensible, source-traceable construction, with
each prompt's category grounded in Google's ranking and its intent in a
reproducible taxonomy, not as a frequency ranking of what users most often type.
Stating this boundary, rather than implying a precision the data cannot support,
is itself part of assessing source quality honestly.

## Prompt provenance

| # | Prompt | Category (Google rank) | Intent | Placeholder |
|---|--------|------------------------|--------|-------------|
| 1 | Noise-cancelling headphones | Electronics (1, tie) | Recommendation | $300 |
| 2 | Laptop for video editing | Electronics (1, tie) | Price comparison | $1500 |
| 3 | Smartphone best camera | Electronics (1, tie) | Price comparison | 2026 |
| 4 | Waterproof hiking boots | Apparel (3) | Recommendation | $200 |
| 5 | Vitamin C serums | Health/Beauty (4) | Review analysis | sensitive skin |
| 6 | RAV4 vs CR-V | Automotive (5) | Decision | family |
| 7 | Robot vacuum for pet hair | Home/Garden (6) | Recommendation | $400 |
| 8 | Espresso machine | Grocery/Kitchen (7) | Recommendation | $500 |
| 9 | Building gift for 7-year-old | Toys/Games (9) | Gifting | age 7, $50 |
| 10 | Beginner road bike | Sports/Outdoors (10) | Recommendation | $1000 |
