"""
Purpose:    The 10 shopping prompts, frozen in their exact assignment order, as
            the single source of truth for the whole experiment. They live in
            their own module so the collection loop, filenames, and stats all key
            off the same 1-based index and nothing can reorder them.
Inputs:     None.
Outputs:    None.
Key Steps:  Define PROMPTS; expose numbered_prompts() for 1-based iteration.
How to Run: Imported by collect.py; not run directly.
"""

PROMPTS = [
    "Recommend the best noise-cancelling headphones under $300. Compare at least 3 options on battery life, comfort, and sound quality.",
    "What's the best laptop for video editing under $1500 right now? List specific models with their specs.",
    "Which flagship smartphone has the best camera in 2026? Compare the top 3 with their camera specs.",
    "Find durable waterproof hiking boots for wide feet under $200, with a few specific recommendations.",
    "What are the top-rated vitamin C serums for sensitive skin? Compare ingredients and give specific products.",
    "I'm comparing the Toyota RAV4 and Honda CR-V for a family. Compare them on safety, fuel economy, and cargo space.",
    "Recommend a robot vacuum good for pet hair under $400, with specific models and key features.",
    "What's the best espresso machine for beginners under $500? Compare a few models on ease of use and quality.",
    "Best gift for a 7-year-old who loves building things, under $50. Suggest a few specific products.",
    "Recommend a beginner road bike under $1000, with specific models and what makes each good.",
]


def numbered_prompts():
    """Yield (prompt_index, prompt_text) with a 1-based index, matching the
    assignment's 1-10 numbering used in filenames and stats."""
    for offset, text in enumerate(PROMPTS):
        yield offset + 1, text
