# Coding Principles

## Script Header

Every script must begin with a header comment in this format:

```
Purpose:    What the script does and why the approach was chosen.
Inputs:     Exact file paths and descriptions of each input.
Outputs:    Generated files and their locations.
Key Steps:  High-level logical workflow (not repeating function names).
How to Run: Command to execute the script.
```

The Purpose field must explain **why** key methodological choices were made, not just restate what the code does. A reader should understand the script within 30 seconds by reading the header alone.

## Code Organization

- Input and output paths must be explicit and consistent across scripts.
- Scripts should reflect the research pipeline: Data → Processing → Estimation → Output.
- Extract shared logic into reusable modules; avoid code duplication.
- Keep functions small with one clear responsibility. High-level functions describe workflow; low-level functions handle details.
- Separate loading, cleaning, computation, plotting, and exporting when they are meaningfully distinct.

## Style

- Use early returns to reduce nesting; keep the "happy path" visually obvious.
- Prefer clear, specific names over short abbreviations. Use consistent terminology across the project.
- Comments explain **why**, not **what**. Do not compensate for poor naming with comments.
- Do not over-engineer short, linear code. Refactor only when it improves clarity or responsibility boundaries.