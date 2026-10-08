# Working notes for Claude

## When the user reports a problem

Don't do a heavy verification ritual in response to a bug report — don't go
hunting for exact crop coordinates, running pixel-level diagnostics, building
diff visualizations, or re-running full regression suites with printed
numbers for every prior fix before and after. The user has said directly
that this doesn't help them or the work. Investigate enough to understand
the real cause, fix it, do a basic sanity check, and move on.
