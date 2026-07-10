# Actual NIBP Display-Event Algorithm

Raw NIBP monitor records are grouped by case, filtered to physiologic MAP values, sorted by monitor time, and collapsed into display-level events. The primary rule keeps the first value, keeps value changes, collapses repeated identical values held for less than 90 seconds, and requires at least 120 seconds before a repeated same-value event is treated as a new display event. Sensitivity grids report strict, primary, lenient, and intermediate settings.
