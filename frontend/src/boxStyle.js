// Box identity is separate from fresh/rotten colors, which remain in the result table.
const BOX_COLORS = ['#2563eb', '#9333ea', '#c2410c', '#087f8c', '#db2777', '#047857', '#4f46e5', '#475569']

export const boxColor = (index) => BOX_COLORS[index % BOX_COLORS.length]
