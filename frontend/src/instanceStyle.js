// Instance colors are separate from fresh/rotten colors in the result table.
const INSTANCE_COLORS = ['#2563eb', '#9333ea', '#c2410c', '#087f8c', '#db2777', '#047857', '#4f46e5', '#475569']

export const instanceColor = (index) => INSTANCE_COLORS[index % INSTANCE_COLORS.length]
