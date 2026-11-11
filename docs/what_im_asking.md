# What I am asking

After a big down day on MASI do more big down days show up soon after?

Can a Hawkes model describe that better than saying bad days arrive at random?

GARCH still handles the usual day to day volatility.

## Why both

GARCH is about how wild returns are.

Hawkes is about when the rare worst days show up.

Same market. Different angle. I keep them on the same sample so the comparison stays fair.

## Event rule for this first version

Daily log returns. Mark the worst about 5 percent of days. Those dates feed the Hawkes model.

## Later

I may check if busy or code switched comment days from LixusFinSent line up with those rough market days. Not built out yet.
