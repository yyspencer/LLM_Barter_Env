Here stores the analysis code, which are mostly ipynb.
## Analysis
Whenever write a new script, remember to add the output directories to gitignore. So that the results don't get uploaded. 

## What do these graphs mean?
1. **Ratings (inventory-independent)** - every round, an agent is asked to rate each of its goods 1 to 10 based on how much they want it, *regardless of their current inventory*. These are averaged out every run, binned by condition.
2. **Ideal Bundle** - every round, an agent is asked if it could have any inventory of items (that sum to 4), what would it ideally want?
3. **Net Acquisition** - if an agent gets 3 Bs but gives away 1 B, they have a net acquisition of 2 Bs at that time. Tracks the change in inventory over time.
4. **Trade Behavior** - (a) acceptance probability - if someone offers item X, how often does the agent actually accept it? (b) sample sizes - how many times was each item offered to the agent (multiple items count as only a single instance of "offering") (c) willingness to pay - on average, how much does this agent give away (in accepted trades) as a ratio to the goods they receive? (d) goods sought in outgoing proposals - what goods does the agent most want and act upon? (e) acceptance over rounds - what is the general rate they accept trade offers over time? (expected to decrease near the end)
5. **Condition Compare** - an alternative view of the first graph, binned by good
6. **Ratings (inventory-dependent)** - the first graph but with results from a probe that asks the agents to rate goods *given their current inventory*
