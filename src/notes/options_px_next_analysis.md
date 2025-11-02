#### TO DO LIST

## Exploratory analysis

# Percentage of untargeted runs that lead to change of phase

# Percentage of run in behind untargeted runs that push the defensive line

# Video frame of an untargeted off ball run that made a space that someone occupied and in the end it lead to a goal.
This is the introduction video. To show the impact that that untargeted run had on the shot/goal.


#### NOTES

<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# So now for example I have that there is a big number of untargeted runs. I also identify how many there are by subtype and by team. I also identify if there is a change of phase after these untargeted runs.

I am going to also analyse the avg expected pass completion of these runs and the average likelihood of doing the pass because that is going to give me if these untargeted runs were really dificult to be served and if the probability of being targeted is low. If this is true, then it would be that there are a lot of runs that are made just to create something, not to receive the ball right?

What else can I do to show the importance of analysing these untargeted runs?

You’re on exactly the right path: showing the abundance, diversity, team/tactical patterns, and high difficulty of untargeted runs builds a compelling foundation. To further highlight their importance, here are **concrete, high-impact analyses** you can add—all feasible with your SkillCorner data:

***

### 1. **Teammate Opportunism and Space Creation**

- **Spatial Occupation Analysis:** For each untargeted run, check if a teammate *subsequently* occupies the vacated space or attacks the zone opened by the run (within 2–5 seconds).
    - Results: If this occurs frequently, you prove untargeted runs actively facilitate space for teammates, supporting the idea of "movement for others" as tactical intent.

***

### 2. **Impact on Defensive Structure**

- **Defensive Disruption Index:** Using tracking data, quantify change in opponent defensive spacing or line compactness immediately after untargeted runs (especially high-difficulty ones).
    - Results: If untargeted runs regularly force defensive shape adjustments, it supports their disruptive, strategic value.

***

### 3. **Sequence/Chain Analysis**

- **Trigger for Attacking Sequences:** Identify if untargeted runs precede critical ball events (progressive pass, shot, box entry, line break) more than would be expected by chance.
    - Results: If attacking actions occur at higher rates immediately after untargeted runs, this further demonstrates their impact on generating chances.

***

### 4. **Comparison with Targeted Runs and Random Timings**

- **Relative Effectiveness:** Compare all above metrics (space creation for others, defensive disruption, sequence triggering) for untargeted runs vs. targeted runs vs. randomly sampled possession moments of similar length.
    - Results: If untargeted runs show greater or unique effects, you argue they're deliberate team actions, not just "ignored potential passes."

***

### 5. **Visual Evidence**

- **Heatmaps, Timelines, and Sequence Diagrams:**
    - Show spatial patterns (e.g., untargeted runs overload certain key areas) and timeline overlays of sequences—illustrate moments where untargeted runs clearly manipulate defenders and create space for others.

***

### 6. **Team Tactical Profiles**

- **Team Comparison:** Profile which teams generate the most high-difficulty untargeted runs and link this to their possession style, goal creation, or overall attacking success.
    - Results: Some teams may rely more heavily on collective untargeted movement, tying your findings to real tactical strategies.

***

### 7. **Phase-of-Play Synergy**

- **Phase Analysis:** Show that untargeted runs—especially difficult/unlikely ones—are most common or most effective in specific game phases (build-up, transition, set pieces), indicating designed tactical roles.

***

### 8. **Link to Match Outcomes**

- If you have enough matches, explore whether teams with a higher rate of untargeted, high-difficulty disruptive runs correlate with more chances created or higher expected goals—strengthening the real-world relevance of your findings.

***

### **How to Frame These Analyses**

- "Untargeted runs are not simply ‘missed’ passing options, but serve as tactical decoys, disruptors, and space creators. Their prevalence, difficulty, and indirect effects shape the attacking environment and team strategy in measurable ways."

***

**Summary:**
By adding these targeted analyses—especially around space creation for teammates, defensive disruption, chain/sequence impact, and team/phase profiling—you’ll provide a multi-dimensional, evidence-driven argument for why untargeted off-ball runs should be a major focus in football analytics and coaching. This will strongly differentiate your work from simple descriptive analytics, making your project both innovative and deeply relevant.






## Extra

To see if a run moves defense.I was thinking also get the number of players around the start of the off ball run and get the minimum number of players in that same area during the run.  ALSO, get the players that are around and see if they move in the same direction as the runner. But this would only apply maybe to specific subtypes like comming short, run in behind or something like that. OTHERS subtypes like dropping how are made to get further from the defenders so I can also track the defenders around start frame and the defenders around end frame.

Also, to see if a run moves defense im sure there might be a model or a statistic model where I can say the % of free space or unoccupied space there is in an area. This measure would be useful to see how space creation the player do.


#### For later analysis:
- Space creation
- xthreat increase
- openning passing lines
- line breaks
- passing options

#### Chat ideas for later analysis
A. Synchronization with teammates

Measure how many teammates move in the same direction (±20° angle) as the runner at run start.

Plot a boxplot or violin per subtype.

➡️ Shows how collective or isolated untargeted runs are — e.g. overlaps = coordinated, behind runs = more individual.

✅ B. Spacing impact

Compute average distance between the runner and nearest defender at start vs. end of the run.

Show if untargeted runs pull defenders away (distance increases) or pin them back (distance decreases).

Even if aggregate, it visually proves space creation.

✅ C. Positional chain reaction

For each untargeted run, measure how much the teammate in possession’s space (Voronoi area or average nearest defender distance) changes.

It’s a proxy of how the run frees the ball carrier — without modeling full XThreat yet.


## as an extra:
- Group types of runs like coming short and running behind or shit like this
- Kmeans and identify types of players based on the metrics.

