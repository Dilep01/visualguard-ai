# Demo walkthrough

1. Run `scripts/start.ps1`, then open the dashboard on localhost port 8501.
2. Select **Run synthetic demo**. Explain that the clip and detector are test fixtures.
3. Refresh the job and inspect the annotated video. Three people appear; one has a possible missing helmet until four seconds, one has a helmet, and one has an unknown head state.
4. Inspect the timeline and evidence JSON. At 4 sampled FPS, the eight-second scene should yield 32 samples and one helmet candidate supported through 3.75 seconds.
5. Open Event log, inspect the rule and evidence, and record a review decision.
6. Open Analytics. One demo clip represents approximately 0.4 observed person-minutes and one candidate. This is synthetic arithmetic, not a workplace compliance result.
7. Switch synthetic data off to show that the real-data analytics are empty.
8. Open Model & data to inspect the real checkpoint's metrics. Explain its zero uncovered-head recall at confidence 0.4 and the remaining independent-scene/video evaluation work.
9. Return to Analyze & review, expand the real upload section, and upload `data/demo/real-ppe-sample.jpg`. Inspect the actual model's boxes. This validation image demonstrates integration; it is not a new accuracy test.

Stop services with `scripts/start.ps1 -Stop`. Generated media, logs, and database remain inside `runtime/`.
