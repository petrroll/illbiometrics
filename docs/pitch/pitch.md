---
marp: true
---

# Illbiometrics - pitch deck

Petr Houška, 06/12/2025

---

## Highest quality medium/long term biometrics data correlatable with events s.a. treatment attempts and infections

---

# Current treatment data:
- Lump everyone together & ask improve/not: [Patient-Reported Treatment Outcomes in ME/CFS and Long COVID](https://www.medrxiv.org/content/10.1101/2024.11.27.24317656v1.full)
- Series cases / anecdotal data
- Manual input. s.a. current Amatica
- Manual input of symptoms + automatic correlation with meds: [Guava](https://guavahealth.com/)
- Manual input of symptom severity & correlate yourself: [Visible](https://www.makevisible.com/)

---

# Gaps:
- Patient reported outcomes:
  - No stratification of groups
  - No correlation to other signals
  - Human memory is very unreliable
  - Only signal dimension: helped or not 
- Series cases / anecdotal
  - Heavy sample bias
- Manual input: Amatica, Guava, Visible
  - No aggregation
  - No sharing
  - Manual association of symptoms severity to treatment

---

## Amatica already has data that allows for deep stratification and groups. 

## Let's correlate it with disease progression and events.

---

# Goals:
- Link wearable biometrics devices and extract medium/long term statistics 
  - Garmin, Oura, ?Apple Watch, ?Loop, ?Visible
  - Common stats, s.a. HRV (normalize), P20/50/80 or wake HR, number of wakeups, length of sleep
- Use stats to stratify patients in groups: stats + biomarkers + questionnaire
- Augument event input(*)  with changes in ^^ medium/long term stats
  - Check the aggregates for month before treatment and month after
  - Easier remembering whether treatment helped
  - Additional per treatment/event signal


- (*) Extend treatments with infections & high stress events

---

# Non-goals
- Day to day pacing helper - we're not Visible/Guava
- Day to day symptoms tracking - we're not Visible/Guava
- Automatic correlation dashboard for everything - we're not Guava

---

# Maybe goal?
- Showcase long term progression of medium/long term stats captured
  - Is my P20/P80 HR going up month to month? No?
  - Cheap-ish freemium model to cover some expenses?
- Sharing profiles with doctors / interventions lists with patient-friends?

---

# Limits:
- Wearables can be noisy, PoC tuned on single person and oura only
  - Especially cheaper garmins are all over the place
- Garmin allows API only for enterprise, should be free but ¯\\__(ツ)_/¯
  - Unofficial API but might be maintenance heavy
- Changes to algs. could invalidate long term correlations
  - Oura did it few times already
- Too high dimensionality data can make it harder to differentiate true causality from random correlation
  - More data can be bad


---

# POC