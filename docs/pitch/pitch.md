---
marp: true
---

# Illbiometrics - pitch deck

Petr Houška, 06/12/2025

---

## Highest quality medium/long term biometrics data correlatable with events s.a treatment attempts and infections

---

# Current treatment data:
- Lump everyone together & ask improve/not: [Patient-Reported Treatment Outcomes in ME/CFS and Long COVID](https://www.medrxiv.org/content/10.1101/2024.11.27.24317656v1.full)
- Manual input. s.a. current Amatica
- Manual input of symtoms + automatic correlation with meds: [Guava](https://guavahealth.com/)
- Manual input of symtom severity & correlate yourself: [Visible](https://www.makevisible.com/)

---

# Gaps:
- Patient reported outcoimes:
  - No strattification of groups
  - No correlation to other signals
  - Human memory is very unreliable
  - Only signal dimension: helped or not 
- Manual input: amatica, guava, visible
  - No aggregation
  - No sharing
  - Manual association of symtoms severity to association

---

## Amatica already has data that allows for deep strattification and groups. 

## Let's correlate it with desease progression and events.

---

# Goals:
- Link wearable biolmetrics devices and extract medium/long term statistics 
  - Garmin, Oura, ?Apple watch, ?Loop, ?Visisble
  - Common stats, s.a. HRv (normalize), p20/50/80 or wake HR, number of weakups, length of sleep
- Use stats to stattify patients in groups: stats + biomarkers + questionaire
- Augument event input(*)  with changes in ^^ medium/long term stats
  - Check the aggregates for month before treatment and month after
  - Easier remembering whether treatment helped
  - Additional per treatment/event signal


- (*) Extend treatments with infections & high stress events

---

# Non-goals
- Day to day pacing helper - we're not Visible/Guava
- Day to day symtoms tracking - we're not visible/Guava
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
  - Unofficial API but might be maintainence heavy
- Changes to algs. could invalidate long term correlations
  - Oura did it few times already
- Too high dimensionality data can make it harder to differentiate true causaility from random correlation
  - More data can be bad


---

# POC