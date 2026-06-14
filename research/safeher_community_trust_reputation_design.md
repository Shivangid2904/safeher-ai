# SafeHer Community Trust and Reputation System - Engineering Design Document

## 1. Introduction
The SafeHer application relies on crowdsourced intelligence to dynamically update its safety routing. However, crowdsourced systems are highly susceptible to noise, spam, and malicious manipulation. The Community Trust and Reputation System serves as the immune system of the platform, ensuring that only high-quality, verified data impacts the SafeHer Risk Index (SRI) and routing engine.

## 2. User Reputation Scoring ($R_{user}$)
User reputation determines how much weight the system assigns to a user's reports. 
*   **Scale:** $0.0$ to $1.0$.
*   **Initialization:** To mitigate abuse from newly created accounts, all users start with a baseline reputation of $0.3$ ("Untrusted").
*   **Reputation Gains:** Increases when their reports are corroborated by other users (upvoted or independently reported nearby) or verified by moderators/official data.
*   **Reputation Penalties:** Decreases severely if their reports are repeatedly dismissed by high-reputation users or flagged as false. 
*   **Privacy Note:** To adhere to the Privacy Architecture, reputation is tracked via a pseudonymous ID (`rep_id`) distinct from the user's PII.

## 3. Report Credibility Scoring ($C_{report}$)
Every submitted community report is immediately assigned a Credibility Score ($C_{report} \in [0, 1]$).
$C_{report}$ is calculated as a weighted sum of several factors:
1.  **Submitter Reputation (40%):** The $R_{user}$ of the person reporting.
2.  **Spatial Proximity (20%):** Validates against GPS spoofing. A report submitted while the device's GPS is 5 meters from the event scores high; reporting an event 50km away scores zero.
3.  **Media Evidence (20%):** Reports containing an attached photo or audio snippet receive a credibility boost (processed via ML to ensure relevance and filter NSFW content).
4.  **Historical Account Age & Activity (20%):** Older, consistently active accounts are granted slightly higher baseline credibility.

## 4. Handling Abuse Vectors
> [!CAUTION]
> The system must automatically detect and mitigate various attack vectors to prevent the routing engine from being poisoned.

*   **False Reports:** If a report achieves a low credibility score and receives "Dismiss/False" votes from passing users, the system permanently archives it and applies a multiplier penalty to the original reporter's $R_{user}$.
*   **Spam:** Strict rate-limiting is applied based on $R_{user}$. A new user ($R=0.3$) might be limited to 1 report per hour. A trusted user ($R=0.9$) can report more frequently. 
*   **Coordinated Attacks (Review Bombing):** If an unusually high volume of reports targets a specific geographic radius within a short timeframe (e.g., 50 reports on one block in 10 minutes), a **Velocity Lock** is triggered. The system caps the aggregate impact of these reports on the SRI until a Moderator manually reviews the cluster or a highly trusted user verifies it.
*   **Sybil Attacks (Fake Accounts):** Defended at the perimeter via mandatory SMS/OTP verification during account creation and device fingerprinting to prevent a single malicious actor from operating 100 accounts from one phone to upvote their own false reports.

## 5. Report Verification Workflows
Reports mature through active and passive verification.
*   **Active Verification:** When a user navigates near an active report (e.g., "Broken Streetlight"), the app briefly prompts them at a safe juncture: *"Is the streetlight still broken here? [Yes] [No]"*. This acts as an explicit upvote or downvote.
*   **Passive Verification:** If 50 users navigate through a recently reported "Suspicious Activity" zone and none of them trigger an SOS or submit a corroborating report, the original report's credibility decays faster.

## 6. Trust Decay Over Time
Reports do not live forever. Their impact on the SRI decays exponentially based on the category of the report.
$$Credibility(t) = C_{initial} \cdot e^{-\lambda t}$$
Where $\lambda$ is the decay constant specific to the report category:
*   **Transient Events** (e.g., "Suspicious Group", "Harassment"): Fast decay ($\lambda$ is high). The report loses influence in a matter of hours.
*   **Infrastructure Events** (e.g., "Broken Streetlight", "No Sidewalk"): Slow decay ($\lambda$ is low). The report persists for days or weeks until actively downvoted or fixed by the municipality.

## 7. Moderator Tools
A web-based dashboard is provided for SafeHer Trust & Safety Operations.
*   **Anomaly Map:** Highlights "Velocity Locks" (suspected coordinated attacks) and areas with rapidly shifting SRI.
*   **Global Suppression:** One-click ability to invalidate a cluster of reports and freeze the accounts that submitted them.
*   **Official Verification:** Moderators can promote a community report to an "Official Alert," overriding normal decay and credibility limits (e.g., verifying an active police situation via news feeds).

## 8. System Integrations (SRI & Routing Engine)
**Impact on SRI:** 
The active dynamic risk modifier for a road segment is a function of the reports on it:
$$Dynamic\_SRI = \sum (Severity_{report} \cdot C_{report}(t))$$

**Impact on Routing:**
*   **High Credibility, High Severity:** Immediately spikes the edge cost in the A* algorithm. Users are actively re-routed away.
*   **Low Credibility, High Severity:** Marginally increases the edge cost. If it's still the fastest route, the app will route the user through it but display a warning: *"Unverified report ahead, exercise caution."*

## 9. Database Schema Additions
To implement the trust system, the following tables/columns are required:

**Table: `user_reputation`**
*   `rep_id` (UUID, Primary Key - decoupled from PII)
*   `reputation_score` (Float, default 0.3)
*   `reports_submitted` (Int)
*   `reports_verified` (Int)

**Table: `community_reports` (Updates)**
*   `report_id` (UUID)
*   `rep_id` (UUID, FK)
*   `category` (Enum)
*   `initial_credibility` (Float)
*   `current_credibility` (Float - updated via chron job or trigger)
*   `status` (Enum: PENDING, VERIFIED, REJECTED, VELOCITY_LOCKED)

**Table: `report_verifications`**
*   `verification_id` (UUID)
*   `report_id` (UUID, FK)
*   `rep_id` (UUID, FK) - The user verifying
*   `vote` (SmallInt: +1 or -1)
*   `timestamp` (DateTime)

## 10. Fairness and Abuse Risks
> [!WARNING]
> **Vague Reporting Categories & Implicit Bias:**
> Providing a generic "Suspicious Person" category often leads to implicit racial or class bias (e.g., reporting unhoused individuals simply for existing in a space). 
> *Mitigation:* Remove vague categories. Force users to report specific *actions* or *environmental factors* (e.g., "Verbal Harassment", "Following", "Poor Lighting") to reduce discriminatory reporting.

> [!WARNING]
> **The "Mob Mentality" Trap:**
> High-reputation users wield immense power. If a clique of high-reputation users decides to maliciously downvote legitimate reports from a specific neighborhood, the system could incorrectly label that area as safe.
> *Mitigation:* Cap the maximum influence a single user's vote can have on a report, regardless of how high their reputation is. Enforce logarithmic scaling on reputation power.
