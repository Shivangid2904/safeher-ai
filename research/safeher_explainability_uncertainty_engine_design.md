# SafeHer Explainability and Uncertainty Engine - Engineering Design Document

## 1. Introduction
The SafeHer Explainability and Uncertainty Engine sits alongside the core routing engine. Its primary goal is to foster trust by making the SafeHer Risk Index (SRI) transparent, quantifying the certainty of its recommendations, and presenting this data to the user in an actionable, human-readable format.

## 2. Explaining the Risk Score (Feature Attribution)
To explain *why* a location received its score, the engine deconstructs the composite SRI back into its foundational features.

*   **Rule-Based/Linear Models:** If the SRI is a weighted sum of factors (e.g., $SRI = w_1 \cdot L + w_2 \cdot R + w_3 \cdot H$), the explanation engine isolates the terms contributing the most variance from the baseline.
*   **Machine Learning Models:** If a non-linear model (e.g., Random Forest, Gradient Boosting) is used to infer risk, the engine will utilize SHAP (SHapley Additive exPlanations) values to determine the marginal contribution of each feature to the final risk score for a specific road edge.

The top 2-3 factors with the highest positive contribution to the risk score are flagged as "Primary Contributors," while significant negative contributions are flagged as "Safety Factors" (e.g., "Well-lit").

## 3. Confidence Scoring Framework
Confidence ($C \in [0, 1]$) represents the engine's certainty in its calculated risk score. It is derived from three dimensions:

1.  **Data Density:** The volume of data points available for a specific geographic tile or road segment (e.g., number of recent community reports, presence of municipal lighting data).
2.  **Data Recency (Decay):** Older data yields lower confidence. A community report from 10 minutes ago provides high confidence; a report from 24 hours ago provides low confidence.
3.  **Source Reliability:** Official municipal data or verified safe havens have a reliability score of 1.0. Unverified, single-user community reports have a lower reliability score.

$$Confidence = \sum (Weight_i \cdot Density_i \cdot Recency_i \cdot Reliability_i)$$

## 4. Distinguishing Risk Types & Handling Missing Data
To provide accurate context, the system categorizes the source of the risk score:

*   **Measured Risk:** The score is derived primarily from direct, recent observations. (e.g., User reports of a broken streetlight within the last hour, real-time municipal API data). 
    *   *Handling:* These carry the highest confidence scores.
*   **Inferred Risk:** No direct recent measurements exist, but the score is calculated via spatial interpolation (e.g., an adjacent street has a high crime rate) or temporal historical averages (e.g., "This street is usually dark at this time"). 
    *   *Handling:* Applied when direct data is missing. Results in a moderate confidence score.
*   **Unknown Risk:** Severe lack of both recent and historical data. (e.g., A newly constructed road, or a rural area with no SafeHer users). 
    *   *Handling:* The engine assigns a default "Baseline Risk" corresponding to the city average, but flags it with an extremely low confidence score.

## 5. User-Facing Explanations (UX/NLG)
Raw numerical scores and SHAP values must be translated using Natural Language Generation (NLG) templates to ensure they are easily digestible.

**Data Structure to UI Translation:**
Instead of passing raw JSON to the frontend: `{"sri": 82, "conf": 0.85, "top_factors": ["lighting_poor", "recent_report"]}`

**The UI renders using text templates:**

> **Risk = 82**
>
> ⚠️ **Primary contributors:**
> *   Poor lighting
> *   Recent harassment reports
> *   No nearby safe havens
>
> 🛡️ **Confidence: High** *(Based on 3 recent reports and municipal data)*

**Positive Explanations (Low Risk):**

> **Risk = 15**
>
> ✨ **Safety factors:**
> *   Well-lit area
> *   High foot traffic
> *   Near 2 Safe Havens
>
> 🛡️ **Confidence: Very High**

## 6. Database Schema Additions
To support explainability, the existing PostgreSQL/PostGIS architecture requires schema extensions to persist metadata without bloating the primary routing graph.

**New Table: `edge_risk_profiles`** (1-to-1 mapping with the main `road_edges` table)
*   `edge_id` (UUID, Foreign Key)
*   `sri_score` (Float, 0.0 - 100.0)
*   `confidence_score` (Float, 0.0 - 1.0)
*   `risk_category` (ENUM: 'MEASURED', 'INFERRED', 'UNKNOWN')
*   `risk_attributions` (JSONB) - *Stores key-value pairs of feature contributions (e.g., `{"lighting": 0.4, "reports": 0.5, "havens": -0.1}`). JSONB is preferred over strict columns to allow rapid evolution of features.*
*   `last_calculated_at` (Timestamp)

## 7. Impact of Uncertainty on Route Selection
Uncertainty ($U = 1 - Confidence$) acts as a penalty in the A* routing algorithm. A user navigating at night is generally *risk-averse*. They prefer a route with a known, moderate risk over a route that *might* be safe but has completely unknown risk.

**Upper Confidence Bound (UCB) Routing:**
Instead of routing based on the expected risk ($SRI_{expected}$), the routing engine uses the pessimistic UCB risk:

$$SRI_{routing\_cost} = SRI_{expected} + (k \cdot Uncertainty)$$

Where $k$ is an uncertainty aversion parameter. This ensures the engine avoids "Unknown Risk" areas (which get heavily penalized by the $Uncertainty$ term) unless the known alternative is verifiably dangerous.

## 8. Potential Biases and Fairness Concerns
The explainability engine must actively monitor and mitigate systemic biases inherent in crowdsourced safety data:

> [!WARNING]
> **Reporting Bias (The "Affluence" Bias):** 
> Affluent neighborhoods often have higher app adoption and more frequent reporting of minor incidents, while marginalized neighborhoods might have lower adoption, leading to "Unknown Risk".
> *Mitigation:* The engine clearly labels "Unknown Risk" rather than defaulting it to "Safe." It must not heavily penalize areas just because they lack smartphone users.

> [!WARNING]
> **Feedback Loops (Predictive Policing Effect):** 
> If an area is marked high risk, users avoid it. With fewer users, natural surveillance drops (making it actually less safe) and no new positive reports are generated to lower the risk score. The area becomes permanently "redlined."
> *Mitigation:* Implement "Confidence Decay." If an area has no new reports, its risk score should slowly regress to the mean (baseline), and its confidence should drop to "Unknown," encouraging exploration or manual auditing rather than permanent avoidance.

> [!CAUTION]
> **Malicious Reporting / Trolling:** 
> Coordinated false reports could be used to artificially inflate the risk of a specific business or neighborhood.
> *Mitigation:* The confidence framework handles this by assigning low reliability to unverified, clustered reports from new or low-reputation accounts, requiring manual review before significantly altering the SRI.
