# SafeHer Risk Index (SRI) - System Design Document

**Status:** Draft | **Author:** Lead Architect | **Project:** SafeHer AI

## 1. Introduction
The SafeHer Risk Index (SRI) is the core algorithmic component of the SafeHer AI platform. It replaces the synthetic risk prototype with a comprehensive, data-driven scoring system that evaluates the safety of urban environments specifically for women. The SRI is designed to generate a localized risk score (0-100) that powers the Safe-Route Recommendation Engine.

---

## 2. Safety Factors & Categorization

To accurately measure safety, the SRI relies on a multi-dimensional analysis of the urban environment.

### 2.1 Environmental Factors
*   **Street Lighting Levels:** 
    *   *Why:* Poorly lit areas significantly increase the perception of danger and the actual likelihood of isolated incidents.
    *   *Data Source:* City municipal data, satellite imagery (Nightlights data), community reports.
    *   *Type:* Static (infrastructure) & Dynamic (outages).
*   **Line of Sight / Visibility:**
    *   *Why:* Blind corners, overgrown vegetation, and narrow alleys create hiding spots and reduce escape routes.
    *   *Data Source:* Google Street View API, OpenStreetMap (OSM) building footprints, user feedback.
    *   *Type:* Static.
*   **Weather Conditions:**
    *   *Why:* Heavy rain or snow can reduce visibility, empty streets of bystanders, and slow down emergency responses.
    *   *Data Source:* Weather APIs (OpenWeatherMap, Tomorrow.io).
    *   *Type:* Real-time.

### 2.2 Infrastructure Factors
*   **Proximity to Safe Havens (24/7 Stores, Hospitals, Police Stations):**
    *   *Why:* Presence of open, populated establishments provides immediate refuge in an emergency.
    *   *Data Source:* Google Places API, Yelp API, OpenStreetMap.
    *   *Type:* Dynamic (based on opening hours).
*   **Public Transit Density:**
    *   *Why:* Well-connected areas with active bus stops/subways offer quick transit options but can also attract loitering if poorly maintained.
    *   *Data Source:* General Transit Feed Specification (GTFS) data from local transit authorities.
    *   *Type:* Static.
*   **CCTV & Emergency Infrastructure:**
    *   *Why:* The presence of cameras and emergency blue-light phones acts as a deterrent and provides security.
    *   *Data Source:* City open data portals, user mapping.
    *   *Type:* Static.

### 2.3 Crime Factors
*   **Historical Crime Rates (Violent & Theft):**
    *   *Why:* General historical patterns indicate the systemic safety baseline of a neighborhood.
    *   *Data Source:* Local police department open data (e.g., UK Police API, Chicago Data Portal).
    *   *Type:* Dynamic (Updated daily/weekly).
*   **Gender-Based Harassment/Violence:**
    *   *Why:* Specifically tracking catcalling, stalking, and harassment which are often underreported to police but heavily affect women's mobility.
    *   *Data Source:* Community reports within SafeHer, localized NGOs (e.g., Safecity).
    *   *Type:* Real-time / Dynamic.

### 2.4 Temporal Factors
*   **Time of Day / Day of Week:**
    *   *Why:* Risk profiles shift dramatically after sunset, during late-night hours, and on weekends.
    *   *Data Source:* System clock mapped against sunrise/sunset APIs.
    *   *Type:* Real-time.

### 2.5 Community Factors
*   **Crowd Density / "Eyes on the Street":**
    *   *Why:* Jane Jacobs’ theory—active streets with bystanders deter crime and increase perceived safety.
    *   *Data Source:* Mobile carrier data (if accessible), public transit ridership, foot-traffic APIs (e.g., Safegraph).
    *   *Type:* Real-time.
*   **Community Safety Reports (Crowdsourcing):**
    *   *Why:* Real-time warnings from other SafeHer users about suspicious activity, broken lights, or feeling unsafe.
    *   *Data Source:* SafeHer App (User-Generated Content).
    *   *Type:* Real-time.

### 2.6 Personal/Contextual Factors
*   **User Familiarity:**
    *   *Why:* A user is less vulnerable in a neighborhood they know well compared to an unknown area.
    *   *Data Source:* User profile / App history.
    *   *Type:* Dynamic.

---

## 3. Initial Mathematical Scoring Framework

The SRI produces a score from **0 (Maximum Safety) to 100 (Maximum Risk)** for a given geographic node $N$ at time $t$. 

**Base Equation:**
$$SRI(N, t) = \alpha \cdot C(N) + \beta \cdot I(N, t) + \gamma \cdot E(N, t) - \delta \cdot P(N, t)$$

Where:
*   $C(N)$ = Crime Factor Score (Historical + Recent reports)
*   $I(N, t)$ = Infrastructure Factor (Negative score if isolated, positive if near Safe Havens at time $t$)
*   $E(N, t)$ = Environmental Factor (Lighting, visibility, weather)
*   $P(N, t)$ = Protective Community Factors (Crowds, 'Eyes on the street')
*   $\alpha, \beta, \gamma, \delta$ = Learned weights from the Machine Learning model.

**Time-of-Day Multiplier:**
The base score is multiplied by a temporal factor $T(t)$.
$$Final\_SRI = SRI(N, t) \times T(t)$$
*If $t$ is 2 AM in a commercial district that is closed, $T(t) = 1.5$ (Risk amplification). If $t$ is 2 PM, $T(t) = 0.8$ (Risk reduction).*

---

## 4. Feature Schema (PostgreSQL / PostGIS)

To support this model, spatial data will be stored in PostgreSQL using the PostGIS extension. 

### Table: `spatial_nodes`
Represents intersections or road segments.
```sql
CREATE TABLE spatial_nodes (
    node_id UUID PRIMARY KEY,
    geom GEOMETRY(Point, 4326),          -- Coordinates
    base_crime_score FLOAT,              -- Pre-computed historical crime risk
    lighting_quality INT,                -- 1 (Poor) to 5 (Excellent)
    visibility_score INT,                -- 1 (Blind corners) to 5 (Open)
    has_cctv BOOLEAN,
    last_updated TIMESTAMP
);
```

### Table: `safe_havens`
```sql
CREATE TABLE safe_havens (
    haven_id UUID PRIMARY KEY,
    name VARCHAR(255),
    geom GEOMETRY(Point, 4326),
    type VARCHAR(50),                    -- 'Police', '24/7 Store', 'Hospital'
    opening_hours JSONB                  -- Store operating hours
);
```

### Table: `community_reports`
```sql
CREATE TABLE community_reports (
    report_id UUID PRIMARY KEY,
    user_id UUID,
    geom GEOMETRY(Point, 4326),
    report_type VARCHAR(50),             -- 'Harassment', 'Suspicious', 'Broken Light'
    severity INT,                        -- 1 to 5
    is_active BOOLEAN,                   -- Fades out after X hours
    created_at TIMESTAMP
);
```

---

## 5. Integration with Safe-Route Recommendation Engine

Traditional routing engines (Google Maps, Waze) use **Dijkstra's** or **A*** algorithms to find the shortest path, where the "edge weight" is simply *distance* or *time*.

The SafeHer Routing Engine modifies the edge weight to account for the SRI:
$$Edge\_Weight(A \rightarrow B) = (w_1 \cdot Distance) + (w_2 \cdot \overline{SRI}_{AB})$$

**How it works:**
1.  The user requests a route from Origin to Destination.
2.  The engine pulls the road network graph from PostGIS via pgRouting.
3.  Instead of calculating the shortest distance, it queries the dynamic SRI score for all edges (streets) in the bounding box based on the *current time*.
4.  Edges with high SRI scores (e.g., an unlit alleyway at midnight) are given artificially high path costs.
5.  The A* algorithm naturally avoids these high-cost edges, routing the user through slightly longer, but significantly safer, well-lit main streets with open businesses.

---

## 6. Implementation Roadmap

To build this systematically, we separate factors into Phase 1 (MVP) and Phase 2 (Deferred).

### Phase 1 MVP (Feasible & High Impact)
> [!IMPORTANT] 
> Focus on data that is easily accessible via open APIs and static databases to quickly deliver a working routing engine.

*   **Temporal Factors:** Time of day, sunrise/sunset calculations.
*   **Historical Crime:** Batch imported monthly from police open-data APIs.
*   **Safe Havens:** Queried from Google Places API (24/7 stores, police, hospitals).
*   **Street Lighting (Proxy):** Based on major vs. minor roads in OpenStreetMap (main roads are assumed better lit than alleys).
*   **Community Reports:** Basic capability for users to pin a "feel unsafe" or "harassment" marker.

### Phase 2 (Deferred to Long-Term Vision)
> [!NOTE] 
> These require complex real-time integrations, expensive APIs, or high user density.

*   **Real-time Foot Traffic:** Safegraph or telecom data is expensive and complex to integrate.
*   **Weather API Integration:** Less critical for MVP, can be added later as a risk multiplier.
*   **Live CCTV Analysis:** Massive technical and privacy hurdles.
*   **Micro-level Lighting/Visibility:** Requires deep scanning of Google Street View using Computer Vision to analyze exact line-of-sight and lumen levels.
*   **User Familiarity:** Requires long-term tracking of user location data, which introduces heavy privacy considerations.
