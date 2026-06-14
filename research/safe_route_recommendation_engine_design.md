# SafeHer Safe Route Recommendation Engine - Engineering Design Document

## 1. Introduction
The SafeHer Safe Route Recommendation Engine is responsible for generating personalized, safety-optimized navigational paths for users. By leveraging the SafeHer Risk Index (SRI), PostgreSQL/PostGIS, OSMnx road graphs, community reports, and designated safe havens, the engine aims to balance travel time with passenger safety.

## 2. Routing Algorithm Comparison
To select the core algorithm for the recommendation engine, we evaluate three primary approaches:

*   **Dijkstra's Algorithm:**
    *   *Pros:* Guarantees the absolute shortest/safest path. Simple to implement and understand.
    *   *Cons:* Computationally expensive for large spatial networks as it explores uniformly in all directions. Lacks the heuristic advantage of directional routing.
*   **A* Search Algorithm:**
    *   *Pros:* Significantly faster than Dijkstra due to its heuristic function (e.g., Euclidean distance to the destination), which guides the search directionally. Guarantees optimal path if the heuristic is admissible (never overestimates the cost).
    *   *Cons:* The heuristic must be carefully designed, especially when edge weights represent abstract concepts like "risk" rather than pure distance. An inadmissible heuristic can lead to sub-optimal routes.
*   **Multi-objective Routing (e.g., Pareto-optimal routing, NAMOA*):**
    *   *Pros:* Naturally handles competing objectives (travel time vs. safety) by finding a set of Pareto-optimal routes, allowing the user to choose their preferred trade-off.
    *   *Cons:* Extremely computationally demanding. Finding all Pareto-optimal paths is generally NP-hard. Real-time application for long routes on mobile devices is highly challenging.

> [!TIP]
> **Algorithm Selection:** A modified **A* Search Algorithm** is chosen as the core engine. It provides the necessary performance for real-time mobile applications while allowing for customizable edge weights to factor in both time and risk. To achieve the effect of multi-objective routing without the exponential computational cost, we will parameterize the A* cost function to generate specific route "modes" (Fastest, Safest, Balanced).

## 3. Incorporating Risk into Edge Weights
In a standard routing graph, an edge ($E$) connecting two nodes represents a road segment. The weight ($W$) of this edge typically represents travel time or distance. In SafeHer, $W$ must be a composite function of both travel time ($T$) and the SafeHer Risk Index ($SRI$).

Let:
*   $T_e$: Estimated travel time for edge $e$.
*   $SRI_e$: SafeHer Risk Index for edge $e$ (normalized, e.g., 0.0 to 1.0, where 1.0 is highest risk).
*   $C_e$: The composite cost weight for edge $e$.

The composite cost function is defined as:
$$C_e = \alpha \cdot T_e + \beta \cdot f(SRI_e)$$

Where $\alpha$ and $\beta$ are weighting coefficients that determine the route mode. The function $f(SRI_e)$ translates the risk index into a cost penalty. A non-linear function (e.g., exponential) is recommended for $f(SRI_e)$ so that extremely high-risk areas are heavily penalized and effectively avoided unless absolutely necessary:

$$f(SRI_e) = e^{k \cdot SRI_e} - 1$$
*(where $k$ is a tuning parameter).*

## 4. Route Modes Design
By adjusting the $\alpha$ and $\beta$ parameters in the composite cost function, the engine generates three distinct route options for the user:

### 4.1 Fastest Route
*   **Objective:** Minimize travel time, with minimal regard for safety unless risk is critical.
*   **Parameters:** $\alpha = 1.0$, $\beta = 0.1$ (or a very small non-zero value).
*   **Behavior:** Acts similarly to a standard navigation app. The small $\beta$ ensures that if two routes have identical times, the slightly safer one is chosen, but time is the primary driver.

### 4.2 Safest Route
*   **Objective:** Minimize risk exposure, regardless of the increased travel time.
*   **Parameters:** $\alpha = 0.1$, $\beta = 1.0$.
*   **Behavior:** The algorithm will take significant detours to avoid high-SRI areas. It actively routes towards well-lit streets, higher foot traffic areas, and proximity to designated Safe Havens.

### 4.3 Balanced Route (Recommended)
*   **Objective:** Provide a sensible trade-off. Avoid high-risk areas without excessive detours.
*   **Parameters:** $\alpha = 0.5$, $\beta = 0.5$ (Weights dynamically tuned based on user feedback).
*   **Behavior:** Penalizes risky streets but bounds the maximum acceptable detour time. For example, it might enforce a constraint: "Find the safest route that adds no more than 20% to the fastest travel time." This can be implemented using a Constrained Shortest Path First (CSPF) approach built on top of A*.

## 5. Real-Time Community Reports Impact
Community reports (e.g., "suspicious activity", "streetlights out") must affect routing dynamically.

1.  **Event Ingestion:** When a user submits a report, it is geocoded and assigned a severity score and a decay half-life (e.g., a "suspicious person" report decays faster than a "broken streetlight" report).
2.  **Spatial Mapping:** The report is mapped to the nearest edges in the PostGIS road graph.
3.  **Dynamic SRI Modification:** An in-memory caching layer (e.g., Redis) holds dynamic modifiers for edge weights. The base SRI from the database is summed with the dynamic modifier from recent reports.
    $$SRI_{effective} = SRI_{base} + \sum (Report_{severity} \cdot TimeDecayFactor)$$
4.  **Threshold Triggers:** If a severe report is verified, it can set the $SRI_{effective}$ of an edge to effectively $\infty$, temporarily closing the road to the routing engine.

## 6. Route Recalculation
If a user is currently navigating and a new community report impacts their planned route:

1.  **Background Monitoring:** The user's active route geometry (a sequence of edge IDs) is registered in a spatial pub/sub system.
2.  **Intersection Check:** When a new report updates the graph, the system checks if the affected edges intersect with any active user routes.
3.  **Threshold Evaluation:** If an active route is impacted, the engine evaluates the new cost of the remaining journey. If the new cost exceeds a certain threshold (or if the edge is now marked impassable/critical risk), a recalculation is triggered.
4.  **Re-routing:** The engine runs the A* algorithm from the user's *current GPS location* to the destination, using the updated $SRI_{effective}$ graph. The user is prompted with the new, safer route.

## 7. Route Confidence Scoring
Users need to understand *why* a route is recommended and how safe it actually is. We assign a Confidence Score (0-100%) to every generated route.

The score is derived from:
*   **Data Density (30% weight):** How much historical and real-time data exists for the edges on this route? Routes through areas with high community engagement and frequent sensor updates score higher.
*   **Average SRI (40% weight):** Inversely proportional to the average $SRI_{effective}$ of the route. Lower risk = higher confidence.
*   **Safe Haven Proximity (20% weight):** The percentage of the route that falls within a defined radius (e.g., 200m) of verified Safe Havens.
*   **Time of Day Variance (10% weight):** If the routing is done during a time where historical data variance is high (e.g., 2 AM vs 2 PM), confidence is slightly lowered to reflect uncertainty.

> [!NOTE]
> **UX Display:** Displayed as a shield icon with a percentage (e.g., 🛡️ 92% Safe Route) and a brief summary ("Well-lit, 3 Safe Havens nearby").

## 8. Computational Bottlenecks & Scaling Concerns

### 8.1 Bottlenecks
*   **Database I/O for Large Graphs:** Fetching graph data from PostGIS for every routing request is too slow.
*   **Heuristic Calculation in A*:** Complex composite cost functions ($T_e$ + $SRI_e$) calculated on the fly for thousands of node expansions during a search.
*   **Concurrent Recalculations:** A major incident in a dense area could trigger simultaneous route recalculations for hundreds of active users.

### 8.2 Scaling Solutions
*   **In-Memory Graph Processing:** Do not query PostGIS for real-time routing. Use specialized graph routing engines (like pgRouting or Valhalla) and load the network graph into RAM. PostGIS is used for storage and batch updates of the base graph.
*   **Tile-Based Graph Partitioning:** Split the OSMnx graph into geographic tiles. Load only the tiles relevant to the user's start and end points into memory.
*   **Caching Base Routes:** Cache the Fastest route calculations for common origin-destination pairs. Dynamic risk factors usually affect the Safest and Balanced routes more frequently.
*   **Asynchronous Processing:** Use message queues (e.g., RabbitMQ, Kafka) to handle incoming community reports and pub/sub for triggering user recalculations without blocking the main API threads.
*   **Customizable Route Planning (CRP) / Contraction Hierarchies (CH):** For extreme scaling, implement CRP or CH to pre-compute shortcuts in the graph. This makes A* exponentially faster. CRP is particularly well-suited for routing where edge weights (like risk) change frequently, as it allows rapid metric customization without fully rebuilding the graph hierarchy.
