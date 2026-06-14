# SafeHer Privacy and Security Architecture - Engineering Design Document

## 1. Introduction
The SafeHer platform handles highly sensitive data, primarily the real-time geographic locations of women navigating potentially unsafe environments. A breach of this data is not merely a regulatory violation; it is an immediate physical threat to the users. This document outlines the security architecture designed to protect user identity and location data from both external breaches and internal misuse.

## 2. Data Collection & Sensitivity Classification
To implement the principle of least privilege, all collected data is categorized by sensitivity:

| Data Type | Examples | Classification | Encryption Required |
| :--- | :--- | :--- | :--- |
| **Live Location** | GPS Coordinates during active trip | **CRITICAL** | E2EE (where possible), In-Memory only |
| **Identity Data** | Name, Phone Number, Email, Password Hash | **HIGH** | At Rest (Field-level) & In Transit |
| **Historical Routes** | Past origin/destination pairs, paths taken | **HIGH** | At Rest (Anonymized after X days) |
| **Trusted Contacts** | Phone numbers of friends/family | **HIGH** | At Rest (Field-level) |
| **Community Reports** | Incident descriptions, photos, time, location | **MODERATE** | At Rest (Volume-level) |
| **Device Meta** | OS version, App version, Crash logs | **LOW** | In Transit |

## 3. Privacy-Preserving Location Storage
Storing exact GPS coordinates of where users start and end their journeys inevitably reveals their home and workplace addresses.

*   **Spatial Fuzzing:** The system does not store the exact start/end GPS coordinates. Instead, it snaps the location to the nearest major intersection or applies a randomized noise vector (e.g., $\pm 200$ meters) before saving historical route data.
*   **K-Anonymity for Heatmaps:** When aggregating historical routes to improve the SafeHer Risk Index (SRI), the system ensures $k$-anonymity. A route segment's data is only used if at least $k$ (e.g., $k=5$) different users have traversed that segment within the time window.
*   **Ephemeral Live Tracking:** Live location data during navigation is stored exclusively in high-speed, volatile memory (e.g., Redis with short TTLs) and is never written to persistent disk unless an SOS is triggered.

## 4. Anonymous Community Reporting
Community reports are essential, but users may fear retaliation for reporting crimes or suspicious behavior. 

*   **Decoupling Identity:** The `community_reports` database table does not contain a `user_id` foreign key. 
*   **Blind Signatures / Tokenization:** To prevent spam while maintaining anonymity, the app uses a blinded token system (similar to the Privacy Pass protocol). A user authenticates and receives a batch of cryptographically signed "report tokens." When submitting a report, the app spends a token. The server verifies the token is valid (preventing bot spam) but mathematically cannot link the token back to the user account that requested it.

## 5. Trusted Contact Architecture (E2EE)
Users can share their live location with chosen "Trusted Contacts."

*   **Mutual Consent:** Establishing a connection requires a cryptographically signed handshake (User A sends an invite link, User B accepts within the app).
*   **End-to-End Encryption (E2EE):** SafeHer servers must *not* be able to read live location streams meant for trusted contacts. The navigating user's app encrypts the GPS coordinates using a symmetric key shared only with the trusted contact's device (via a Diffie-Hellman key exchange). The server merely relays the encrypted payload via WebSockets.

## 6. SOS Workflows
When a user feels in immediate danger, the privacy model shifts to prioritize physical safety.

*   **Triggers:** In-app slider, hardware button sequence (e.g., pressing power button 5 times), or voice activation ("SafeHer, Help").
*   **Workflow:**
    1.  **E2EE Downgrade:** If integrated with local Law Enforcement APIs, the app temporarily stops E2EE and sends unencrypted location data directly to the SafeHer server to be routed to emergency dispatchers.
    2.  **Audio/Video Capture:** The app begins recording 10-second encrypted bursts of audio/video, uploading them to a secure, write-only AWS S3 bucket.
    3.  **Broadcast:** SMS alerts with a temporary tracking web-link are dispatched to all Trusted Contacts.

## 7. Data Retention Policies
Compliant with strict data minimization principles:

*   **Live Location:** Deleted immediately upon trip completion (TTL < 4 hours).
*   **Historical Routes:** Kept in a pseudonymized state for 7 days to allow for user disputes or delayed SOS investigations. After 7 days, they are irreversibly aggregated into the routing graph and the individual records are hard-deleted.
*   **Account Data:** Hard-deleted within 30 days of an account deletion request ("Right to be Forgotten").

## 8. Role-Based Access Control (RBAC)
Internal access to systems is strictly controlled:

*   **System Administrators:** Access to infrastructure (AWS, databases), but no access to decryption keys for field-level encrypted PII.
*   **Support Staff:** Access to basic account metadata via an internal portal. **No access** to historical routes or trusted contacts unless the user generates a temporary, time-bound "Support PIN" in their app to grant access.
*   **Law Enforcement Portal:** A dedicated, audited portal where police can request data. Data is only released after uploading a valid subpoena or exigent circumstance warrant, manually reviewed by the legal team.

## 9. Encryption Strategy
*   **In Transit:** All API traffic enforces TLS 1.3. The mobile applications utilize Certificate Pinning to prevent Man-in-the-Middle (MitM) attacks, even if the device's root certificate store is compromised.
*   **At Rest:** 
    *   Volumes/Disks: AES-256 block-level encryption (e.g., AWS EBS encryption).
    *   Database Fields: Application-layer encryption for `phone_number` and `email` using AES-256-GCM. The database administrator seeing a database dump cannot read these fields.

## 10. Abuse Prevention Systems
*   **Rate Limiting:** API Gateway restricts the number of login attempts, report submissions, and route requests per IP and per device ID.
*   **Anomaly Detection:** Heuristics flag unusual behavior (e.g., a single device generating reports across opposite sides of a city within minutes).
*   **Account Verification:** Mandatory OTP verification via SMS during onboarding to deter bot farms.

## 11. Compliance
*   **GDPR (General Data Protection Regulation):** Implements Data Controller standards. Ensures explicit opt-in consent for location tracking, provides a simple mechanism to download all user data (Data Portability), and automated data deletion workflows.
*   **Indian DPDP Act (Digital Personal Data Protection Act):** Fulfills the obligations of a Data Fiduciary. Ensures notice and consent are provided in clear, localized languages. Establishes a grievance redressal mechanism (in-app support portal) and designates a Data Protection Officer (DPO).

## 12. Unique Privacy Risks for Women's Safety Apps

> [!CAUTION]
> **Risk 1: Stalkerware & Abusive Partners**
> *Threat:* An abusive partner has physical access to the victim's phone or forces them to open the app to see their routes.
> *Mitigation:* 
> 1. **Biometric App Lock:** Require FaceID/Fingerprint to open the app, even if the phone is unlocked.
> 2. **Discreet Mode:** A UI toggle that makes the app look like a generic utility (e.g., a calculator or generic map) until a specific gesture is performed.
> 3. **Ambiguous Notifications:** Push notifications should never say "You are on a high-risk street." They should be generic: "Map update available."

> [!WARNING]
> **Risk 2: Honey-potting (Reverse Engineering Safe Routes)**
> *Threat:* Predators download the app, figure out the "Safest Routes" that the engine frequently recommends to women at night, and wait along those specific paths.
> *Mitigation:* 
> 1. **Route Randomization:** The routing engine must never be entirely deterministic. If there are 3 similarly safe routes, the engine should randomly select between them to distribute foot traffic and prevent predictable patterns.
> 2. **Continuous Monitoring:** If community reports indicate sudden suspicious activity on a traditionally "safe" route, the engine must aggressively reroute traffic away from it instantly.
