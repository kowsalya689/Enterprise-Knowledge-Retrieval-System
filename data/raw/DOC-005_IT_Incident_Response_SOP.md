# Incident Response Procedure (SOP)

*Nimbus Industries Pvt. Ltd. (fictional)*

## Document Control

| Field | Value |
|---|---|
| Document ID | DOC-005 |
| Title | Incident Response Procedure (SOP) |
| Version | 1.2 |
| Owner | Security Operations Manager |
| Department | IT Security |
| Classification | Confidential |
| Effective date | 2026-03-20 |
| Review cycle | Every 12 months |

## 1. Purpose

This Standard Operating Procedure (SOP) describes how Nimbus Industries Pvt. Ltd. detects, classifies, contains, eradicates and learns from information security incidents. It supports the Information Security Policy (DOC-002).

## 2. Scope and Definitions

The SOP covers any event that threatens the confidentiality, integrity or availability of Company information or systems, such as malware, phishing, data leakage, unauthorised access, ransomware or loss of a device.

| Term | Meaning |
|---|---|
| Event | Any observable occurrence in a system or network. |
| Incident | An event that harms or threatens to harm information or systems. |
| IRT | Incident Response Team, led by the Incident Commander. |
| Incident Commander (IC) | The person with authority to direct the response until closure. |
| Containment | Actions that stop an incident from spreading. |
| RCA | Root Cause Analysis, completed after the incident. |

## 3. Severity Classification

| Severity | Definition | Example | Response time | Update frequency |
|---|---|---|---|---|
| P1 Critical | Major impact on operations or confirmed breach of personal or Restricted data | Ransomware on production servers; customer database leaked | Acknowledge within 15 minutes | Every 30 minutes |
| P2 High | Significant impact on one business unit or likely data exposure | Compromised administrator account; malware on several laptops | Acknowledge within 30 minutes | Every 1 hour |
| P3 Medium | Limited impact, contained to one user or system | Single phishing click with no credential entry; lost laptop with disk encryption | Acknowledge within 4 hours | Every 8 hours |
| P4 Low | No impact, policy violation or false alarm | Blocked spam campaign; unapproved software installed | Acknowledge within 1 business day | On closure |

## 4. Roles and Responsibilities

| Role | Responsibility |
|---|---|
| Incident Commander | Declares the incident, sets severity, coordinates the team, approves containment actions that stop business systems. |
| Security analyst | Triage, evidence collection, log analysis. |
| IT operations engineer | Executes containment, restores systems from backup. |
| Communications lead | Prepares internal and external messages after approval by the IC. |
| Legal and Data Protection Officer | Decides on regulatory notification and advises on evidence handling. |
| Management sponsor | Approves major business decisions such as shutting down a system for more than 4 hours. |

## 5. Procedure

### 5.1 Step 1: Detect and report

- Any employee who suspects an incident reports it to the IT Service Desk (extension 4444) or security@nimbus-industries.example without delay.
- The Service Desk creates a ticket with the reporter's name, time, systems involved and a short description.
- Automated alerts from the SIEM and endpoint protection tools also create tickets.

### 5.2 Step 2: Triage and classify

- A security analyst reviews the ticket within the response time for its suspected severity.
- The analyst confirms whether it is an incident and assigns a severity P1 to P4.
- For P1 and P2, the analyst pages the Incident Commander on the on-call rota.

### 5.3 Step 3: Contain

- Isolate affected devices from the network using the endpoint protection console.
- Disable compromised accounts and revoke active sessions and tokens.
- Block malicious IP addresses and domains at the firewall and mail gateway.
- Preserve evidence: take disk or memory images before rebooting or reimaging where possible.
- Record every action with a time stamp in the incident log.

### 5.4 Step 4: Eradicate and recover

- Remove malware, close the vulnerability and reset passwords for affected accounts.
- Restore systems from the last known clean backup; backups are taken daily and kept for 35 days.
- Monitor restored systems closely for 7 days for signs of re-infection.

### 5.5 Step 5: Communicate

The Communications lead informs affected business owners and, for P1 and P2, the executive team. If personal data is involved, the Data Protection Officer decides on notification to the regulator and to affected individuals; the Company aims to notify within 72 hours of confirming a personal data breach. No one may speak to the media without approval of the Chief Executive Officer.

### 5.6 Step 6: Close and learn

- The IC closes the incident after services are normal and the root cause is known.
- A post-incident review is held within 5 working days of closure for P1 and P2 incidents.
- The RCA report lists the timeline, root cause, impact, what went well, what went wrong and action items with owners and dates.
- Lessons are added to the knowledge base and the awareness training.

## 6. Playbooks for Common Incidents

### 6.1 Phishing email

Ask the reporter not to click or forward the message. Search the mail system for the same message and quarantine all copies. If anyone entered credentials, force a password reset and revoke sessions, and treat the case as P2 or higher.

### 6.2 Ransomware

Disconnect the infected device from the network immediately but leave it powered on. Declare P1. Do not pay or contact the attacker without approval of the CEO and Legal. Restore from clean backups after the cause is removed.

### 6.3 Lost or stolen device

Report within 1 hour. The Service Desk locks or remotely wipes the device, disables the user's sessions and checks whether the disk was encrypted. A lost unencrypted device holding Confidential data is handled as P2.

### 6.4 Suspected data leakage

Identify what data and how many people are affected, stop the sharing link or transfer, and involve the Data Protection Officer immediately.

## 7. Escalation Contacts

| Level | Contact | When to call |
|---|---|---|
| 1 | IT Service Desk, extension 4444 | Any suspected incident |
| 2 | On-call security analyst, 24x7 pager | P1 or P2 incidents |
| 3 | Incident Commander | Confirmed P1 or P2 |
| 4 | Chief Information Security Officer | Any breach of personal data or P1 lasting more than 2 hours |
| 5 | Chief Executive Officer | Public impact, regulator contact or ransom demand |

## 8. Records and Metrics

Incident records are kept for 3 years. The security team reports mean time to acknowledge, mean time to contain and the number of incidents by severity to the CISO each month. The SOP is tested with a tabletop exercise twice a year and reviewed every 12 months.

## Revision History

| Version | Date | Approved by | Change summary |
|---|---|---|---|
| 1.2 | 2026-03-20 | Security Operations Manager | Current approved version |
| Previous | 2025 | Security Operations Manager | Annual review and wording updates |
