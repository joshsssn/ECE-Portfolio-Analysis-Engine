# Upgrade vers Refinitiv Data : Guide Complet des Opportunités

Ce document détaille **exhaustivement** comment le passage de **yfinance** à **Refinitiv (LSEG Data API)** peut transformer l'actuel moteur ECE en une plateforme d'analyse de grade institutionnel.

---

## Table des Matières
1. [Valorisation & DCF](#1-valorisation--dcf)
2. [Analyse de Portefeuille & Risque](#2-analyse-de-portefeuille--risque)
3. [Optimisation d'Allocation](#3-optimisation-dallocation)
4. [Screening & Data Sourcing](#4-screening--data-sourcing)
5. [Modèles Quantitatifs StarMine](#5-modèles-quantitatifs-starmine)
6. [News & Sentiment Analysis](#6-news--sentiment-analysis)
7. [Données d'Ownership & Insider Trading](#7-données-downership--insider-trading)
8. [Données de Marché Avancées (Tick History)](#8-données-de-marché-avancées-tick-history)
9. [Fixed Income & Obligations](#9-fixed-income--obligations)
10. [Corporate Actions & M&A](#10-corporate-actions--ma)
11. [Alternative Data](#11-alternative-data)
12. [ESG & Investissement Responsable](#12-esg--investissement-responsable)

---

## 1. Valorisation & DCF
**Fichier impacté :** `valuation_engine.py`

Le gain principal est le passage d'une analyse **rétrospective** (basée sur le passé) à une analyse **prospective** (basée sur le consensus des analystes sell-side).

### 1.1 Consensus IBES (Estimations)
| Aujourd'hui (yfinance) | Avec Refinitiv |
|------------------------|----------------|
| Projections basées sur la moyenne de croissance historique 3Y/5Y | Récupérer la moyenne (Mean), médiane et écart-type des prévisions de centaines d'analystes |

*   **Champs disponibles :** `TR.RevenueMean`, `TR.EbitdaMean`, `TR.NetIncomeMean` pour les 1, 2, 3, 5 prochaines années fiscales.
*   **Avantage :** Le DCF utilise des prévisions réelles, pas une extrapolation naïve.

### 1.2 SmartEstimates (Estimations Corrigées)
*   **Concept :** Au lieu du simple consensus (moyenne de tous les analystes), SmartEstimate pondère les prévisions en fonction de :
    1.  **Recency :** Les prévisions récentes (< 120 jours) comptent plus.
    2.  **Track Record :** Les analystes historiquement précis sont surpondérés.
*   **Utilité :** Prédiction de la direction de l'Earnings Surprise avec ~70% de précision si l'écart SmartEstimate vs Consensus dépasse 2%.

### 1.3 WACC Institutionnel
*   **Champ :** `TR.WACC`
*   **Avantage :** Plus besoin de calculer manuellement le WACC avec Beta + ERP. Refinitiv le calcule avec un coût de la dette basé sur les obligations réelles de l'entreprise.

### 1.4 Modèle StarMine Intrinsic Value (IV)
*   **Concept :** Modèle propriétaire Dividend Discount Model (DDM) multi-stades qui ajuste les prévisions pour éliminer les biais optimistes des analystes.
*   **Utilité :** Sert de "Fair Value" de référence pour comparer avec ton propre DCF.

---

## 2. Analyse de Portefeuille & Risque
**Fichier impacté :** `portfolio_reconstruction.py`, `stress_testing.py`

### 2.1 Stress Testing Macro-Économique
*   **Données disponibles :** Séries macro (CPI Inflation, Fed Funds Rate, 10Y Treasury Yield, EUR/USD, Brent Crude, etc.)
*   **Utilité :** Calculer la sensibilité du portefeuille à un choc de taux de +100bps ou une récession.

### 2.2 Analyse de Style (Fama-French / Barra)
*   **Concept :** Décomposer la performance selon des facteurs : Value, Growth, Size, Quality, Momentum, Low Volatility.
*   **Utilité :** Savoir *pourquoi* le portefeuille gagne ou perd (exposition Value vs Growth, etc.)

### 2.3 Corrélation & Covariance Point-in-Time
*   **Concept :** Utiliser des matrices de covariance qui reflètent les données disponibles à la date T (évite le "look-ahead bias").
*   **Utilité :** Backtests plus réalistes.

---

## 3. Optimisation d'Allocation
**Fichier impacté :** `optimal_allocation.py`

### 3.1 Risque de Crédit (StarMine Credit Risk)
*   **Champ :** `TR.CombinedCreditRiskModelScore`, `TR.ProbabilityOfDefault1Y`
*   **Utilité :** Ajuster le poids max autorisé si l'entreprise est en zone de détresse financière (évite de sur-allouer sur des "value traps").

### 3.2 Short Interest
*   **Champs :** `TR.ShortInterestPctFloat`, `TR.ShortInterest1MoPctChange`
*   **Utilité :** Intégrer le ratio de ventes à découvert comme signal de risque ou de short squeeze potentiel.

### 3.3 Volatilité Implicite (Options)
*   **Données :** Prix des options, IV skew.
*   **Utilité :** Utiliser l'IV comme proxy du risque perçu par le marché (meilleur que la volatilité historique).

---

## 4. Screening & Data Sourcing
**Fichier impacté :** `run_from_screener.py`

### 4.1 Universe Discovery API
*   **Concept :** Au lieu de charger un CSV statique, le script demande dynamiquement à l'API :
    *   "Toutes les actions Tech US avec upside DCF > 20%, Beta < 1.2, croissance EBITDA > 5%"
*   **Utilité :** Fin du travail manuel de sélection de tickers.

### 4.2 Point-in-Time Data
*   **Concept :** Garantie que les données utilisées pour un backtest sont celles disponibles à la date T.
*   **Utilité :** Évite le "survivorship bias" et le "look-ahead bias".

---

## 5. Modèles Quantitatifs StarMine
Suite complète de modèles "alpha" prêts à l'emploi.

### 5.1 Analyst Revisions Model (ARM)
*   **Concept :** Prédit les changements futurs de sentiment des analystes.
*   **Inputs :** Révisions d'EPS, EBITDA, Revenue, et changements de recommandation (Buy/Hold/Sell).
*   **Output :** Score de 1 à 100 (100 = sentiment très haussier).

### 5.2 Earnings Quality Model
*   **Concept :** Évalue la durabilité et la fiabilité des bénéfices.
*   **Utilité :** Détecter les entreprises qui "gonflent" leurs résultats avec de l'ingénierie comptable.

### 5.3 Relative Valuation Model
*   **Concept :** Compare la valorisation d'une action à ses pairs en ajustant pour la croissance attendue.
*   **Utilité :** Identifier les actions statistiquement sous-évaluées vs leurs comparables.

### 5.4 Short Interest Model
*   **Concept :** Combine le short interest avec d'autres signaux pour scorer le potentiel de short squeeze ou de chute.

### 5.5 Price Momentum Model
*   **Concept :** Score de momentum technique ajusté pour la volatilité.

### 5.6 Predicted Surprise
*   **Concept :** Probabilité qu'une entreprise batte ou manque le consensus sur ses prochains résultats.

---

## 6. News & Sentiment Analysis
**Nouveau module potentiel :** `news_sentiment.py`

### 6.1 Machine Readable News (MRN)
*   **Concept :** Flux de news structuré en temps réel avec métadonnées (entité concernée, pertinence, nouveauté).
*   **Sources :** Reuters News + 130 sources tierces.
*   **Latence :** Ultra-faible (microsecondes) pour le trading algorithmique.

### 6.2 News Sentiment Score
*   **Champs :** Sentiment (positif/neutre/négatif), Relevance, Novelty, Confidence.
*   **Utilité :** 
    *   Détecter une chute brutale de sentiment avant que le prix ne réagisse.
    *   Générer des signaux "event-driven" (M&A, earnings, géopolitique).

### 6.3 Buzz Metrics
*   **Concept :** Volume anormal de mentions d'une entreprise dans les news.
*   **Utilité :** Signal d'alerte précoce sur des événements inhabituels.

---

## 7. Données d'Ownership & Insider Trading
**Nouveau module potentiel :** `ownership_signals.py`

### 7.1 Ownership API
*   **Données :** Structure de l'actionnariat (institutionnels, insiders, fonds).
*   **Historique :** Jusqu'à 1997 pour certains marchés.
*   **Utilité :** Identifier les changements de participation des gros acteurs (Blackrock, Vanguard, etc.)

### 7.2 Insider Transactions
*   **Données :** Achats/ventes des dirigeants (CEO, CFO, Board Members).
*   **Utilité :** Signal haussier si les insiders achètent massivement après une chute du cours.

### 7.3 Top Holders Concentration
*   **Données :** % du capital détenu par les 10 plus gros actionnaires.
*   **Utilité :** Risque de "overhang" si un gros actionnaire vend.

---

## 8. Données de Marché Avancées (Tick History)
**Pour trading algorithmique et TCA.**

### 8.1 Tick-by-Tick Data
*   **Granularité :** Nanoseconde.
*   **Couverture :** Depuis 1996, 500+ venues mondiales.
*   **Utilité :** Backtest de stratégies haute fréquence, analyse de microstructure.

### 8.2 VWAP / TWAP Pré-calculés
*   **Concept :** Prix moyen pondéré par volume (VWAP) ou temps (TWAP) disponible directement.
*   **Utilité :** Transaction Cost Analysis (TCA), benchmark d'exécution.

### 8.3 Carnet d'Ordres (Level 2 / Level 3)
*   **Données :** Profondeur du marché (bids/asks à plusieurs niveaux).
*   **Utilité :** Analyse de la liquidité, détection de murs d'ordres.

---

## 9. Fixed Income & Obligations
**Si extension vers les obligations corporate.**

### 9.1 Bond Pricing
*   **Couverture :** 3.2 millions d'instruments, 800+ contributeurs.
*   **Types :** Government, Corporate, Municipal, Convertibles.

### 9.2 Credit Spreads
*   **Concept :** Écart de rendement entre une obligation corporate et le taux sans risque.
*   **Utilité :** Identifier les obligations mal pricées (spread trop large = opportunité).

### 9.3 Yield Curves
*   **Couverture :** 70+ devises.
*   **Données :** Courbes bid/offer par secteur, notation, pays.
*   **Utilité :** Valorisation des positions illiquides, gestion du risque de taux.

---

## 10. Corporate Actions & M&A
**Fichier impacté :** Ajustements de prix dans `portfolio_reconstruction.py`.

### 10.1 Corporate Actions
*   **Types :** Dividendes, splits, spin-offs, rights issues, fusions.
*   **Utilité :** Ajuster automatiquement les prix historiques pour éviter les sauts artificiels.

### 10.2 M&A Database
*   **Couverture :** 1.51 million de deals depuis les années 1970.
*   **Données :** Target, Acquirer, deal value, advisors, deal terms (1000+ champs par deal).
*   **Utilité :** Screening de cibles potentielles, analyse de primes d'acquisition.

### 10.3 IPO / Equity New Issues
*   **Couverture :** 462,000 deals depuis 1970.
*   **Données :** Prix d'offre, underwriters, use of proceeds.
*   **Utilité :** Analyse de performance post-IPO.

---

## 11. Alternative Data
**Data non traditionnelle pour signaux alpha.**

### 11.1 Job Postings (via LinkUp)
*   **Concept :** Nombre d'offres d'emploi publiées/retirées par une entreprise.
*   **Signal :** Hausse des offres = expansion prévue (bullish). Baisse = restructuration (bearish).

### 11.2 Patent Data
*   **Concept :** Nouvelles demandes de brevets par entreprise.
*   **Signal :** Indicateur avancé d'innovation et de R&D.

### 11.3 Satellite Imagery (via partenaires)
*   **Concept :** Images de parkings de supermarchés, niveaux de stockage de pétrole, etc.
*   **Signal :** Proxy en temps réel du chiffre d'affaires avant publication des résultats.

### 11.4 Web Traffic (via partenaires)
*   **Concept :** Données de trafic web sur les sites des entreprises.
*   **Signal :** Corrélation avec l'activité commerciale (e-commerce).

---

## 12. ESG & Investissement Responsable
**Fichier impacté :** `screening`, `reporting`.

### 12.1 ESG Scores
*   **Couverture :** 80% de la capitalisation mondiale, 450+ métriques.
*   **Historique :** Depuis 2002.
*   **Composantes :** E (Environnement), S (Social), G (Gouvernance).

### 12.2 Carbon Footprint
*   **Données :** Émissions CO2 Scope 1, 2, 3.
*   **Utilité :** Calculer l'empreinte carbone du portefeuille.

### 12.3 Controversy Flags
*   **Concept :** Alertes sur les entreprises impliquées dans des scandales (environnementaux, sociaux, corruption).
*   **Utilité :** Exclusion des entreprises à risque réputationnel.

### 12.4 EU Taxonomy Alignment
*   **Concept :** % du chiffre d'affaires aligné avec les objectifs de durabilité européens.
*   **Utilité :** Conformité réglementaire pour les fonds "Article 8" ou "Article 9".

---

## Synthèse : Matrice de Priorisation

| Fonctionnalité | Impact sur ton outil | Effort d'intégration | Priorité |
|---------------|---------------------|---------------------|----------|
| IBES Estimates | ⭐⭐⭐⭐⭐ | Moyen | **P0** |
| SmartEstimates | ⭐⭐⭐⭐ | Faible | P1 |
| StarMine Credit Risk | ⭐⭐⭐⭐ | Moyen | P1 |
| Ownership / Insider | ⭐⭐⭐ | Moyen | P2 |
| News Sentiment | ⭐⭐⭐⭐ | Élevé | P2 |
| ESG Scores | ⭐⭐⭐ | Faible | P2 |
| Tick History | ⭐⭐ | Très élevé | P3 |
| Alternative Data | ⭐⭐⭐ | Élevé | P3 |

---

## Prochaines Étapes Recommandées
1.  **Obtenir l'accès Refinitiv :** Contacter LSEG pour un compte API (ou utiliser un terminal Workspace existant).
2.  **POC sur IBES :** Modifier `DataFetcher.fetch()` pour récupérer les estimations forward au lieu de la croissance historique.
3.  **Tester SmartEstimates :** Intégrer le "Predicted Surprise" comme filtre dans le screener.
4.  **Ajouter le Credit Risk :** Créer une contrainte d'allocation max basée sur `TR.ProbabilityOfDefault1Y`.

---

**Verdict Final :** Le passage à Refinitiv transforme le projet d'une "calculatrice financière" à un "système d'aide à la décision" capable de justifier des investissements selon les mêmes standards que ceux utilisés en salle de marché professionnelle.
