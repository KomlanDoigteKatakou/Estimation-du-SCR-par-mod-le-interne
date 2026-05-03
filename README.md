# Estimation du SCR par modèle interne

Etude mathématique et implémnetation de deux méthodes d'estimation du
Solvency Capital Requirement (Solvabilité II), dans le cadre du modèle
actif-passif de Bauer, Bergmann & Reuss (2010).

---

## Contenu

**`rapport.pdf`** — Analyse des simulations imbriquées (biais asymptotique,
optimisation Lagrangienne, complexité $O(\varepsilon^{-3})$) et de MLMC
(identité télescopique, conditions de Giles, échantillonnage adaptatif,
complexité théorique $O(\varepsilon^{-2}|\log\varepsilon|^2)$).

**`nested_mc_scr.cpp`** — Implémentation C++ des simulations imbriquées,
avec simulation exacte du modèle de Vasicek (Cholesky joint sur $(dW,
\int r\,du)$) et couplage correct des browniens actif/taux.
Quelques approximations sur les flux et l'actualisation subsistent — voir
le rapport. Résultat obtenu : $\widehat{\text{SCR}} \approx 1322$
(référence Bauer et al. : 1249,3 ; écart discuté dans le rapport).

---

## À venir

Implémentation MLMC — la théorie est dans le rapport, le code non encore
finalisé.

---

## Auteur

Komlan Katakou — M1 Mathématiques générales, Université Lyon 1 (2025–2026)
