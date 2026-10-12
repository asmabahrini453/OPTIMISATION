# Projet d'optimisation : LASSO, SVM et EM (mélange de gaussiennes)

**Auteur :** [Votre nom] — [Formation / numéro étudiant]
**Cours :** [Intitulé du cours] — [Année universitaire]

## Objet du projet

Ce projet étudie trois problèmes d'optimisation à partir de la lecture 4 et de la lecture 19 d'EE227C :

| Projet | Problème | Méthodes étudiées |
|---|---|---|
| 2 | LASSO (moindres carrés + pénalité ℓ1) | sous-gradient, gradient stochastique par mini-batch, ISTA / FISTA |
| 3 | SVM linéaire (perte charnière + pénalité ℓ2) | sous-gradient, gradient stochastique par mini-batch, SDCA (méthode duale) |
| 6 | Mélange de gaussiennes | algorithme EM, variante CEM, maximisation directe par L-BFGS |

Pour chaque problème, je ne me contente pas de décrire le code : j'explique le problème posé, le fonctionnement des méthodes, leur fiabilité selon les réglages (pas, taille de mini-batch, initialisation, critère d'arrêt, paramètre de régularisation), leurs limites et des alternatives. Les erreurs que j'ai repérées dans les notebooks d'origine sont signalées et vérifiées numériquement.

## Contenu du dossier

| Fichier | Contenu |
|---|---|
| `Projet_Optimisation_notebook1_LASSO.ipynb` | Projet 2 : notebook LASSO commenté + expériences E1 à E5 + synthèse |
| `Projet_Optimisation_notebook2_SVM.ipynb` | Projet 3 : notebook SVM commenté + expériences E1 à E5 + synthèse ; contient aussi les sections *Sparse Inverse Covariance Estimation* et *Gradient madness* de la même lecture, commentées (sans expérience supplémentaire) |
| `Projet_Optimisation_notebook3_EM.ipynb` | Projet 6 : notebook EM commenté + expériences E1 à E7 + synthèse |
| `README.md` | Ce fichier |

## Comment lire un notebook

Chaque notebook garde le texte et le code d'origine, auxquels j'ai ajouté :

- des **commentaires dans le code** (`# Attention :` quand j'ai repéré un défaut) ;
- des blocs **« Ce que j'ai compris »** (ce que fait la cellule et pourquoi), **« Critique »** (erreurs, limites, fragilités) et **« Interprétation / Résultats mesurés »** (ce que montrent les chiffres) ;
- des **cellules de vérification** qui recalculent les valeurs citées dans les blocs ;
- une partie **expériences** (E1, E2, …) avec le paramètre modifié, ce qui reste constant et les métriques ;
- une **synthèse** : tableau récapitulatif, applicabilité, fiabilité, limites, amélioration à retenir, questions possibles du professeur.

Ce qui n'est pas dans les documents du cours est signalé par **(complément)**.

## Installation et exécution

**Environnement :** Python 3 (notebooks exécutés avec Python 3.13) et Jupyter.

```bash
pip install numpy scipy pandas matplotlib scikit-learn jupyter
pip install autograd        # seulement pour les sections « Sparse Inverse Covariance » et « Gradient madness » du notebook SVM
```

**Lancer :**

```bash
jupyter notebook
```

puis ouvrir un notebook et faire *Run All*.

### Points importants

1. **Exécuter les cellules dans l'ordre.** Les notebooks d'origine redéfinissent les variables globales `A`, `b`, `m`, `n`, `x_opt`, `objective` et `gradient` dans presque chaque section. Exécuter une cellule dans le désordre donne des erreurs ou des figures fausses (par exemple une ligne « optimal » à 500 au lieu de 5).
2. **Les sections *Sparse Inverse Covariance* et *Gradient madness*** exécutent `import autograd.numpy as np`, qui remplace `np` pour la suite du notebook. Elles sont placées à la fin du notebook SVM pour cette raison. Leurs sorties sont celles du notebook d'origine ; je n'ai pas relancé ces cellules.
3. **Reproductibilité :** les expériences utilisent des graines fixées et des générateurs aléatoires locaux (`default_rng`) pour ne pas modifier la suite aléatoire du notebook d'origine. Les chiffres cités dans les blocs viennent de mes exécutions ; une autre version des bibliothèques peut donner de petits écarts.
4. **Durée :** les expériences demandent du temps. Le notebook EM prend environ 15 à 25 minutes en entier ; le temps de calcul des autres notebooks dépend de la machine.

## Principaux résultats

*(Valeurs mesurées sur les instances simulées décrites dans chaque notebook ; voir les notebooks pour les conditions exactes.)*

### Projet 2 : LASSO (m = 100, n = 1000, α = 0,1, sans bruit)
- Le sous-gradient du notebook (pas constant 0,1) **ne converge pas** : plateau de f − f* ≈ 4,9 (médiane sur 10 instances). Le pas 0,2 diverge ; le meilleur pas testé (0,01) laisse encore f − f* ≈ 0,1.
- **ISTA et FISTA** atteignent la précision machine ; ISTA atteint 10⁻³ en ≈ 680 itérations, FISTA en ≈ 127. Le résultat est vérifié contre scikit-learn.
- Le **résidu proximal** sert de critère d'arrêt ; la norme du sous-gradient ne s'annule pas à l'optimum.
- Le gradient stochastique n'apporte rien sur ce petit problème (m = 100).

### Projet 3 : SVM (m = 1000, n = 100, α = 0,05)
- Le sous-gradient du notebook d'origine **somme** les pertes au lieu de les moyenner (facteur m), et la courbe est tracée avec un autre α (0,1) que celui de l'optimisation (0,05). Ces deux points sont vérifiés par différences finies et par comparaison avec SDCA et scikit-learn.
- Avec le sous-gradient corrigé, un pas bien choisi donne f − f* de l'ordre de 10⁻⁴ à 10⁻⁵ ; le pas trop petit est le principal danger.
- **SDCA** (méthode duale) est la plus fiable et fournit un **écart de dualité** qui sert de critère d'arrêt certifié.
- Le mini-batch n'améliore pas la précision à budget de passes égal sur ce problème.

### Projet 6 : EM (mélange de gaussiennes 1D)
- Mon EM, avec l'étape M corrigée, retrouve scikit-learn (même log-vraisemblance). L'étape M du notebook d'origine met q² au lieu de q dans la variance (σ sous-estimé d'environ 8 %), et la courbe « likelihood » du notebook n'est pas la log-vraisemblance.
- La vraisemblance **ne baisse jamais** avec l'EM corrigé (600 exécutions) ; avec l'étape M du notebook, elle baisse dans 76 % des exécutions testées.
- L'**initialisation** décide du résultat : k-means et quantiles réussissent dans 100 % des cas ; les points aléatoires échouent jusqu'à 10 % du temps ; une initialisation à moyennes égales échoue toujours.
- Plus les composantes se chevauchent, plus EM est lent ; le critère d'arrêt est trompeur dans ce cas.
- CEM est beaucoup plus rapide mais biaisé ; L-BFGS demande 4 à 5 fois moins d'évaluations mais réussit moins souvent depuis des points aléatoires (90 %).

## Limites

- Données **simulées** (matrices gaussiennes, mélanges de gaussiennes), pas de données réelles.
- Dimensions modestes ; peu d'instances (5 à 100 selon l'expérience) : les chiffres donnent des **ordres de grandeur**, pas des valeurs précises.
- Non testé : données bruitées ou corrélées, très grand nombre d'exemples (pour le mini-batch), choix du paramètre de régularisation par validation croisée, EM en dimension supérieure à 1, choix du nombre de composantes.
- Les deux sections *Sparse Inverse Covariance* et *Gradient madness* sont commentées sans expérience supplémentaire.

## Références

- Notes EE227C (Hardt) : chap. 2 « Gradient method », chap. 3 « Strong convexity », chap. 8 « Nesterov », chap. 10 « Stochastic optimization » (déf. 10.3), § 11.5 « Implicit regularization », chap. 15 « Fenchel duality » (§ 15.2, exemple 15.6), chap. 19 « Alternating minimization and EM » ; notebooks des lectures 4 et 19.
- Notre cours d'optimisation (LASSO, méthode proximale, sous-gradient, gradient stochastique, SVM et dualité) et les slides d'optimisation (mini-batch).
- J. Jacques, *Model-based clustering and classification, partie 1* (fichier `MBCC-part1.pdf`) : log-vraisemblance (p. 46), étapes E et M (p. 47-49), propriétés et initialisation (p. 50), convergence (p. 51-54), exercice 3 (p. 55), variantes CEM et SEM (p. 56-57).
