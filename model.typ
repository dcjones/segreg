#set page(margin: 2.5cm)
#set par(justify: true)

Segreg (#strong[Seg]mentation-aware #strong[Reg]ression) is general purpose
system for modeling gene expression count data in the presence of uncertainty,
usually deriving from imprecise cell segmentation. This document will give a
little of the background and mathematical description of the model.


= Background: Probabilistic Cell Segmentation

The foundation and precursor to Segreg is Proseg (#strong[Pro]babilistic
#strong[Seg]mentation). Proseg is a cell segmentation method informed primarily by
the transcript data itself. It models gene expression on a per-cell basis, and
runs an MCMC sampler to estimate cell segmentation that best explains the
observed spatial distribution of transcripts.

What Proseg reports is (among other things, like cell boundary polygons), is a point
estimate of transcript counts for every gene and cell $X$, where $X_(c g)$ gives the
estimated number of transcripts of gene $g$ in cell $c$.

Because it is a sampler, we also have a great deal of flexibility when it comes
to estimating uncertainty. What Proseg reports is two matrices estimating
expected inflow and outflow, respectively. Inflow, for gene $g$ and cell $c$, is
given by $accent(f, arrow.l)_(c g)$ and outflow $accent(f, arrow.r)_(c
g)$. The former estimates the expected number of transcripts of gene $g$ that
are erroneously assigned to cell $c$ in the point estimate $X$. The latter, is
the expected number that belong to $c$ but are erroneously assigned to other
cells.

These are both estimated using the following procedure:
1. Proseg is run for some number of samples.
2. The current sampler state is recorded as the point estimate $X$.
3. The sampler is then run for further number of samples to estimate the inflow and outflow expectations.

In this second phase of sampling, we track how each transcript changes state during the sampling, effectively computing
the posterior probability of the transcript assignment conditioned on where it was assigned in the point estimate $X$.

With these two matrices, we have the tools necessary to build to more robustly
model the gene expression matrix reported in $X$.

= A Segmentation-Aware Regression Model

The basis of Segreg is a pretty straightforward Poisson log-linear regression model over
the gene expression point estimate $X$. The basic model we adopt is:
$ X_(c g) tilde.op text("Poisson")(lambda_(c g)) $
and
$ log(lambda_(c g)) = D_(c :) B_(: g) $
where $D$ is the design matrix and $B$ is the regression coefficient matrix.

We expect $X$ to be a reasonable estimate of the true gene expression matrix,
but also be contaminated because of genuine uncertainty about which transcripts
belong to which cells.

Towards that end, we posit that the observed counts $X$ are generated from a
true (latent) rate $lambda$ that is inflated by a known contamination term
$delta$. The contamination represents transcripts that are erroneously assigned
to a cell due to segmentation uncertainty. When net inflow is positive, the
observed expression is higher than the true rate would suggest; when net outflow
dominates, the observed expression is lower.
$ X_(c g) tilde.op text("Poisson")(lambda_(c g) + delta_(c, g)) $

Since $delta$ represents the aggregate expected error, we plug in the estimates
from Proseg's sampling procedure, taking the further step to clamp values to be
non-negative to avoid numerical issues.
$ delta_(c, g) = max(0, accent(f, arrow.l)_(c g) - accent(f, arrow.r)_(c g)) $

// TODO: I don't love the lambda notation for quantities that proseg estimates.

= Segmentation error as a parameter

We can further elaborate on this model by instead of using a fixed estimate of $delta_(c g)$,
estimating it as a model parameter

= Amortized Inference

TODO: Explain the variational inference procedure used to fit the model.
