
We will explain and apply in R the **Permutation-Based Cluster-Mass** method proposed by [Maris and Oostenveld, 2007](https://doi.org/10.1016/j.jneumeth.2007.03.024) and developed in R by [Frossard and Renaud, 2018](https://cran.r-project.org/web/packages/permuco/vignettes/permuco_tutorial.pdf), using EEG data. Finally the **All-Resolution Inference** from [Rosenblatt et al. 2018](https://doi.org/10.1016/j.neuroimage.2018.07.060) is applied in order to compute the lower bound for the true discovery proportion inside the clusters computed.

# Packages

First of all, you need to install and load the following packages:

```{r}
devtools::install_github("angeella/ARIeeg")
devtools::install_github("bnicenboim/eeguana")
devtools::install_github("jaromilfrossard/permuco")
library(ARIeeg)
library(dplyr)
library(eeguana)
library(ggplot2)
library(tidyr)
library(purrr)
library(abind)
library(permuco4brain)
library(hommel)
```

# Data

The Dataset from the package ```ARIeeg``` is an **ERP experiment** composed by:

- 20 Subjects,
- 32 Channels
- Stimuli: pictures. Conditions:
    1. (f): fear (face)
    2. (h): happiness (face)
    3. (d): disgust (face)
    4. (n): neutral (face)
    5. (o): object

We have one observation for each subject and each stimulus. You can load it using:

```{r}
load(system.file("extdata", "data_eeg_emotion.RData", package = "ARIeeg"))
```

We transform the data as ```eeg_lst``` class object from the package ```eeguana```:

```{r}
data = utilsTOlst(data=dati)
is_eeg_lst(data)
```

and we drop off the final $5$ channels: 

```{r}
chan_to_rm <- c("RM"  ,  "EOGvo" ,"EOGvu"
                , "EOGhl", "EOGhr")
data <- 
  data %>%
  select(-one_of(chan_to_rm))
```

Finally, we segment the data and select two conditions, i.e., **disgust face** and **object**:

```{r}
data_seg <- data %>%
  eeg_segment(.description %in% c(3,5),
              lim = c(min(dati$timings$time), max(dati$timings$time))
  ) %>% eeg_baseline()  %>%
  mutate(
    condition =
      description
  ) %>%
  select(-c(type,description))
```

Some plot to understand the global mean difference between the two conditions:

```{r}

data_seg %>%
  select(Fp1,Fp2, F3, F4) %>%
  ggplot(aes(x = .time, y = .value)) +
  geom_line(aes(group = condition))  +
  stat_summary(
    fun = "mean", geom = "line", alpha = 1, size = 1.5,
    aes(color = condition),show.legend = TRUE
  ) +
  facet_wrap(~.key) +
  geom_vline(xintercept = 0, linetype = "dashed") +
  geom_vline(xintercept = .17, linetype = "dotted") +
  theme(legend.position = "bottom")
```

# Analysis

## Multiple testing problem?

The aim is to test if the difference of brain signal during the two conditions is different from $0$ for each time points, i.e., $500$, and for each channel, i.e., $27$. Therefore, we have $500 \cdot 27$ statistical tests to perform at group-level, so considering the **random subject effect**. The multiple testing problem is then obvious, and correction methods as Bonferroni or similar don't capture the time-spatial correlation structure of the statistical tests, the cluster mass method, proposed by [Maris and Oostenveld, 2007](https://doi.org/10.1016/j.jneumeth.2007.03.024), is then used. It is based on **permutation theory**, and it gains some power respect to other procedure correcting at level of spatial-temporal cluster instead of at level of single tests. It is similar to the cluster mass in the fMRI framework, but in this case, the *voxels*, i.e., the single object of the analysis, are expressed in terms of combination time-points/channels. The method is then able to gain some power respect to some traditional conservative FWER correction method exploiting the spatial-temporal structure of the data.

## Repeated Measures Anova Model

The cluster mass method is based on the **Repeated Measures Anova**, i.e.,

$$
y 
$$

where $1_{N \times 1}$ is a matrix with ones and

  1. $\mu$ is the **intercept**;
  2. $y \in \mathbb{R}^{N \times 1}$ is the response variables, i.e., the **signal**, in our case $N = n_{subj} \times n_{stimuli} = 40$;
  3. $X_{\eta} \in \mathbb{R}^{N \times n_{stimuli}}$ is the **design matrix** describing the **fixed effect** regarding the stimuli, and $\eta \in \mathbb{R}^{n_{stimuli} \times 1}$ the corresponding parameter of interest;
  4. $X_{\pi} \in \mathbb{R}^{N \times n_{subj}}$ is the **design matrix** describing the **random effect** regarding the subjects, and $\pi \in \mathbb{R}^{n_{subj} \times 1}$ the corresponding parameter.
  5. $X_{\eta \pi}$ is the **design matrix** describing the **interaction effects** between subjects and conditions;
  6. $\epsilon \in \mathbb{R}^{N \times 1}$ is the **error term** with $0$ mean and variance $\sigma^2 I_N$.

Therefore, $y \sim (\mathbb{1}\mu + X_{\eta} \eta, \Sigma)$, $\pi \sim (0, \sigma^2_{\pi} I_{nsubj})$ and $\eta \pi \sim (0,\text{cov}(\eta \pi))$.

We want to make inference on $\eta$, such that $H_0: \eta = 0$ vs $H_1: \eta \ne 0$. We do that using the **F statistic**, i.e.,

$$
F = \dfrac{y^\top H_{X_{\eta}} y / (n_{stimuli} - 1)}{ y^\top H_{X_{\eta \pi}}y/(n_{stimuli} -1)(n_{subj} -1)} 
$$
where $H_{X}$ is the **projection matrix**, i.e., $H_{X} = X(X^\top X)^{-1} X^\top$. In order to compute this test, we use an alternative definition of $F$ based on the residuals:

$$
F = \dfrac{r^\top H_{X_{\eta}} r / (n_{stimuli} - 1)}{ r^\top H_{X_{\eta \pi}}r/(n_{stimuli} -1)(n_{subj} -1)} 
$$

where $r = (H_{X_{\eta}} + H_{X_{\eta\pi}})y$. For further details, see @Kherad.

So, let the group of permutation, including the identity transformation, $\mathcal{P}$, we use $r^\star = P r$, where $P \in \mathcal{P}$ to compute the null distribution of our test $\mathcal{R}$ and then the p-value, i.e.,

$$
\text{p-value} = \dfrac{1}{B} \sum_{r^\star_b \in \mathcal{R}} \mathbb{I}(|r^\star_b| \ge |r|)
$$

if the two-tailed is considered.

We have this model for each time point $t \in \{1, \dots, 500\}$ and each channel, so finally we will have $n_{\text{time-points}} \times n_{\text{channels}}$ statistical tests/p-values (raw).

## Spatio-temporal Cluster mass 

Then, we need to construct the **spatial-temporal clusters** in order to correct the raw p-values for the FWER. In this case, we will use the theory of graph, where the vertices represent the channels, and the edges represent the **adjacency** relationship. The adjacency must be defined using prior information, therefore the three-dimensional Euclidean distance between channels is used. Two channels are defined adjacent if their Euclidean distance is less than a threshold $\delta$, where $\delta$ is the smallest euclidean distance that produces a connected graph. This is due to the fact that a connected graph implies no disconnected sub-graph. Having sub-graphs implies that some tests cannot, by design, be in the same cluster, which is not a useful assumption for this analysis. ([Frossard and Renaud, 2018](https://cran.r-project.org/web/packages/permuco/vignettes/permuco_tutorial.pdf); [Frossard, 2019](10.13097/archive-ouverte/unige:125617)).

Then, having the spatial adjacency definition, we need to define the temporal one. We reproduce this graph $n_{\text{time-points}}$ times, the edges between all pairs of two vertices (tests) are associated with the same electrode when they are temporally adjacent. The final graph has a total of vertices equals to the number of tests ($n_{\text{channels}} \times n_{\text{time-points}}$). The following figure represents the case of $64$ channels and $3$ temporal measures:

![Example of graph of adjacency from [Frossard, 2019](10.13097/archive-ouverte/unige:125617)](Image/cluster.JPG)

We then delete all the vertices in which statistics are below a threshold, e.g., the $95$ percentile of the null distribution of the $F$ statistics. So, we have a new graph composed of **multiple connected components**. Then, each connected component is interpreted as a spatiotemporal cluster. Finally, for each connected component, we compute the cluster-mass statistic using the sum (or sum of squares) of statistics of that particular connected component.

The cluster-mass null distribution is computed by permutations while maintaining spatio-temporal correlations among tests. Permutations must be performed without changing the position of electrodes nor mixing time-points. Concretely, after transforming the responses using the permutation method in [Kherad Pajouh and Renaud, 2014](https://link.springer.com/article/10.1007/s00362-014-0617-3), they are sorted in a three-dimensional array. It has the design (subjects $\times$ stimuli) in the first dimension, time in the second one and electrodes in the third one. Then, only the first dimension is permuted to create a re-sampled response (or 3D array). Doing so, it does not reorder time-points, neither electrodes, therefore, the spatiotemporal correlations are maintained within each permuted sample.

# Permuco4brain

In R, all of this is possible thanks to the permuco and permuco4brain packages developed by [Frossard and Renaud, 2018](https://cran.r-project.org/web/packages/permuco/vignettes/permuco_tutorial.pdf).

1. Construct the $y$. We need to construct the three-dimensional **signal array**, having dimensions $40 \times 500 \times 27$:

```{r}
signal <- 
    data_seg%>%
    signal_tbl()%>%
    group_by(.id)%>%
    nest()%>%
    mutate(data = map(data,~as.matrix(.x[-1])))%>%
    pull(data)%>%
    invoke(abind,.,along = 3)%>%
    aperm(c(3,1,2))

dim(signal)
```

2. Construct the $X_{\eta \pi}$:

```{r}
design <- 
  segments_tbl(data_seg)%>%
  select(.subj, condition)
dim(design)
```

3. Construct the **graph**, using $\delta = 53mm$:

```{r}
graph <- position_to_graph(channels_tbl(data_seg), name = .channel, delta = 53,
                             x = .x, y = .y, z = .z)
plot(graph)
```

4. Define the **repeated measures ANOVA formula**: 

```{r}
formula <- signal ~ condition + Error(.subj/(condition))
```

Finally, run the main function:

```{r}
model <- permuco4brain::brainperm(formula = formula,
                                  data = design,
                                  graph = graph,
                                  np = 5000,
                                  multcomp = "clustermass",
                                  return_distribution = TRUE)
```

where np indicates the number of permutation.

Then, we can analyze the output:

```{r}
print(model)
```

We have only one significant cluster (32), with p-value equals to $0.0002$ and cluster mass equals to $56102.363567$. It is composed by $27$ channels (the total set), with main channels P8. You can see in details the components of this cluster in

```{r}
names(model$multiple_comparison$condition$clustermass$cluster$membership[which(as.vector(model$multiple_comparison$condition$clustermass$cluster$membership)==11)])
```

You can see the significant cluster (in red) at fixed time points (e.g. 160) using plot:

```{r}
plot(model, samples = 160)
```

and the significant cluster over time and over channels using:

```{r}
image(model)
```

where the significant clusters are represented in a colour-scale and the non-significant one in grey. The white pixels are tests which statistic are below the threshold.

# ARI in EEG cluster mass

However, our significant cluster (11) says only that at least one combination channels/time-points is different from $0$, we don't know how many combinations are significant (**spatial specificity paradox**). So, we can apply ARI to understand the lower bound of the number of true discovery proportion:

```{r}
ARIeeg(model = model)
```

So, we have at least $16.58\%$ truly active component in the cluster $32$.

# References

 - Maris, E., & Oostenveld, R. (2007). Nonparametric statistical testing of EEG-and MEG-data. Journal of neuroscience methods, 164(1), 177-190.

 - Kherad-Pajouh, S., & Renaud, O. (2015). A general permutation approach for analyzing repeated measures ANOVA and mixed-model designs. Statistical Papers, 56(4), 947-967.
 
 - Frossard, J. (2019). Permutation tests and multiple comparisons in the linear models and mixed linear models, with extension to experiments using electroencephalography. DOI: 10.13097/archive-ouverte/unige:125617.
 
 - Frossard, J. & O. Renaud (2018). Permuco: Permutation Tests for Regression, (Repeated Measures) ANOVA/ANCOVA and Comparison of Signals. R Packages.


