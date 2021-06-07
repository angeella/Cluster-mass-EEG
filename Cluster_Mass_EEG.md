We will explain and apply in R the **Permutation-Based Cluster-Mass** method proposed by [Maris and Oostenveld, 2007](https://doi.org/10.1016/j.jneumeth.2007.03.024) and developed in R by [Frossard and Renaud, 2018](https://cran.r-project.org/web/packages/permuco/vignettes/permuco_tutorial.pdf), using EEG data. The Cluster-Mass is computed considering the time series of one channel (**Temporal Cluster-Mass**) and the time series of multiple channels (**Spatial-Temporal Cluster-Mass**). Finally the **All-Resolution Inference** from [Rosenblatt et al. 2018](https://doi.org/10.1016/j.neuroimage.2018.07.060) is applied in order to compute the lower bound for the true discovery proportion inside the clusters computed.

# Packages

First of all, you need to install and load the following packages:

```{r 1, warning=FALSE,message=FALSE}
#devtools::install_github("angeella/ARIeeg")
#devtools::install_github("bnicenboim/eeguana")
#devtools::install_github("jaromilfrossard/permuco")
library(ARIeeg)
library(dplyr)
library(eeguana)
library(ggplot2)
library(tidyr)
library(purrr)
library(abind)
library(permuco4brain)
library(permuco)
library(hommel)
library(plotly)
library(tidyverse)
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

```{r 2}
load(system.file("extdata", "data_eeg_emotion.RData", package = "ARIeeg"))
```

We transform the data as ```eeg_lst``` class object from the package ```eeguana```:

```{r 3}
data = utilsTOlst(data=dati)
is_eeg_lst(data)
```

and we drop off the final $5$ channels: 

```{r 4}
chan_to_rm <- c("RM"  ,  "EOGvo" ,"EOGvu"
                , "EOGhl", "EOGhr")
data <- 
  data %>%
  select(-one_of(chan_to_rm))
```

Finally, we segment the data and select two conditions, i.e., **disgust face** and **object**:

```{r 5, warning=FALSE,message=FALSE}
data_seg <- data %>%
  eeg_segment(.description %in% c(3,5),
              .lim = c(min(dati$timings$time), max(dati$timings$time))
  ) %>% eeg_baseline()  %>%
  mutate(
    condition =
      description
  ) %>%
  select(-c(type,description))
```

Some plot to understand the global mean difference between the two conditions:

```{r}

A<-data_seg %>%
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
  theme(legend.position = "bottom")+
  scale_color_manual(labels = c("Disgust", "Object"), values = c("#80bfff", "#ff8080"))
ggplotly(A)
```

# Theory

## Multiple testing problem?

The aim is to test if the difference of brain signal during the two conditions is different from $0$ for each time points, i.e., $500$. If the full set of channels is considered, we have also test for each channel, i.e., $27$, returning a total number of tests equals $500 \cdot 27$. Therefore, we have $500$ or $500 \cdot 27$ statistical tests to perform at group-level, so considering the **random subject effect**. The multiple testing problem is then obvious, and correction methods as Bonferroni or similar don't capture the time(-spatial) correlation structure of the statistical tests, the cluster mass method, proposed by [Maris and Oostenveld, 2007](https://doi.org/10.1016/j.jneumeth.2007.03.024), is then used. It is based on **permutation theory**, and it gains some power respect to other procedure correcting at level of (spatial-)temporal cluster instead of at level of single tests. It is similar to the cluster mass in the fMRI framework, but in this case, the *voxels*, i.e., the single object of the analysis, are expressed in terms of time-points or in terms of combination time-points/channels. The method is then able to gain some power respect to some traditional conservative FWER correction method exploiting the (spatial-)temporal structure of the data.

## Repeated Measures Anova Model

The cluster mass method is based on the **Repeated Measures Anova**, i.e.,

$$
y = 1_{N \times 1} \mu +  \eta X^{\eta} + \pi X^{\pi} +  \eta \pi X^{\eta \pi} + \epsilon
$$

where $1_{N \times 1}$ is a matrix with ones and

  1. $\mu$ is the **intercept**;
  2. $y \in \mathbb{R}^{N \times 1}$ is the response variables, i.e., the **signal**, in our case $N = n_{subj} \times n_{stimuli} = 40$;
  3. $X^{\eta} \in \mathbb{R}^{N \times n_{stimuli}}$ is the **design matrix** describing the **fixed effect** regarding the stimuli, and $\eta \in \mathbb{R}^{n_{stimuli} \times 1}$ the corresponding parameter of interest;
  4. $X^{\pi} \in \mathbb{R}^{N \times n_{subj}}$ is the **design matrix** describing the **random effect** regarding the subjects, and $\pi \in \mathbb{R}^{n_{subj} \times 1}$ the corresponding parameter.
  5. $X^{\eta \pi}$ is the **design matrix** describing the **interaction effects** between subjects and conditions;
  6. $\epsilon \in \mathbb{R}^{N \times 1}$ is the **error term** with $0$ mean and variance $\sigma^2 I_N$.

Therefore, $y \sim (1\mu + X^{\eta} \eta, \Sigma)$, $\pi \sim (0, \sigma^2_{\pi} I_{nsubj})$ and $\eta \pi \sim (0,\text{cov}(\eta \pi))$.

We want to make inference on $\eta$, such that $H_0: \eta = 0$ vs $H_1: \eta \ne 0$. We do that using the **F statistic**, i.e.,

$$
F = \dfrac{y^\top H_{X^{\eta}} y / (n_{stimuli} - 1)}{ y^\top H_{X^{\eta \pi}}y/(n_{stimuli} -1)(n_{subj} -1)} 
$$
where $H_{X}$ is the **projection matrix**, i.e., $H_{X} = X(X^\top X)^{-1} X^\top$. In order to compute this test, we use an alternative definition of $F$ based on the residuals:

$$
F_r = \dfrac{r^\top H_{X^{\eta}} r / (n_{stimuli} - 1)}{ r^\top H_{X^{\eta \pi}}r/(n_{stimuli} -1)(n_{subj} -1)} 
$$

where $r = (H_{X^{\eta}} + H_{X^{\eta\pi}})y$. For further details, see [Kherad Pajouh and Renaud, 2014](https://link.springer.com/article/10.1007/s00362-014-0617-3).

So, let the group of permutation, including the identity transformation, $\mathcal{P}$, we use $r^\star = P r$, where $P \in \mathcal{P}$ to compute the null distribution of our test, i.e., $\mathcal{R}$, and then the p-value, i.e.,

$$
\text{p-value} = \dfrac{1}{B} \sum_{F^\star_r \in \mathcal{R}} \mathbb{I}(|F^\star_r| \ge |F_r|)
$$

if the two-tailed is considered, where $F^\star_r = f(r^\star)$.

We have this model for each time point $t \in \{1, \dots, 500\}$ and each channel, so finally we will have $n_{\text{time-points}} \times n_{\text{channels}}$ statistical tests/p-values (raw).

## Temporal Cluster mass

This method has been proposed by [Maris and Oostenveld, 2007](https://doi.org/10.1016/j.jneumeth.2007.03.024) and is commonly implemented
in specialised software of EEG data analysis. It relies on a continuity argument that implies that an effect will appear into clusters of adjacent timeframes. Based on all time-specific statistics, we form these clusters using a threshold $\tau$ as follows

<div style="text-align:center" markdown="1">
![Example of cluster mass EEG from [Frossard, 2019](10.13097/archive-ouverte/unige:125617)](Image/clusterMass.JPG)
</div>

All the adjacent time points for which the statistics are above this threshold define one cluster $C_i$ for $i \in \{1, \dots, n_C\}$, where $n_C$ is the number of clusters found. We assign to each time point in the same cluster $C_i$, the same cluster-mass statistic $m_i = f(C_i)$ where $f$ is a function that aggregates the statistics of the whole cluster into a scalar; typically the sum of the $F$ statistics or the sum of squared of the $t$ statistics. The cluster-mass null distribution $\mathcal{M}$ is computed by repeating the process described above for each permutation. The contribution of a permutation to the cluster-mass null distribution is the maximum over all cluster-masses for this permutation. To test the significance of an observed cluster Ci, we compare its cluster-mass $m_i = f(C_i)$ with the cluster-mass null distribution $\mathcal{M}$. The p-value of the effect at each time within a cluster $C_i$ is the p value associated with this cluster, i.e. 

$$
p_i = \dfrac{1}{n_P} \sum_{m_i^\star \in \mathcal{M}} 1\{m_i^\star  \ge m_i\}
$$

where $m_i^\star$ is computed considering the permuted statistic. This method makes sense for EEG data analysis because if a difference of cerebral activity is believed to happen at a time $s$ for a given factor, it is very likely that the time $s + 1$ (or $s ??? 1$) will show this difference too.

## Spatial-temporal Cluster mass 

In this case, we will use the theory of graph, where the vertices represent the channels, and the edges represent the **adjacency** relationship. The adjacency must be defined using prior information, therefore the three-dimensional Euclidean distance between channels is used. Two channels are defined adjacent if their Euclidean distance is less than a threshold $\delta$, where $\delta$ is the smallest euclidean distance that produces a connected graph. This is due to the fact that a connected graph implies no disconnected sub-graph. Having sub-graphs implies that some tests cannot, by design, be in the same cluster, which is not a useful assumption for this analysis. ([Frossard and Renaud, 2018](https://cran.r-project.org/web/packages/permuco/vignettes/permuco_tutorial.pdf); [Frossard, 2019](10.13097/archive-ouverte/unige:125617)).

Then, having the spatial adjacency definition, we need to define the temporal one. We reproduce this graph $n_{\text{time-points}}$ times, the edges between all pairs of two vertices (tests) are associated with the same electrode when they are temporally adjacent. The final graph has a total of vertices equals to the number of tests ($n_{\text{channels}} \times n_{\text{time-points}}$). The following figure represents the case of $64$ channels and $3$ temporal measures:

<div style="text-align:center" markdown="1">
![Example of graph of adjacency from [Frossard, 2019](10.13097/archive-ouverte/unige:125617)](Image/cluster.JPG)
</div>

We then delete all the vertices in which statistics are below a threshold, e.g., the $95$ percentile of the null distribution of the $F$ statistics. So, we have a new graph composed of **multiple connected components**. Then, each connected component is interpreted as a spatial-temporal cluster. Finally, for each connected component, we compute the cluster-mass statistic using the sum (or sum of squares) of statistics of that particular connected component.

The cluster-mass null distribution is computed by permutations while maintaining spatial-temporal correlations among tests. Permutations must be performed without changing the position of electrodes nor mixing time-points. Concretely, after transforming the responses using the permutation method in [Kherad Pajouh and Renaud, 2014](https://link.springer.com/article/10.1007/s00362-014-0617-3), they are sorted in a three-dimensional array. It has the design (subjects $\times$ stimuli) in the first dimension, time in the second one and electrodes in the third one. Then, only the first dimension is permuted to create a re-sampled response (or 3D array). Doing so, it does not reorder time-points, neither electrodes, therefore, the spatial-temporal correlations are maintained within each permuted sample.

# Application

In R, all of this is possible thanks to the permuco and permuco4brain packages developed by [Frossard and Renaud, 2018](https://cran.r-project.org/web/packages/permuco/vignettes/permuco_tutorial.pdf).

## Temporal Cluster-Mass

So, we select one channel from our dataset, e.g. the Fp1:

```{r}
Fp1 <- data_seg %>% select(Fp1)
```

1. Construct the $y$. We need to construct the three-dimensional **signal matrix**, having dimensions $40 \times 500$:

```{r}
signal_Fp1 <- Fp1%>%
    signal_tbl()%>%
    group_by(.id)%>%
    nest()%>%
    mutate(data = map(data,~as.matrix(.x[-1])))%>%
    pull(data)%>%
    invoke(abind,.,along = 2)%>%
    aperm(c(2,1))
dim(signal_Fp1)
```

2. Construct the $X_{\eta \pi}$, having dimensions $40 \times 2$:

```{r}
design <- 
  segments_tbl(Fp1)%>%
  select(.subj, condition)
dim(design)
```

3. Define the **repeated measures ANOVA formula**: 

```{r}
f <- signal_Fp1 ~ condition + Error(.subj/(condition))
```

Thanks to the permuco package, we can apply the temporal cluster-Mass for the channel Fp1:

```{r}

lm_Fp1 <- clusterlm(f,data = design)
print(lm_Fp1)
```

and the corresponding plot:

```{r}
plot(lm_Fp1)
```

### ARI in EEG cluster mass

However, our significant cluster says only that at least one test is different from $0$, we don't know how many tests/time-points are significant (**spatial specificity paradox**). So, we can apply ARI to understand the lower bound of the number of true discovery proportion. The cluster is composed by the time points from $161$ to $246$, i.e., the size of the cluster is equal to $86$.

```{r}
praw <- lm_Fp1$multiple_comparison$condition$uncorrected$main[,2]
cluster <- c(161:246)

discoveries(hommel(praw), ix = cluster)
```

Therefore, we have at least $62\%$ of true active time points in the cluster computed.

## Spatial-Temporal Cluster-Mass

1. Construct the $y$. We need to construct the three-dimensional **signal array**, having dimensions $40 \times 500 \times 27$:

```{r 7}
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

```{r 8}
design <- 
  segments_tbl(data_seg)%>%
  select(.subj, condition)
dim(design)
```

3. Construct the **graph**, using $\delta = 53mm$:

```{r fig.align="center"}
graph <- position_to_graph(channels_tbl(data_seg), name = .channel, delta = 53,
                             x = .x, y = .y, z = .z)
plot(graph)
```

4. Define the **repeated measures ANOVA formula**: 

```{r 10}
f <- signal ~ condition + Error(.subj/(condition))
```

Finally, run the main function:

```{r}
model <- permuco4brain::brainperm(formula = f,
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

We have only one significant cluster (32), with p-value equals to $0.0002$. It is composed by $27$ channels (the total set), with main channels P7. You can see in details the components of this cluster in

```{r}
names(model$multiple_comparison$condition$clustermass$cluster$membership[which(as.vector(model$multiple_comparison$condition$clustermass$cluster$membership)==32)])
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

### ARI in EEG cluster mass

However, our significant cluster (11) says only that at least one combination channels/time-points is different from $0$, we don't know how many combinations are significant (**spatial specificity paradox**). So, we can apply ARI to understand the lower bound of the number of true discovery proportion:

```{r}
ARIeeg(model = model)
```

So, we have at least $15\%$ truly active component in the cluster $32$.

# References

 - Maris, E., & Oostenveld, R. (2007). Nonparametric statistical testing of EEG-and MEG-data. Journal of neuroscience methods, 164(1), 177-190.

 - Kherad-Pajouh, S., & Renaud, O. (2015). A general permutation approach for analyzing repeated measures ANOVA and mixed-model designs. Statistical Papers, 56(4), 947-967.
 
 - Frossard, J. (2019). Permutation tests and multiple comparisons in the linear models and mixed linear models, with extension to experiments using electroencephalography. DOI: 10.13097/archive-ouverte/unige:125617.
 
 - Frossard, J. & O. Renaud (2018). Permuco: Permutation Tests for Regression, (Repeated Measures) ANOVA/ANCOVA and Comparison of Signals. R Packages.





