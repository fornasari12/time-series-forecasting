# Forecasting with the Temporal Fusion Transformer

![img.png](img.png)

Multi-horizon forecasting often contains a complex mix of inputs – including
static (i.e. time-invariant) covariates, known future inputs, and other exogenous
time series that are only observed in the past – without any prior information
on how they interact with the target. Several deep learning methods have been
proposed, but they are typically ‘black-box’ models which do not shed light on
how they use the full range of inputs present in practical scenarios. In this paper, we introduce the Temporal Fusion Transformer (TFT) – a novel attentionbased architecture which combines high-performance multi-horizon forecasting
with interpretable insights into temporal dynamics. To learn temporal relationships at different scales, TFT uses recurrent layers for local processing and
interpretable self-attention layers for long-term dependencies. TFT utilizes specialized components to select relevant features and a series of gating layers to
suppress unnecessary components, enabling high performance in a wide range of
scenarios. On a variety of real-world datasets, we demonstrate significant performance improvements over existing

* [PyTorch Tutorial - Temporal Fusion Transformers](https://pytorch-forecasting.readthedocs.io/en/latest/tutorials/stallion.html)
* [PyTorch Forecasting Announcement](https://towardsdatascience.com/introducing-pytorch-forecasting-64de99b9ef46#:~:text=What%20is%20PyTorch%20Forecasting%3F,easily%20trained%20with%20pandas%20dataframes.)
* [PyTorch Tutorial - TimeSeriesDataSet](https://pytorch-forecasting.readthedocs.io/en/latest/api/pytorch_forecasting.data.timeseries.TimeSeriesDataSet.html#timeseriesdataset)
* [Paper - Temporal Fusion Transformers
for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/pdf/1912.09363.pdf)
* [Flight Demand Forecasting with
Transformers](https://arxiv.org/pdf/2111.04471v1.pdf)


* Check:
  * UBER:
    * [UBER BLOG](https://eng.uber.com/neural-networks/)
    * [UBER PAPER](http://roseyu.com/time-series-workshop/submissions/TSW2017_paper_3.pdf)
    * [UBER EXAMPLE](https://forecasters.org/wp-content/uploads/gravity_forms/7-c6dd08fee7f0065037affb5b74fec20a/2017/07/Laptev_Nikolay_ISF2017.pdf)
    * [UBER EXAMPLE](https://conferences.oreilly.com/strata/strata-ca-2018/cdn.oreillystatic.com/en/assets/1/event/269/Detecting%20time%20series%20anomalies%20at%20Uber%20scale%20with%20recurrent%20neural%20networks%20Presentation.pdf)
    * [UBER M4](https://eng.uber.com/m4-forecasting-competition/)
  * [time-series-workshop](http://roseyu.com/time-series-workshop/)
  * [Large-Scale Unusual Time Series Detection](https://robjhyndman.com/papers/icdm2015.pdf)
  * [Mcompetitions code](https://github.com/Mcompetitions/M5-methods/tree/master/Code%20of%20Winning%20Methods)
  * [Ocado Demand Forecasting](https://www.ocadogroup.com/technology/blog/how-we-improve-forecasting-and-availability-neural-networks-and-deep-learning)
  * [Unit8](https://unit8.com/casestudies/)
  * [TimeSeries_CNN_Classification_Using_Grid_Representation](https://github.com/junyoung-jamong/TimeSeries_CNN_Classification_Using_Grid_Representation)
  
  
  