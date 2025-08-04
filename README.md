This project explores X-ray-based materials characterization using deep learning to classify aluminum alloys based on their absorption-driven spectral analysis. Building on prior work with YOLO, we focus on two aluminum materials—pure aluminum (Al 1050) and aluminum-copper alloy (Al 2017)—imaged with lab-based X-rays at 50 keV and 300 μA.

Samples with varying thicknesses (1 mm, 2 mm, 5 mm) and an increased wedge size (7 cm) improve spatial resolution and absorption contrast. We implement and compare multiple CNN architectures, carefully tuning hyperparameters and addressing class imbalance through advanced loss functions like focal loss and class-balanced loss.

Our results demonstrate that CNN-based spectral classification, combined with optimized imaging and loss strategies, provides an accurate, scalable solution for automated material identification. This work advances the integration of deep learning and X-ray imaging for consistent, automated materials characterization workflows.


Building upon the general framework for CNN training, this table summarizes specific hyper parameter settings used across various architectures explored in this study. Although the foundational strategies, such as scheduling the learning rate, regularizing, and selecting optimizer, remain consistent, their specific values and combinations are tailored to the demands of each model.
<table>
  <caption>
    Table 1: Hyperparameter configurations used for different CNN and FNN architectures.
  </caption>
  <thead>
    <tr>
      <th>Model</th>
      <th>Optimizer</th>
      <th>Activation</th>
      <th>Learning Rate</th>
      <th>Weight Decay</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>FNN</td>
      <td>SGD</td>
      <td>ReLU</td>
      <td>0.01</td>
      <td>0.001</td>
    </tr>
    <tr>
      <td>ResNet18</td>
      <td>Adam</td>
      <td>ReLU</td>
      <td>0.001</td>
      <td>0.0001</td>
    </tr>
    <tr>
      <td>ResNet34</td>
      <td>SGD</td>
      <td>Leaky ReLU</td>
      <td>0.01</td>
      <td>0.0001</td>
    </tr>
    <tr>
      <td>ResNet50</td>
      <td>RMSprop</td>
      <td>GELU</td>
      <td>0.0005</td>
      <td>0.0005</td>
    </tr>
    <tr>
      <td>VGG16</td>
      <td>SGD + Momentum</td>
      <td>ReLU</td>
      <td>0.01</td>
      <td>0.0001</td>
    </tr>
    <tr>
      <td>EfficientNetB0</td>
      <td>RMSprop / Adam</td>
      <td>Swish (SiLU)</td>
      <td>0.002</td>
      <td>0.0001</td>
    </tr>
  </tbody>
</table>
