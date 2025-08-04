This project explores **X-ray-based materials characterization** using **deep learning** to classify aluminum alloys based on their absorption-driven spectral analysis. Building on prior work with YOLO, we focus on two **aluminum materials—pure aluminum (Al 1050) and aluminum-copper alloy (Al 2017)** imaged with lab-based X-rays at 50 keV and 300 μA.

Samples with varying **thicknesses (1 mm, 2 mm, 5 mm)** and an increased **wedge size (7 cm)** improve spatial resolution and absorption contrast. We implement and compare multiple **CNN architectures**, carefully tuning hyperparameters and addressing class imbalance through advanced loss functions like focal loss and class-balanced loss.

Our results demonstrate that **CNN-based spectral classification**, combined with optimized imaging and loss strategies, provides an accurate, scalable solution for automated material identification. This work advances the integration of deep learning and X-ray imaging for consistent, automated materials characterization workflows.


Building upon the general framework for CNN training, this table summarizes specific **hyper parameter settings** used across various architectures explored in this study. Although the foundational strategies, such as scheduling the learning rate, regularizing, and selecting optimizer, remain consistent, their specific values and combinations are tailored to the demands of each model.
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

**Summary of CNN Model Metrics Performance:**
<table style="border-collapse: collapse; width: 100%; border: 1px solid black;">
    <thead>
        <tr style="background-color: #f2f2f2;">
            <th style="border: 1px solid black; padding: 8px;">Model</th>
            <th style="border: 1px solid black; padding: 8px;">Average Accuracy</th>
            <th style="border: 1px solid black; padding: 8px;">Train Loss</th>
            <th style="border: 1px solid black; padding: 8px;">Validation Loss</th>
            <th style="border: 1px solid black; padding: 8px;">AP (Pure)</th>
            <th style="border: 1px solid black; padding: 8px;">AP (Cu)</th>
            <th style="border: 1px solid black; padding: 8px;">AUC (Avg)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td style="border: 1px solid black; padding: 8px;">ResNet18</td>
            <td style="border: 1px solid black; padding: 8px;">99.4%</td>
            <td style="border: 1px solid black; padding: 8px;">∼0.10</td>
            <td style="border: 1px solid black; padding: 8px;">∼0.15</td>
            <td style="border: 1px solid black; padding: 8px;">0.96</td>
            <td style="border: 1px solid black; padding: 8px;">0.95</td>
            <td style="border: 1px solid black; padding: 8px;">0.96</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 8px;">ResNet50</td>
            <td style="border: 1px solid black; padding: 8px;">99.7%</td>
            <td style="border: 1px solid black; padding: 8px;">∼0.05</td>
            <td style="border: 1px solid black; padding: 8px;">∼0.06</td>
            <td style="border: 1px solid black; padding: 8px;">0.97</td>
            <td style="border: 1px solid black; padding: 8px;">0.96</td>
            <td style="border: 1px solid black; padding: 8px;">0.96</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 8px;">VGG16</td>
            <td style="border: 1px solid black; padding: 8px;">84.0%</td>
            <td style="border: 1px solid black; padding: 8px;">∼0.10</td>
            <td style="border: 1px solid black; padding: 8px;">∼0.20</td>
            <td style="border: 1px solid black; padding: 8px;">0.94</td>
            <td style="border: 1px solid black; padding: 8px;">0.93</td>
            <td style="border: 1px solid black; padding: 8px;">0.94</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 8px;">EfficientNetB0</td>
            <td style="border: 1px solid black; padding: 8px;">98.9%</td>
            <td style="border: 1px solid black; padding: 8px;">∼0.10</td>
            <td style="border: 1px solid black; padding: 8px;">∼0.15</td>
            <td style="border: 1px solid black; padding: 8px;">0.93</td>
            <td style="border: 1px solid black; padding: 8px;">0.93</td>
            <td style="border: 1px solid black; padding: 8px;">0.94</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 8px;">FNN</td>
            <td style="border: 1px solid black; padding: 8px;">84.0%</td>
            <td style="border: 1px solid black; padding: 8px;">∼0.20</td>
            <td style="border: 1px solid black; padding: 8px;">∼0.38</td>
            <td style="border: 1px solid black; padding: 8px;">0.88</td>
            <td style="border: 1px solid black; padding: 8px;">0.91</td>
            <td style="border: 1px solid black; padding: 8px;">0.90</td>
        </tr>
    </tbody>
</table>
