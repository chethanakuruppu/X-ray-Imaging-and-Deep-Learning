This is the initial README file.
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
