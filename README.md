This repository contains the code for the paper "Reveal the Emergence and Development of Number Sense through Brain-Inspired Spiking Neural Network".

## Directory Structure

- **Stimulus Data Generation**  
    - `data_feneratation.py`: Code for generating stimulus data.

- **Network Implementation**  
    - `learning/init_snn.py`: Initialization of the spiking neural network.
    - `learning/approximate_learning.py`: Approximate learning algorithm implementation.
    - `learning/precise_learning.py`: Precise learning algorithm implementation.

- **Analysis Tools**  
    - `NSN_analyze.py`: Main analysis script for number sense network.
    - `analyze/distance_effect.py`: Analysis of distance effect.
    - `analyze/time_effect.py`: Analysis of time effect.
    - `analyze/weight_analyze.py`: Analysis of network weights.

- **Information Theory**  
    - `lagrange_information_theory.py`: Information theory calculations using Lagrange methods.

## Getting Started

1. Clone this repository:
     ```bash
     git clone https://github.com/yourusername/snn_number_sense.git
     cd snn_number_sense
     ```

2. Install the required dependencies:
     ```bash
     pip install -r requirements.txt
     ```

3. Run the data generation script:
     ```bash
     python data_feneratation.py
     ```

4. Train and analyze the network using the scripts in the `learning` and `analyze` directories. For example:
    ```bash
     python learning/init_snn.py --T 4
     ```
<!-- 
## Citation

If you use this code, please cite our paper:

```
@article{your_paper_citation,
    title={Reveal the Emergence and Development of Number Sense through Brain-Inspired Spiking Neural Network},
    author={Author Names},
    journal={Journal Name},
    year={202X}
}
``` -->

## License

This project is licensed under the Apache2.0 License.