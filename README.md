# PyBasie - Psychometric function evaluation using Basie in Python
PyBasie is a Python-based GUI tool for evaluating Speech Reception Thresholds (SRTs) in (reverberant) speech-in-noise conditions. It leverages a Bayesian adaptive method to perform precise psychometric function evaluations, following the methodology outlined in [1]. This tool is designed for researchers and practitioners in auditory and psychophysical studies and was used in the study presented in [to come]

<p float="none" align="middle">
  <img src="docs/gui.png" width="50%" hspace="2%"/>
</p>

> **Figure 1**: Preview of the GUI used for SRT estimation.


## Requirements
- **Python** (get it [here](https://www.python.org/downloads)).
- **Github** (get it [here](https://git-scm.com/download/win)).
- **[Optional] MATLAB** to reproduce the experiments in [to come].

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/ed1016/pybasie.git
   ```
2. Install dependencies: 
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the GUI:
   ```bash
   cd pybasie
   python psycest_gui.py
   ```

## Experiment reproduction [Optional]
Reproduce the experimental setup in [to come] using the MATLAB scripts provided in the `utils` folder:
- `create_babble.m`: Generates the babble noise.
- `generate_data.m`: Creates the reverberant speech-in-noise files.
- `sentence_mapping.txt`: Contains the mapping from audio file to spoken sentence.

## Detailed Features

## To do
- [ ] Give details on GUI usage
- [ ] Run MC simulations (and provide code) to show convergence VS matlab and VS usual methods
- [ ] Write up latex doc with more details about the method 


 ## Citing
 Please cite this work using
 ```
@article{Doire2017,
  title={Robust and efficient Bayesian adaptive psychometric function estimation},
  author={Doire, Clement SJ and Brookes, Mike and Naylor, Patrick A},
  journal={The Journal of the Acoustical Society of America},
  volume={141},
  number={4},
  pages={2501--2512},
  year={2017}
}
```


## References
[1]&nbsp; Doire, C.S., Brookes, M. and Naylor, P.A., 2017. [Robust and efficient Bayesian adaptive psychometric function estimation](https://pubs.aip.org/asa/jasa/article/141/4/2501/1059157). The Journal of the Acoustical Society of America, 141(4), pp.2501-2512.<br>
[2]&nbsp; House, A.S., Williams, C., Hecker, M.H. and Kryter, K.D., 1963. [Psychoacoustic speech tests: A modified rhyme test](https://pubs.aip.org/asa/jasa/article/35/11_Supplement/1899/617588). The Journal of the Acoustical Society of America, 35(11_Supplement), pp.1899-1899.<br>
[3]&nbsp; Cooke, M., Valentini-Botinhao, C. and Mayo, C., 2019. [Hurricane natural speech corpus-higher quality version](https://datashare.ed.ac.uk/handle/10283/347).
 
